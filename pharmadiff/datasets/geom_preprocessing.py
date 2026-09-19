"""
geom_preprocessing.py
Preprocesses the GEOM-Drugs dataset for PharmaDiff.
Downloads raw parquet shard from Hugging Face if needed, parses 3D conformers,
extracts 3D pharmacophores, removes hydrogens, computes dataset statistics, and saves
processed train/val/test splits to data/geom/processed/ matching GeomDrugsDataset requirements.
"""

import os
import pickle
import pathlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

import pharmadiff.datasets.dataset_utils as dataset_utils
from pharmadiff.datasets.dataset_utils import save_pickle
from pharmadiff.metrics.metrics_utils import compute_all_statistics
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore
from pharmadiff.datasets.geom_dataset import full_atom_encoder

HF_PARQUET_URL = "https://huggingface.co/datasets/eamag/drugs-75k/resolve/main/data/train-00000-of-00004.parquet"


def download_raw_parquet(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "geom_drugs_shard0.parquet"
    if not parquet_path.exists():
        print(f"Downloading GEOM-Drugs shard to {parquet_path}...")
        import urllib.request
        urllib.request.urlretrieve(HF_PARQUET_URL, parquet_path)
        print(f"Downloaded ({parquet_path.stat().st_size / (1024*1024):.1f} MB).")
    else:
        print(f"Found cached GEOM-Drugs shard at {parquet_path}.")
    return parquet_path


def process_and_save_split(split_name: str, records: list, out_dir: Path, remove_h: bool = True):
    RDLogger.DisableLog('rdApp.*')
    out_dir.mkdir(parents=True, exist_ok=True)
    h = 'noh' if remove_h else 'h'

    atom_encoder = full_atom_encoder
    if remove_h:
        atom_encoder = {k: v - 1 for k, v in atom_encoder.items() if k != 'H'}

    data_list = []
    mols_list = []
    all_smiles = []
    all_smiles_noh = []

    print(f"Processing split '{split_name}' ({len(records)} molecules)...")
    for rec in tqdm(records, desc=f"Building {split_name}"):
        smiles = rec['smiles']
        conformers_blocks = rec['conformers']
        energies = rec.get('energies', None)

        if len(conformers_blocks) == 0:
            continue

        # Sort by energy if available, otherwise take first
        if energies is not None and len(energies) == len(conformers_blocks):
            sorted_indices = np.argsort(energies)
        else:
            sorted_indices = list(range(len(conformers_blocks)))

        # Take the lowest energy conformer
        chosen_idx = sorted_indices[0]
        mol_block = conformers_blocks[chosen_idx]

        try:
            conformer = Chem.MolFromMolBlock(mol_block, removeHs=False)
            if conformer is None:
                continue
            Chem.SanitizeMol(conformer)
            Chem.Kekulize(conformer)
        except Exception:
            continue

        try:
            data, pos_mean = dataset_utils.mol_to_torch_geometric(conformer, full_atom_encoder, smiles)
            pharmacophore = mol_to_torch_pharmacophore(conformer, pos_mean, name='geom')
            if pharmacophore is None:
                continue

            if remove_h:
                data, pharmacophore = dataset_utils.remove_hydrogens(data, pharmacophore)

            data_list.append({"ligand": data, "pharmacophore": pharmacophore})
            mols_list.append(data)
            all_smiles.append(smiles)

            try:
                mol_no_h = Chem.RemoveHs(conformer, sanitize=False)
                if mol_no_h is not None:
                    s_noh = Chem.MolToSmiles(mol_no_h)
                    if s_noh is not None:
                        all_smiles_noh.append(s_noh)
            except Exception:
                pass
        except Exception:
            continue

    print(f"Successfully processed {len(data_list)} valid items for split '{split_name}'.")
    assert len(data_list) > 0, f"No valid data processed for {split_name}!"

    # Collate and save .pt
    ligands = [item['ligand'] for item in data_list]
    pharmacophores = [item['pharmacophore'] for item in data_list]
    collated_data = {'ligand': ligands, 'pharmacophore': pharmacophores}
    pt_path = out_dir / f"{split_name}_{h}.pt"
    torch.save(collated_data, pt_path)
    print(f"Saved {pt_path}")

    # Compute statistics
    print(f"Computing dataset statistics for '{split_name}'...")
    statistics = compute_all_statistics(mols_list, atom_encoder,
                                        charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5})

    save_pickle(statistics.num_nodes, out_dir / f"{split_name}_n_{h}.pickle")
    np.save(out_dir / f"{split_name}_atom_types_{h}.npy", statistics.atom_types)
    np.save(out_dir / f"{split_name}_bond_types_{h}.npy", statistics.bond_types)
    np.save(out_dir / f"{split_name}_charges_{h}.npy", statistics.charge_types)
    save_pickle(statistics.valencies, out_dir / f"{split_name}_valency_{h}.pickle")
    save_pickle(statistics.bond_lengths, out_dir / f"{split_name}_bond_lengths_{h}.pickle")
    np.save(out_dir / f"{split_name}_angles_{h}.npy", statistics.bond_angles)
    save_pickle(set(all_smiles), out_dir / f"{split_name}_smiles.pickle")
    save_pickle(set(all_smiles_noh), out_dir / f"goem_{split_name}_smiles_noh.pickle")
    print(f"Completed saving all statistics for '{split_name}' in {out_dir}")


def prepare_geom_dataset(data_dir: Path, n_train: int = 400, n_val: int = 50, n_test: int = 50,
                         remove_h: bool = True, seed: int = 42):
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    parquet_path = download_raw_parquet(raw_dir)

    print(f"Loading parquet {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    total_needed = n_train + n_val + n_test
    print(f"Dataset has {len(df)} molecules. Subsetting {total_needed} molecules...")

    # Shuffle deterministically
    df_sample = df.sample(n=min(len(df), total_needed + 100), random_state=seed).reset_index(drop=True)

    records = []
    for idx, row in df_sample.iterrows():
        records.append({
            'smiles': row['smiles'],
            'conformers': row['conformers'],
            'energies': row.get('conformer_energies', None)
        })

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:n_train + n_val + n_test]

    print(f"Splits allocated: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")
    process_and_save_split("train", train_records, processed_dir, remove_h=remove_h)
    process_and_save_split("val", val_records, processed_dir, remove_h=remove_h)
    process_and_save_split("test", test_records, processed_dir, remove_h=remove_h)
    print("\n>>> GEOM dataset preparation complete! Ready for training and evaluation. <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/venkatgb/keshaw/PharmaDiff/data/geom")
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=50)
    args = parser.parse_args()

    prepare_geom_dataset(Path(args.data_dir), n_train=args.n_train, n_val=args.n_val, n_test=args.n_test)
