"""
pharmadiff/datasets/plinder_dataset.py
=======================================
Drop-in replacement for GeomDrugsDataset backed by PLINDER ligands.

Pipeline fix vs previous version
----------------------------------
The previous version called Chem.RemoveHs() in preprocessing before saving
the pickle. This caused a subtle pipeline deviation:
  - GEOM: conformer WITH H -> mol_to_torch_geometric + mol_to_torch_pharmacophore
          -> remove_hydrogens() does real work (masks H, re-centers on heavy atoms)
  - Plinder (old): conformer WITHOUT H -> both functions -> remove_hydrogens()
          was effectively a no-op (no H to remove), but still did x-=1 shift
          and recalculated mean_pos on already-centered coords.

While the final tensor values were numerically equivalent, this deviated from
the tested GEOM code path and made it harder to reason about correctness.

This version saves mols WITH H (matching GEOM exactly). The process() method
is now identical to GeomDrugsDataset.process() — no special cases.

Integration
-----------
1. Place this file at pharmadiff/datasets/plinder_dataset.py
2. In pharmadiff/main.py, add 'plinder' branch alongside 'geom':
       elif cfg.dataset.name == 'plinder':
           from pharmadiff.datasets.plinder_dataset import PlinderDataModule, PlinderInfos
           datamodule = PlinderDataModule(cfg)
           dataset_infos = PlinderInfos(datamodule, cfg)
3. Place configs/dataset/plinder.yaml in PharmaDiff/configs/dataset/
4. Place configs/experiment/plinder_no_h_adaptive.yaml in PharmaDiff/configs/experiment/
5. Training command:
       python main.py dataset=plinder dataset.remove_h=True +experiment=plinder_no_h_adaptive
"""

import os
import pathlib
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from hydra.utils import get_original_cwd

import pharmadiff.datasets.dataset_utils as dataset_utils
from pharmadiff.datasets.dataset_utils import load_pickle, save_pickle
from pharmadiff.datasets.abstract_dataset import (
    AbstractDatasetInfos,
    AbstractAdaptiveDataModule,
)
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore
from pharmadiff.metrics.metrics_utils import compute_all_statistics
from pharmadiff.utils import PlaceHolder


# Same atom vocabulary as GEOM-DRUGS
full_atom_encoder = {
    "H":  0, "B":  1, "C":  2, "N":  3, "O":  4,
    "F":  5, "Al": 6, "Si": 7, "P":  8, "S":  9,
    "Cl": 10, "As": 11, "Br": 12, "I":  13,
    "Hg": 14, "Bi": 15,
}

# Import FAMILY_MAPPING for output_dims (same as GEOM)
from pharmadiff.datasets.pharmacophore_utils import FAMILY_MAPPING


class PlinderDataset(InMemoryDataset):
    """
    PyG InMemoryDataset backed by PLINDER ligands.
    process() is identical to GeomDrugsDataset.process().
    """

    def __init__(self, split, root, remove_h,
                 transform=None, pre_transform=None, pre_filter=None):
        assert split in ["train", "val", "test"]
        self.split    = split
        self.remove_h = remove_h
        # "geom" activates the 3-7 point pharmacophore subsampling distribution
        # calibrated for drug-like molecules — appropriate for PLINDER's
        # filtered drug-like ligand set
        self.name = "geom"

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)
        self.full_data_dict = torch.load(self.processed_paths[0], weights_only=False)

        self.ligand        = self.full_data_dict["ligand"]
        self.pharmacophore = self.full_data_dict["pharmacophore"]

        self.statistics = dataset_utils.Statistics(
            num_nodes    = load_pickle(self.processed_paths[1]),
            atom_types   = torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types   = torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types = torch.from_numpy(np.load(self.processed_paths[4])),
            valencies    = load_pickle(self.processed_paths[5]),
            bond_lengths = load_pickle(self.processed_paths[6]),
            bond_angles  = torch.from_numpy(np.load(self.processed_paths[7])),
        )
        self.smiles = load_pickle(self.processed_paths[8])

    def __len__(self):
        return len(self.ligand)

    def __getitem__(self, idx):
        return {"ligand": self.ligand[idx], "pharmacophore": self.pharmacophore[idx]}

    @property
    def raw_file_names(self):
        return {
            "train": ["train_data.pickle"],
            "val":   ["val_data.pickle"],
            "test":  ["test_data.pickle"],
        }[self.split]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        b = self.split
        return [
            f"{b}_{h}.pt",
            f"{b}_n_{h}.pickle",
            f"{b}_atom_types_{h}.npy",
            f"{b}_bond_types_{h}.npy",
            f"{b}_charges_{h}.npy",
            f"{b}_valency_{h}.pickle",
            f"{b}_bond_lengths_{h}.pickle",
            f"{b}_angles_{h}.npy",
            f"{b}_smiles.pickle",
            f"plinder_{b}_smiles_{h}.pickle",
        ]

    def download(self):
        raise ValueError(
            "PLINDER data must be preprocessed with plinder_preprocessing.py. "
            f"Expected raw files in: {self.raw_dir}"
        )

    def process(self):
        """
        Identical to GeomDrugsDataset.process().
        Mols in the pickle have H — remove_hydrogens() handles H removal
        and re-centering exactly as in GEOM.
        """
        RDLogger.DisableLog("rdApp.*")
        all_data = load_pickle(self.raw_paths[0])

        data_list    = []
        mols_list    = []
        all_smiles   = []
        all_smiles_noh = []
        skipped      = 0

        for smiles, conformers in tqdm(all_data, desc=f"Processing {self.split}"):
            all_smiles.append(smiles)

            # Only the first (and only) conformer — crystal bound pose
            for conformer in conformers[:1]:
                try:
                    Chem.SanitizeMol(conformer)
                    Chem.Kekulize(conformer)
                except Exception:
                    skipped += 1
                    continue

                data, pos_mean = dataset_utils.mol_to_torch_geometric(
                    conformer, full_atom_encoder, smiles
                )
                pharmacophore = mol_to_torch_pharmacophore(
                    conformer, pos_mean, name=self.name
                )
                if pharmacophore is None:
                    skipped += 1
                    continue

                if self.remove_h:
                    # Identical to GEOM: conformer has H, remove_hydrogens
                    # strips them and re-centers on heavy atoms
                    data, pharmacophore = dataset_utils.remove_hydrogens(
                        data, pharmacophore
                    )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append({"ligand": data, "pharmacophore": pharmacophore})
                mols_list.append(data)

                try:
                    mol_noh = Chem.RemoveHs(conformer, sanitize=False)
                    if mol_noh is not None:
                        smi_noh = Chem.MolToSmiles(mol_noh)
                        if smi_noh:
                            all_smiles_noh.append(smi_noh)
                except Exception:
                    pass

        print(f"Processed {len(data_list):,}  |  skipped {skipped:,}")

        torch.save(self._collate(data_list), self.processed_paths[0])

        stats = compute_all_statistics(
            mols_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        )
        save_pickle(stats.num_nodes,    self.processed_paths[1])
        np.save(self.processed_paths[2], stats.atom_types)
        np.save(self.processed_paths[3], stats.bond_types)
        np.save(self.processed_paths[4], stats.charge_types)
        save_pickle(stats.valencies,    self.processed_paths[5])
        save_pickle(stats.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], stats.bond_angles)
        save_pickle(set(all_smiles),    self.processed_paths[8])
        save_pickle(set(all_smiles_noh), self.processed_paths[9])

    @staticmethod
    def _collate(data_list):
        return {
            "ligand":        [d["ligand"]        for d in data_list],
            "pharmacophore": [d["pharmacophore"] for d in data_list],
        }


class PlinderDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        train = PlinderDataset(split="train", root=root_path, remove_h=cfg.dataset.remove_h)
        val   = PlinderDataset(split="val",   root=root_path, remove_h=cfg.dataset.remove_h)
        test  = PlinderDataset(split="test",  root=root_path, remove_h=cfg.dataset.remove_h)

        self.remove_h   = cfg.dataset.remove_h
        self.statistics = {
            "train": train.statistics,
            "val":   val.statistics,
            "test":  test.statistics,
        }
        super().__init__(cfg, train, val, test)


class PlinderInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h         = cfg.dataset.remove_h
        self.need_to_strip    = False
        self.statistics       = datamodule.statistics
        self.name             = "geom"
        self.atom_encoder     = full_atom_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()

        if self.remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }
            self.atom_decoder = [k for k in full_atom_encoder if k != "H"]
        else:
            self.atom_decoder = list(full_atom_encoder.keys())

        # Identical to GeomInfos — must call complete_infos with statistics
        # and atom_encoder so n_nodes, atom_types, edge_types are populated
        super().complete_infos(self.statistics, self.atom_encoder)

        self.input_dims = PlaceHolder(
            X=self.num_atom_types, charges=6, E=5, y=1, pos=3,
            pharma_feat=len(FAMILY_MAPPING), pharma_coord=3
        )
        self.output_dims = PlaceHolder(
            X=self.num_atom_types, charges=6, E=5, y=0, pos=3,
            pharma_feat=len(FAMILY_MAPPING), pharma_coord=3
        )

    def to_one_hot(self, X, charges, E, node_mask):
        X       = F.one_hot(X, num_classes=self.num_atom_types).float()
        E       = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,
                                  y=None, pos=None,
                                  pharma_coord=None, pharma_feat=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()

    def pharma_to_one_hot(self, X, pharma_mask):
        X = F.one_hot(X, num_classes=len(FAMILY_MAPPING) + 1).float()
        X = X[:, :, 1:] * pharma_mask.unsqueeze(-1)
        return X