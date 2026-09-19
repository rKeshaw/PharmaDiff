#!/usr/bin/env python3
"""
P²Diff — Standalone Inference Script
=====================================
Generates molecules conditioned on an arbitrary PDB pocket + SDF pharmacophore.

Usage:
  python inference.py \\
      --pdb pocket.pdb \\
      --pharma pharmacophore.sdf \\
      --n_poses 10 \\
      --out_dir results/ \\
      [--device cuda:0] \\
      [--steps 500]

Input format:
  --pdb     : PDB file with the binding pocket residues (any standard PDB format)
  --pharma  : SDF file with pharmacophore features (one molecule per feature;
              atom types encode feature types: D=donor, A=acceptor, H=hydrophobic,
              R=aromatic, P=positive, N=negative)

Output format:
  results/
    generated_mols.sdf     — all K generated molecules (SDF format)
    generated_mols.pkl     — list of RDKit Mol objects
    summary.json           — per-pose metrics (validity, SMILES, MW, QED, SAS)
"""

import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

# Workspace-relative imports
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

# Hydra for config
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# PharmaDiff model
from pharmadiff.diffusion_model import FullDenoisingDiffusion
from pharmadiff.datasets.pharmacophore import dataset_info_from_config
import pharmadiff.utils as utils


# ══════════════════════════════════════════════════════════════════════════════
# Config & Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CKPT = os.path.join(WORKSPACE, "checkpoints/plinder-scaled-pocket_resume/best.ckpt")
DEFAULT_CONFIG_DIR = os.path.join(WORKSPACE, "configs")


def load_cfg(config_dir=DEFAULT_CONFIG_DIR):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config",
                      overrides=["+experiment=plinder_scaled_pocket.yaml"])
    return cfg


def load_model(ckpt_path, cfg, device):
    """Load P²Diff model from checkpoint."""
    print(f"  Loading checkpoint: {os.path.basename(ckpt_path)}")
    dataset_infos = dataset_info_from_config(cfg)
    model = FullDenoisingDiffusion(cfg, dataset_infos, train_smiles=[])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    print(f"  Model loaded on {device}")
    return model, dataset_infos


# ══════════════════════════════════════════════════════════════════════════════
# Pocket Parsing
# ══════════════════════════════════════════════════════════════════════════════

ATOM_TYPES = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'P': 5,
              'Cl': 6, 'Br': 7, 'I': 8, 'B': 9, 'Si': 10}
DEFAULT_ATOM_TYPE = 0  # Carbon fallback


def parse_pdb_pocket(pdb_path):
    """
    Parse a PDB file to extract pocket atom positions and feature vectors.
    Returns:
        pos  (Tensor, [N, 3]) : 3D coordinates (Å)
        feat (Tensor, [N, F]) : one-hot atom type features
    """
    positions = []
    atom_types = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip()
            if not element:
                # Fall back to columns 12-16 atom name
                atom_name = line[12:16].strip()
                element = ''.join(c for c in atom_name if c.isalpha())[:2]
            element = element.capitalize()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            positions.append([x, y, z])
            atom_types.append(ATOM_TYPES.get(element, DEFAULT_ATOM_TYPE))

    if not positions:
        raise ValueError(f"No ATOM/HETATM records found in {pdb_path}")

    pos  = torch.tensor(positions, dtype=torch.float)
    n    = len(atom_types)
    feat = torch.zeros(n, len(ATOM_TYPES), dtype=torch.float)
    for i, at in enumerate(atom_types):
        feat[i, at] = 1.0

    print(f"  Pocket: {n} atoms parsed from {os.path.basename(pdb_path)}")
    return pos, feat


# ══════════════════════════════════════════════════════════════════════════════
# Pharmacophore Parsing
# ══════════════════════════════════════════════════════════════════════════════

PHARMA_FEATURE_TYPES = {
    'D': 0, 'DA': 0,  # Donor / DonorAcceptor
    'A': 1,           # Acceptor
    'H': 2,           # Hydrophobic
    'R': 3, 'AR': 3,  # Aromatic
    'P': 4,           # Positive ionisable
    'N': 5,           # Negative ionisable
}
N_PHARMA_FEAT = 6


def parse_pharmacophore_sdf(sdf_path):
    """
    Parse pharmacophore SDF. Each molecule record encodes one feature:
      - Atom 0: centroid (position)
      - Atom 1 (if present): direction vector endpoint
      - Molecule name / property 'PHARMA_TYPE' encodes feature type
    Returns:
        pharma_pos   (Tensor, [M, 3])
        pharma_feat  (Tensor, [M, N_PHARMA_FEAT])
        pharma_dir   (Tensor, [M, 3]) — unit vectors (zeros if no direction)
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    positions, feats, directions = [], [], []

    for mol in suppl:
        if mol is None:
            continue
        conf = mol.GetConformer()
        centroid = np.array(conf.GetAtomPosition(0))

        # Feature type from name or property
        feat_type_str = mol.GetProp('PHARMA_TYPE') if mol.HasProp('PHARMA_TYPE') \
                        else mol.GetProp('_Name') if mol.HasProp('_Name') else 'H'
        feat_type_str = str(feat_type_str).strip().upper()
        feat_idx = PHARMA_FEATURE_TYPES.get(feat_type_str, 2)  # default hydrophobic

        # Direction
        direction = np.zeros(3)
        if mol.GetNumAtoms() >= 2:
            tip = np.array(conf.GetAtomPosition(1))
            d   = tip - centroid
            norm = np.linalg.norm(d)
            if norm > 1e-3:
                direction = d / norm

        one_hot = np.zeros(N_PHARMA_FEAT)
        one_hot[feat_idx] = 1.0

        positions.append(centroid)
        feats.append(one_hot)
        directions.append(direction)

    if not positions:
        raise ValueError(f"No pharmacophore features found in {sdf_path}")

    print(f"  Pharmacophore: {len(positions)} features parsed from {os.path.basename(sdf_path)}")
    return (
        torch.tensor(np.array(positions),  dtype=torch.float),
        torch.tensor(np.array(feats),      dtype=torch.float),
        torch.tensor(np.array(directions), dtype=torch.float),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Condition Construction
# ══════════════════════════════════════════════════════════════════════════════

def build_condition(pocket_pos, pocket_feat, pharma_pos, pharma_feat, pharma_dir,
                    cfg, dataset_infos, device):
    """
    Construct a PlaceHolder condition for a single complex.
    """
    # Pocket dict matching DataLoader format
    pocket_dict = {
        'pos':     pocket_pos.unsqueeze(0).to(device),   # [1, N, 3]
        'x':       pocket_feat.unsqueeze(0).to(device),  # [1, N, F]
        'res_idx': torch.zeros(pocket_pos.shape[0],      # dummy
                               dtype=torch.long).unsqueeze(0).to(device),
    }

    # Pharma condition
    pharma_cond = utils.PlaceHolder(
        X=pharma_feat.unsqueeze(0).to(device),        # [1, M, F]
        pos=pharma_pos.unsqueeze(0).to(device),        # [1, M, 3]
        charges=torch.zeros(1, pharma_feat.shape[0],   # [1, M, 1]
                            1, device=device),
        E=None, y=None,
    )

    # Single-item dense batch (molecule graph will be built by model internally)
    # Use a dummy molecule graph — model will generate from scratch
    n_nodes_guess = cfg.dataset.max_n_nodes if hasattr(cfg.dataset, 'max_n_nodes') else 30
    dense_batch = utils.PlaceHolder(
        X=torch.zeros(1, n_nodes_guess,
                      dataset_infos.num_atom_types, device=device),
        E=torch.zeros(1, n_nodes_guess, n_nodes_guess,
                      dataset_infos.num_edge_types, device=device),
        pos=torch.zeros(1, n_nodes_guess, 3, device=device),
        charges=torch.zeros(1, n_nodes_guess, 1, device=device),
        y=torch.zeros(1, 0, device=device),
        node_mask=torch.ones(1, n_nodes_guess, dtype=torch.bool, device=device),
    )

    sample_cond = utils.PlaceHolder(
        X=pharma_cond.X,
        pos=pharma_cond.pos,
        charges=pharma_cond.charges,
        E=None, y=None,
        pocket_pos=pocket_dict['pos'],
        pocket_x=pocket_dict['x'],
    )

    return dense_batch, sample_cond


# ══════════════════════════════════════════════════════════════════════════════
# Molecule Metrics
# ══════════════════════════════════════════════════════════════════════════════

def mol_metrics(mol):
    if mol is None:
        return {'valid': False, 'smiles': None, 'mw': None, 'qed': None, 'sas': None}
    try:
        smiles = Chem.MolToSmiles(mol)
        mw     = Descriptors.MolWt(mol)
        qed    = QED.qed(mol)
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from sascorer import calculateScore  # type: ignore[import]
        sas = calculateScore(mol)
    except Exception:
        sas = None
    return {'valid': True, 'smiles': smiles, 'mw': mw, 'qed': qed, 'sas': sas}


# ══════════════════════════════════════════════════════════════════════════════
# Main Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(pdb_path, pharma_path, n_poses, out_dir, device_str, steps):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Load config & model
    print("\n[1/5] Loading configuration …")
    cfg = load_cfg()
    if steps != 500:
        ratio = 500 // steps
        cfg.model.general.faster_sampling = ratio
        print(f"  Using {steps} effective steps (faster_sampling={ratio})")

    print("\n[2/5] Loading model checkpoint …")
    model, dataset_infos = load_model(DEFAULT_CKPT, cfg, device)

    # Parse inputs
    print("\n[3/5] Parsing inputs …")
    pocket_pos, pocket_feat = parse_pdb_pocket(pdb_path)
    pharma_pos, pharma_feat, pharma_dir = parse_pharmacophore_sdf(pharma_path)

    # Build condition
    print("\n[4/5] Building condition …")
    dense_batch, sample_cond = build_condition(
        pocket_pos, pocket_feat, pharma_pos, pharma_feat, pharma_dir,
        cfg, dataset_infos, device
    )

    # Sample
    print(f"\n[5/5] Sampling {n_poses} poses …")
    with torch.no_grad():
        samples = model.sample_n_graphs(
            samples_to_generate=n_poses,
            chains_to_save=0,
            samples_to_save=0,
            batch_id=0,
            test=True,
            sample_condition=sample_cond,
        )

    # Collect & evaluate
    mols   = []
    metrics = []
    writer = Chem.SDWriter(os.path.join(out_dir, "generated_mols.sdf"))
    for i, s in enumerate(samples):
        mol = getattr(s, 'rdkit_mol', None)
        if mol is not None:
            mols.append(mol)
            m = mol_metrics(mol)
            m['pose_id'] = i
            metrics.append(m)
            writer.write(mol)
            print(f"  Pose {i:3d}: {m['smiles'][:60] if m['smiles'] else '—'}  "
                  f"Vina=N/A  QED={m['qed']:.3f}" if m.get('qed') else f"  Pose {i}: invalid")
        else:
            metrics.append({'pose_id': i, 'valid': False})
    writer.close()

    with open(os.path.join(out_dir, "generated_mols.pkl"), 'wb') as f:
        pickle.dump(mols, f)
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    n_valid = sum(m.get('valid', False) for m in metrics)
    print(f"\n{'='*55}")
    print(f"  Done. {n_valid}/{n_poses} valid molecules generated.")
    print(f"  Output directory: {out_dir}")
    print(f"  SDF: {os.path.join(out_dir, 'generated_mols.sdf')}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P²Diff: Structure-Aware Molecular Generation Inference"
    )
    parser.add_argument("--pdb",     required=True,  help="Pocket PDB file")
    parser.add_argument("--pharma",  required=True,  help="Pharmacophore SDF file")
    parser.add_argument("--n_poses", type=int, default=10,
                        help="Number of poses to generate (default: 10)")
    parser.add_argument("--out_dir", default="results/inference",
                        help="Output directory (default: results/inference)")
    parser.add_argument("--device",  default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--steps",   type=int, default=500,
                        help="Diffusion steps (500=full, 250=2x faster, 100=5x faster)")
    parser.add_argument("--ckpt",    default=DEFAULT_CKPT,
                        help=f"Checkpoint path (default: {DEFAULT_CKPT})")
    args = parser.parse_args()

    run_inference(
        pdb_path=os.path.abspath(args.pdb),
        pharma_path=os.path.abspath(args.pharma),
        n_poses=args.n_poses,
        out_dir=os.path.abspath(args.out_dir),
        device_str=args.device,
        steps=args.steps,
    )
