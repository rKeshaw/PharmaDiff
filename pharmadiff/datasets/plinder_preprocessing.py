"""
plinder_preprocessing.py
========================
Preprocesses PLINDER (2024-06 / v2) into PharmaDiff-compatible pickle files.

Output (in --output_dir/):
    train_data.pickle
    val_data.pickle
    test_data.pickle

Each entry: (canonical_smiles: str, [RDKit Mol WITH hydrogens])

CRITICAL: Mols are saved WITH hydrogens, matching GEOM exactly.
PlinderDataset.process() calls remove_hydrogens() internally, just like
GeomDrugsDataset — this is the correct and tested code path.

Saving mols WITHOUT H (as the previous version did) caused a pipeline
difference: pharmacophore_to_torch received n=n_heavy instead of n=n_total,
and remove_hydrogens() was a no-op instead of doing the proper index alignment.
The encoding was numerically equivalent but it deviates from the tested GEOM path.

Root causes of low pharmacophore match score (now fixed):
  1. Cofactor contamination: ATP/NAD+/CoA etc. are ligand_is_proper=True
     but have completely different pharmacophore profiles from drug-like molecules.
     Fix: system_ligand_has_cofactor == False
  2. Oligo-peptide/saccharide/nucleotide contamination.
     Fix: ligand_is_oligo_peptide/saccharide/nucleotide == False
  3. NMR and cryo-EM structures: no proper crystallographic bound pose.
     Fix: entry_determination_method == "X-RAY DIFFRACTION"
  4. Dataset too small (28k vs GEOM's 430k): not a data quality bug but
     requires more epochs. Fix: plinder experiment config with adjusted epochs.

Usage
-----
    # Dry run first (annotation stats only, ~1-2 min):
    python plinder_preprocessing.py \\
        --plinder_dir ~/.local/share/plinder/2024-06/v2 \\
        --output_dir /path/to/PharmaDiff/data/plinder/raw \\
        --dry_run

    # Full run (systems must already be extracted):
    python plinder_preprocessing.py \\
        --plinder_dir ~/.local/share/plinder/2024-06/v2 \\
        --output_dir /path/to/PharmaDiff/data/plinder/raw \\
        --n_workers 8
"""

import argparse
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality filter thresholds
# ---------------------------------------------------------------------------
MAX_RESOLUTION     = 2.5    # Angstroms
MIN_LIGAND_MW      = 150.0  # Da
MAX_LIGAND_MW      = 800.0  # Da
MAX_LIGAND_HEAVY_N = 80     # atoms — memory-safe cap for n^2 edge tensor

# X-ray only: NMR and cryo-EM structures do not have a reliable
# crystallographic bound pose — the ligand geometry is less well-defined.
XRAY_METHOD = "X-RAY DIFFRACTION"

# Atom types supported by PharmaDiff (GEOM-DRUGS vocabulary).
ALLOWED_ATOMS = {
    'H', 'B', 'C', 'N', 'O', 'F',
    'Al', 'Si', 'P', 'S', 'Cl',
    'As', 'Br', 'I', 'Hg', 'Bi',
}


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

ANNOTATION_COLUMNS = [
    "system_id",
    "entry_pdb_id",
    "entry_resolution",
    "entry_determination_method",       # NEW: X-ray only
    "system_pass_validation_criteria",
    "system_num_ligand_chains",
    "system_ligand_has_cofactor",       # NEW: exclude cofactors
    "ligand_is_proper",
    "ligand_is_oligo",          
    "ligand_molecular_weight",
    "ligand_rdkit_canonical_smiles",
    "ligand_instance_chain",
    "ligand_is_covalent",
    "ligand_is_fragment",
    "ligand_is_invalid",
    "ligand_is_rdkit_loadable",
    "ligand_posebusters_sanitization",
    "ligand_posebusters_mol_pred_loaded",
]

SPLIT_COLUMNS = ["system_id", "split", "system_pass_statistics_criteria"]


# ---------------------------------------------------------------------------
# Step 1 — Annotation filtering
# ---------------------------------------------------------------------------

def load_and_filter_annotations(plinder_dir: Path) -> pd.DataFrame:
    """
    Loads annotation_table.parquet with predicate pushdown and applies all
    quality filters including the new cofactor/oligo/determination filters.
    """
    ann_path   = plinder_dir / "index"  / "annotation_table.parquet"
    split_path = plinder_dir / "splits" / "split.parquet"

    for p in [ann_path, split_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    log.info("Loading annotation table (predicate pushdown active)...")

    # All boolean/numeric filters pushed to parquet row-group level.
    # String equality filters (entry_determination_method) cannot be pushed
    # down in all pyarrow versions — applied in-memory after load.
    pushdown_filters = [[
        ("ligand_is_proper",                   "==", True),
        ("system_pass_validation_criteria",    "==", True),
        ("entry_resolution",                   "<=", MAX_RESOLUTION),
        ("ligand_molecular_weight",            ">=", MIN_LIGAND_MW),
        ("ligand_molecular_weight",            "<=", MAX_LIGAND_MW),
        ("ligand_is_covalent",                 "==", False),
        ("ligand_is_fragment",                 "==", False),
        ("ligand_is_invalid",                  "==", False),
        ("ligand_is_rdkit_loadable",           "==", True),
        ("ligand_posebusters_sanitization",    "==", True),
        ("ligand_posebusters_mol_pred_loaded", "==", True),
        # New filters pushed down (boolean columns):
        ("system_ligand_has_cofactor",         "==", False),
        ("ligand_is_oligo",                    "==", False),
    ]]

    df = pd.read_parquet(
        ann_path,
        columns=ANNOTATION_COLUMNS,
        filters=pushdown_filters,
        engine="pyarrow",
    )
    log.info(f"After boolean/numeric pushdown filters: {len(df):,} rows")

    # String filter: X-ray only — applied in-memory
    df = df[df["entry_determination_method"] == XRAY_METHOD]
    log.info(f"After X-ray-only filter: {len(df):,} rows")

    # ---- Split labels ----
    log.info("Loading split table...")
    df_split = pd.read_parquet(split_path, columns=SPLIT_COLUMNS, engine="pyarrow")
    df_split = df_split[
        (df_split["split"] != "removed") &
        (df_split["system_pass_statistics_criteria"] == True)
    ]
    log.info(f"Split table after quality gates: {len(df_split):,} rows")

    df = df.merge(df_split[["system_id", "split"]], on="system_id", how="inner")
    log.info(f"After merge with split table: {len(df):,} rows")

    df = df[df["ligand_rdkit_canonical_smiles"].notna()]
    df = df[df["ligand_rdkit_canonical_smiles"].str.strip() != ""]
    log.info(f"After SMILES non-null check: {len(df):,} rows")

    return df


# ---------------------------------------------------------------------------
# Step 2 — Deduplicate to one ligand per system
# ---------------------------------------------------------------------------

def deduplicate_to_one_ligand_per_system(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Deduplicating to one ligand per system (heaviest by MW)...")
    primary = (
        df.sort_values("ligand_molecular_weight", ascending=False)
          .drop_duplicates(subset=["system_id"], keep="first")
          .reset_index(drop=True)
    )
    log.info(f"After deduplication: {len(primary):,} unique systems")
    split_counts = primary["split"].value_counts()
    log.info("Final split distribution:")
    for s in ["train", "val", "test"]:
        log.info(f"  {s:>6}: {split_counts.get(s, 0):>8,}")
    return primary


# ---------------------------------------------------------------------------
# Step 3 — SDF loading from extracted filesystem
# ---------------------------------------------------------------------------

def _find_system_dir(plinder_dir: Path, system_id: str) -> Optional[Path]:
    """Resolves system directory whether stored flat or nested by two-char code."""
    flat = plinder_dir / "systems" / system_id
    if flat.is_dir():
        return flat
    pdb_id = system_id.split("__")[0].split("_")[0]
    if len(pdb_id) >= 3:
        two_char = pdb_id[1:3].lower()
        nested = plinder_dir / "systems" / two_char / system_id
        if nested.is_dir():
            return nested
    return None


def _read_sdf(plinder_dir: Path, system_id: str, ligand_chain: str) -> Optional[str]:
    """
    Reads SDF from extracted filesystem.
    Path: {plinder_dir}/systems/[{two_char}/]{system_id}/ligand_files/{ligand_chain}.sdf
    """
    sys_dir = _find_system_dir(plinder_dir, system_id)
    if sys_dir is None:
        return None
    sdf_path = sys_dir / "ligand_files" / f"{ligand_chain}.sdf"
    if not sdf_path.exists():
        return None
    try:
        return sdf_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _parse_mol(sdf_str: str) -> Optional[Chem.Mol]:
    """
    Parses SDF string into sanitized RDKit Mol WITH hydrogens and 3D coords.

    IMPORTANT: We keep H in the mol saved to pickle.
    PlinderDataset.process() calls remove_hydrogens() internally (if remove_h=True),
    exactly matching the GEOM pipeline where conformers also have H.
    """
    try:
        suppl = Chem.SDMolSupplier()
        suppl.SetData(sdf_str, removeHs=False, sanitize=False)
        mol = next(suppl, None)
        if mol is None:
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=(
                        Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    ),
                )
                Chem.Kekulize(mol, clearAromaticFlags=False)
            except Exception:
                return None

        if mol.GetNumConformers() == 0:
            return None

        # Check heavy atom count (H excluded from cap)
        if mol.GetNumHeavyAtoms() > MAX_LIGAND_HEAVY_N:
            return None

        # Validate heavy atom vocabulary
        heavy_symbols = {a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 1}
        if not heavy_symbols.issubset(ALLOWED_ATOMS):
            unsupported = heavy_symbols - ALLOWED_ATOMS
            log.debug(f"Skipping mol with unsupported atoms: {unsupported}")
            return None

        return mol   # WITH H — matching GEOM convention

    except Exception:
        return None


STANDARD_AMINO_ACIDS = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}  # 20 is 'OTHER'


def _extract_pocket(plinder_dir: Path, system_id: str, ligand_mol: Chem.Mol,
                    cutoff: float = 8.0, max_atoms: int = 150) -> Optional[dict]:
    """
    Extracts pocket heavy atoms from receptor.pdb within `cutoff` Angstroms
    of any ligand heavy atom.
    Path: {plinder_dir}/systems/{system_id}/receptor.pdb
    Returns:
        {'pos': np.ndarray (M, 3), 'res_idx': np.ndarray (M,)} or None
    """
    sys_dir = _find_system_dir(plinder_dir, system_id)
    if sys_dir is None:
        return None
    receptor_pdb = sys_dir / "receptor.pdb"
    if not receptor_pdb.exists():
        return None

    try:
        ligand_conf = ligand_mol.GetConformer()
        lig_pos = ligand_conf.GetPositions()
        heavy_indices = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() != 1]
        if not heavy_indices:
            return None
        lig_heavy_pos = lig_pos[heavy_indices]

        atom_coords = []
        atom_res = []

        with open(receptor_pdb, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM  "):
                    res_name = line[17:20].strip()
                    if res_name in {"HOH", "WAT", "H2O", "DOD", "SO4", "PO4", "ACT"}:
                        continue
                    element = line[76:78].strip().upper() if len(line) >= 78 else ""
                    if not element:
                        atom_name = line[12:16].strip()
                        element = atom_name[0] if atom_name else "C"
                    if element == "H":
                        continue

                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue

                    res_idx = STANDARD_AMINO_ACIDS.get(res_name, 20)
                    atom_coords.append([x, y, z])
                    atom_res.append(res_idx)

        if not atom_coords:
            return None

        atom_coords = np.array(atom_coords, dtype=np.float32)
        dists = np.linalg.norm(atom_coords[:, None, :] - lig_heavy_pos[None, :, :], axis=-1)
        min_dists = dists.min(axis=1)

        pocket_mask = min_dists <= cutoff
        if not np.any(pocket_mask):
            return None

        selected_coords = atom_coords[pocket_mask]
        selected_res = np.array(atom_res, dtype=np.int64)[pocket_mask]
        selected_dists = min_dists[pocket_mask]

        if len(selected_coords) > max_atoms:
            sort_idx = np.argsort(selected_dists)[:max_atoms]
            selected_coords = selected_coords[sort_idx]
            selected_res = selected_res[sort_idx]

        return {
            "pos": selected_coords,
            "res_idx": selected_res,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 4 — Per-process worker
# ---------------------------------------------------------------------------

def _process_one(args: Tuple) -> Optional[Tuple]:
    """Returns (smiles, [mol_with_H], pocket_dict) or (smiles, [mol_with_H]) or None."""
    if len(args) == 4:
        system_id, ligand_chain, smiles, plinder_dir_str = args
        extract_pocket, pocket_cutoff, max_pocket_atoms = True, 8.0, 150
    else:
        system_id, ligand_chain, smiles, plinder_dir_str, extract_pocket, pocket_cutoff, max_pocket_atoms = args

    plinder_dir = Path(plinder_dir_str)

    sdf_str = _read_sdf(plinder_dir, system_id, ligand_chain)
    if sdf_str is None:
        return None

    mol = _parse_mol(sdf_str)
    if mol is None:
        return None

    # Final validity check: canonical SMILES from the loaded mol must be non-empty
    try:
        check = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if not check:
            return None
    except Exception:
        return None

    pocket = None
    if extract_pocket:
        pocket = _extract_pocket(plinder_dir, system_id, mol, cutoff=pocket_cutoff, max_atoms=max_pocket_atoms)

    if pocket is not None:
        return (smiles, [mol], pocket)
    return (smiles, [mol])


# ---------------------------------------------------------------------------
# Step 5 — Process one split and write pickle
# ---------------------------------------------------------------------------

def process_split(
    df_split: pd.DataFrame,
    plinder_dir: Path,
    n_workers: int,
    split_name: str,
    output_dir: Path,
    extract_pocket: bool = True,
    pocket_cutoff: float = 8.0,
    max_pocket_atoms: int = 150,
) -> None:
    systems_dir = plinder_dir / "systems"
    if systems_dir.is_dir():
        disk_systems = set(os.listdir(systems_dir))
        df_split = df_split[df_split["system_id"].isin(disk_systems)].reset_index(drop=True)

    n = len(df_split)
    log.info(f"[{split_name}] {n:,} systems on disk to process | {n_workers} worker(s) | extract_pocket={extract_pocket}")

    args_list = [
        (
            row["system_id"],
            row["ligand_instance_chain"],
            row["ligand_rdkit_canonical_smiles"],
            str(plinder_dir),
            extract_pocket,
            pocket_cutoff,
            max_pocket_atoms,
        )
        for _, row in df_split.iterrows()
    ]

    results, n_skip = [], 0

    if n_workers <= 1:
        for a in tqdm(args_list, desc=split_name, unit="mol"):
            r = _process_one(a)
            if r is not None:
                results.append(r)
            else:
                n_skip += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            fmap = {exe.submit(_process_one, a): a for a in args_list}
            for fut in tqdm(as_completed(fmap), total=len(fmap),
                            desc=split_name, unit="mol"):
                r = fut.result()
                if r is not None:
                    results.append(r)
                else:
                    n_skip += 1

    log.info(
        f"[{split_name}] {len(results):,} kept | {n_skip:,} skipped "
        f"({100 * n_skip / max(n, 1):.1f}% skip rate)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{split_name}_data.pickle"
    with open(out, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"[{split_name}] Saved {len(results):,} entries -> {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess PLINDER v2 -> PharmaDiff pickle files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plinder_dir",
        default=os.path.expanduser("~/.local/share/plinder/2024-06/v2"),
        help="Root of the PLINDER 2024-06/v2 download (systems/ must be extracted).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write {train,val,test}_data.pickle.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=8,
        help="Parallel worker processes for SDF loading.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Filter annotations and print statistics without loading any SDFs.",
    )
    parser.add_argument(
        "--no_pocket", action="store_true",
        help="Disable pocket extraction (ligand-only mode).",
    )
    parser.add_argument(
        "--pocket_cutoff", type=float, default=8.0,
        help="Distance cutoff (Angstroms) from ligand heavy atoms to define pocket.",
    )
    parser.add_argument(
        "--max_pocket_atoms", type=int, default=150,
        help="Maximum pocket heavy atoms to retain per system.",
    )
    args = parser.parse_args()

    plinder_dir = Path(args.plinder_dir).expanduser()
    output_dir  = Path(args.output_dir).expanduser()

    df = load_and_filter_annotations(plinder_dir)
    df = deduplicate_to_one_ligand_per_system(df)

    if args.dry_run:
        log.info("Dry run complete — no files written.")
        return

    extract_pocket = not args.no_pocket

    for split_name in args.splits:
        df_split = df[df["split"] == split_name].reset_index(drop=True)
        if len(df_split) == 0:
            log.warning(f"[{split_name}] 0 systems after filtering. Skipping.")
            continue
        process_split(
            df_split=df_split,
            plinder_dir=plinder_dir,
            n_workers=args.n_workers,
            split_name=split_name,
            output_dir=output_dir,
            extract_pocket=extract_pocket,
            pocket_cutoff=args.pocket_cutoff,
            max_pocket_atoms=args.max_pocket_atoms,
        )

    log.info("Done. Place pickles in data/plinder/raw/ inside PharmaDiff.")


if __name__ == "__main__":
    main()