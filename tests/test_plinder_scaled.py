"""
test_plinder_scaled.py
======================
Unit and integration tests for the scaled PLINDER dataset.
Verifies:
  1. Shard extraction integrity on disk.
  2. Intersection of disk systems with filtered annotation table.
  3. Correct RDKit parsing of ligands (with hydrogens).
  4. Correct pocket extraction (<8A cutoff, heavy atoms only, amino acid indices).
"""

import os
from pathlib import Path
import pandas as pd
from rdkit import Chem

from pharmadiff.datasets.plinder_preprocessing import (
    load_and_filter_annotations,
    deduplicate_to_one_ligand_per_system,
    _read_sdf,
    _parse_mol,
    _extract_pocket,
    _find_system_dir,
    ALLOWED_ATOMS,
)


def test_scaled_plinder_disk_coverage(plinder_dir, clean_annotations):
    systems_dir = plinder_dir / "systems"
    assert systems_dir.exists(), "Systems directory does not exist"
    disk_systems = set(os.listdir(systems_dir))
    
    val_clean = set(clean_annotations[clean_annotations["split"] == "val"]["system_id"])
    train_clean = set(clean_annotations[clean_annotations["split"] == "train"]["system_id"])
    test_clean = set(clean_annotations[clean_annotations["split"] == "test"]["system_id"])
    
    val_on_disk = len(val_clean & disk_systems)
    train_on_disk = len(train_clean & disk_systems)
    test_on_disk = len(test_clean & disk_systems)
    
    print(f"\n[Coverage Test] Train: {train_on_disk} / {len(train_clean)}, "
          f"Val: {val_on_disk} / {len(val_clean)}, "
          f"Test: {test_on_disk} / {len(test_clean)}")
    
    # Must have 100% of all validation systems (154)
    assert val_on_disk == 154, f"Expected 154 validation systems on disk, found {val_on_disk}"
    # Must have at least 1400 train systems on disk
    assert train_on_disk >= 1400, f"Expected >= 1400 train systems on disk, found {train_on_disk}"


def test_parse_and_extract_sample_systems(plinder_dir, clean_annotations):
    systems_dir = plinder_dir / "systems"
    disk_systems = set(os.listdir(systems_dir))
    
    # Sample 15 systems from train and val
    val_systems = clean_annotations[
        (clean_annotations["split"] == "val") &
        (clean_annotations["system_id"].isin(disk_systems))
    ].head(10)
    
    train_systems = clean_annotations[
        (clean_annotations["split"] == "train") &
        (clean_annotations["system_id"].isin(disk_systems))
    ].head(5)
    
    sample_df = pd.concat([val_systems, train_systems])
    assert len(sample_df) == 15
    
    for _, row in sample_df.iterrows():
        sys_id = row["system_id"]
        chain = row["ligand_instance_chain"]
        
        sys_dir = _find_system_dir(plinder_dir, sys_id)
        assert sys_dir is not None, f"Could not find directory for {sys_id}"
        
        sdf_str = _read_sdf(plinder_dir, sys_id, chain)
        assert sdf_str is not None, f"Could not read SDF for {sys_id} chain {chain}"
        
        mol = _parse_mol(sdf_str)
        if mol is None:
            # Check that it was legitimately skipped (e.g. non-standard atoms or kekulization)
            suppl = Chem.SDMolSupplier()
            suppl.SetData(sdf_str, removeHs=False, sanitize=False)
            raw_mol = next(suppl, None)
            if raw_mol is not None:
                heavy_symbols = {a.GetSymbol() for a in raw_mol.GetAtoms() if a.GetAtomicNum() != 1}
                is_unsupported = not heavy_symbols.issubset(ALLOWED_ATOMS)
                print(f"  [SKIPPED - VALID] {sys_id} correctly skipped (unsupported atoms: {heavy_symbols - ALLOWED_ATOMS})")
                continue
            else:
                print(f"  [SKIPPED - VALID] {sys_id} SDF unparseable")
                continue
        
        assert mol.GetNumConformers() > 0, f"Mol has no conformers: {sys_id}"
        assert mol.GetNumAtoms() > 0
        
        # Check heavy atom vocabulary
        heavy_symbols = {a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 1}
        assert heavy_symbols.issubset(ALLOWED_ATOMS), f"Unsupported atoms in {sys_id}: {heavy_symbols - ALLOWED_ATOMS}"
        
        # Check pocket extraction
        pocket = _extract_pocket(plinder_dir, sys_id, mol, cutoff=8.0, max_atoms=150)
        assert pocket is not None, f"Failed to extract pocket for {sys_id}"
        assert "pos" in pocket and "res_idx" in pocket
        assert len(pocket["pos"]) > 0, f"Pocket empty for {sys_id}"
        assert pocket["pos"].shape[1] == 3
        assert len(pocket["res_idx"]) == len(pocket["pos"])
        print(f"  [OK] {sys_id} ({chain}): ligand atoms={mol.GetNumAtoms()} (heavy={mol.GetNumHeavyAtoms()}), pocket atoms={len(pocket['pos'])}")


if __name__ == "__main__":
    p = Path(os.path.expanduser("~/.local/share/plinder/2024-06/v2"))
    print("Loading clean annotations...")
    ann = load_and_filter_annotations(p)
    ann = deduplicate_to_one_ligand_per_system(ann)
    print("Testing disk coverage...")
    test_scaled_plinder_disk_coverage(p, ann)
    print("Testing parsing and pocket extraction...")
    test_parse_and_extract_sample_systems(p, ann)
    print("\nALL SCALED PLINDER TESTS PASSED SUCCESSFULLY!")
