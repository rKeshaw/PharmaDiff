"""
build_scaled_plinder_pyg.py
===========================
Processes raw pickle files in data/plinder_scaled/raw/ into PyTorch Geometric
processed .pt files and statistics files in data/plinder_scaled/processed/.
"""

import sys
import torch
from pathlib import Path
from pharmadiff.datasets.plinder_dataset import PlinderDataset

def main():
    root = Path("data/plinder_scaled")
    print(f"Building scaled PyG datasets in {root}...")

    splits = ["train", "val", "test"]
    datasets = {}

    for split in splits:
        print(f"\n--- Processing split: {split} ---")
        ds = PlinderDataset(
            split=split,
            root=str(root),
            remove_h=True,
            use_pocket=True,
        )
        datasets[split] = ds
        print(f"[{split}] Processed {len(ds):,} molecules/complexes.")
        assert len(ds) > 0, f"Split {split} is empty!"
        
        # Verify first item
        item = ds[0]
        lig = item["ligand"]
        pharma = item["pharmacophore"]
        pocket = item.get("pocket", None)
        p_str = f"pocket_x={pocket.x.shape}, pocket_pos={pocket.pos.shape}" if pocket else "no pocket"
        print(f"  Sample 0: lig_x={lig.x.shape}, lig_pos={lig.pos.shape}, pharma={pharma.x.shape}, {p_str}")

    print("\n" + "=" * 60)
    print("SCALED PLINDER PYG PROCESSING COMPLETE:")
    for split in splits:
        print(f"  {split:>6}: {len(datasets[split]):>6,} complexes")
    print("=" * 60)

if __name__ == "__main__":
    main()
