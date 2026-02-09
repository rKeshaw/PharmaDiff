import argparse
from pathlib import Path

import torch

from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PLINDER samples for faster training.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/plinder_precomputed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PlinderGraphDataset(split=args.split, pocket_radius=args.pocket_radius)
    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

    saved = 0
    for idx in range(total):
        item = dataset[idx]
        if item is None:
            continue
        payload = {
            "ligand_pos": item["ligand"].pos.cpu(),
            "ligand_x": item["ligand"].x.cpu(),
            "ligand_edge_index": item["ligand"].edge_index.cpu(),
            "ligand_edge_attr": item["ligand"].edge_attr.cpu(),
            "ligand_charges": item["ligand"].charges.cpu(),
            "pocket_pos": item["pocket_pos"].cpu(),
            "pocket_feat": item["pocket_feat"].cpu(),
            "affinity": item["affinity"].cpu(),
            "system_id": item.get("system_id"),
        }
        torch.save(payload, out_dir / f"{idx:06d}.pt")
        saved += 1

    print(f"Saved {saved} samples to {out_dir}")


if __name__ == "__main__":
    main()