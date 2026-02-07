import argparse
from typing import Iterable, Optional
from tqdm import tqdm

from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset

def download_all(
    splits: Iterable[str],
    max_samples: Optional[int] = None,
    start_index: int = 0,
):
    for split in splits:
        print(f"Initializing Plinder {split} Dataset (this triggers index download)...")
        dataset = PlinderGraphDataset(split=split)

        total = len(dataset)
        if max_samples is not None:
            total = min(total, max_samples)

        print(f"Found {len(dataset)} systems. Starting caching {total} samples...")
        # Iterating through the dataset triggers the PlinderSystem download for each entry
        for i in tqdm(range(start_index, total)):
            try:
                _ = dataset[i]
            except Exception as e:
                print(f"Skipping failed system {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache PLINDER systems locally.")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to cache.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of systems per split.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for caching.")
    args = parser.parse_args()
    download_all(args.splits, max_samples=args.max_samples, start_index=args.start_index)
