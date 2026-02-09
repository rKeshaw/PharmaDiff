import argparse
from collections import Counter

from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug PLINDER dataset failures.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    args = parser.parse_args()

    dataset = PlinderGraphDataset(
        split=args.split,
        pocket_radius=args.pocket_radius,
        debug=True,
    )
    counts = Counter()
    for idx in range(min(args.max_samples, len(dataset))):
        item = dataset[idx]
        if item is None:
            counts["none"] += 1
        else:
            counts["ok"] += 1
    print("Summary:", dict(counts))


if __name__ == "__main__":
    main()