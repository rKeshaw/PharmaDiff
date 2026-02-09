import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from pharmadiff.datasets.plinder_datamodule import PlinderDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PLINDER dataset statistics.")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    overrides = ["dataset=plinder"]
    if args.cache_path:
        overrides.append(f"dataset.statistics_cache_path={args.cache_path}")
    if args.max_samples is not None:
        overrides.append(f"dataset.statistics_max_samples={args.max_samples}")
    if args.force:
        overrides.append("dataset.statistics_force_recompute=True")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    _ = PlinderDataModule(cfg)
    print("PLINDER statistics cached.")


if __name__ == "__main__":
    main()