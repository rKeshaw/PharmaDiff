import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from pharmadiff.datasets.plinder_datamodule import PlinderDataModule


def main() -> None:
    # 1. Use parse_known_args() instead of parse_args()
    # This captures known flags (--force) into 'args'
    # And puts everything else (like experiment=plinder) into 'overrides'
    parser = argparse.ArgumentParser(description="Precompute PLINDER dataset statistics.")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    
    args, overrides = parser.parse_known_args()

    # 2. Locate the config directory correctly
    # Assumes structure: pharmadiff/scripts/precompute... -> configs/ is 2 levels up from parent
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    
    # 3. Build the override list
    # We allow the command line overrides (like experiment=plinder) to take precedence
    # But we ensure dataset=plinder is the baseline.
    base_overrides = ["dataset=plinder"]
    
    if args.cache_path:
        base_overrides.append(f"dataset.statistics_cache_path={args.cache_path}")
    if args.max_samples is not None:
        base_overrides.append(f"dataset.statistics_max_samples={args.max_samples}")
    if args.force:
        base_overrides.append("dataset.statistics_force_recompute=True")
        
    # Combine baseline with command-line overrides
    # e.g. ["dataset=plinder", "experiment=plinder"]
    final_overrides = base_overrides + overrides
    
    print(f"--> Computing stats with overrides: {final_overrides}")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=final_overrides)

    # 4. Initialize DataModule
    # This triggers the dataset load. Because we passed experiment=plinder,
    # it will use the correct params (r=10.0, subset=1000) and hit the cache successfully.
    _ = PlinderDataModule(cfg)
    print("PLINDER statistics cached.")


if __name__ == "__main__":
    main()