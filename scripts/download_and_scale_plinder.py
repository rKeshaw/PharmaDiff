"""
download_and_scale_plinder.py
=============================
Scales the local PLINDER dataset by downloading and extracting targeted shards
from Google Cloud Storage (gs://plinder/2024-06/v2/systems/{shard}.zip) using
anonymous access.

Strategy:
  1. Identifies the 87 shards containing 100% of all clean validation systems (154 systems).
  2. Ranks remaining shards by clean training system density to reach target training size (e.g. 2,000 - 5,000+ systems).
  3. Concurrently downloads and unpacks zip shards directly into plinder_dir/systems/.
  4. Automatically marks {shard}_done and removes zip archives to conserve disk space.
  5. Intersects downloaded systems with the filtered annotations and reports exact counts.
"""

import argparse
import io
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from pharmadiff.datasets.plinder_preprocessing import (
    load_and_filter_annotations,
    deduplicate_to_one_ligand_per_system,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

GCS_BUCKET = "plinder"
GCS_PREFIX = "2024-06/v2/systems"


def get_target_shards(
    df: pd.DataFrame,
    target_train: int = 2500,
    include_all_val: bool = True,
    target_test: int = 100,
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Selects a compact set of shards that:
    1. Contains all validation systems (if include_all_val=True).
    2. Contains at least target_train training systems.
    3. Contains at least target_test test systems.
    """
    df = df.copy()
    df["shard"] = df["system_id"].apply(lambda s: s[1:3].lower())

    selected_shards: Set[str] = set()

    # Step 1: Add all val shards
    if include_all_val:
        val_shards = set(df[df["split"] == "val"]["shard"])
        selected_shards.update(val_shards)
        log.info(f"Added {len(val_shards)} validation shards.")

    # Step 2: Add top test shards to meet target_test
    curr_test = df[df["shard"].isin(selected_shards) & (df["split"] == "test")]
    if len(curr_test) < target_test:
        test_counts = (
            df[~df["shard"].isin(selected_shards) & (df["split"] == "test")]
            .groupby("shard")
            .size()
            .sort_values(ascending=False)
        )
        for shard, count in test_counts.items():
            selected_shards.add(shard)
            curr_count = len(df[df["shard"].isin(selected_shards) & (df["split"] == "test")])
            if curr_count >= target_test:
                break
        log.info(f"Expanded to {len(selected_shards)} shards to reach test target >= {target_test}.")

    # Step 3: Add top train shards to meet target_train
    curr_train = df[df["shard"].isin(selected_shards) & (df["split"] == "train")]
    if len(curr_train) < target_train:
        train_counts = (
            df[~df["shard"].isin(selected_shards) & (df["split"] == "train")]
            .groupby("shard")
            .size()
            .sort_values(ascending=False)
        )
        for shard, count in train_counts.items():
            selected_shards.add(shard)
            curr_count = len(df[df["shard"].isin(selected_shards) & (df["split"] == "train")])
            if curr_count >= target_train:
                break
        log.info(f"Expanded to {len(selected_shards)} shards to reach train target >= {target_train}.")

    covered_df = df[df["shard"].isin(selected_shards)]
    log.info("Selected shards summary:")
    for s in ["train", "val", "test"]:
        cnt = (covered_df["split"] == s).sum()
        log.info(f"  {s:>6}: {cnt:>6,}")

    return selected_shards, covered_df


def download_and_extract_shard(
    shard: str,
    systems_dir: Path,
    keep_zip: bool = False,
) -> Tuple[str, bool, str]:
    """
    Downloads gs://plinder/2024-06/v2/systems/{shard}.zip and unpacks it into systems_dir.
    Touches {shard}_done upon completion.
    """
    done_marker = systems_dir / f"{shard}_done"
    if done_marker.is_file():
        return shard, True, "already_done"

    zip_path = systems_dir / f"{shard}.zip"
    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_BUCKET)
        blob_name = f"{GCS_PREFIX}/{shard}.zip"
        blob = bucket.get_blob(blob_name)
        if blob is None:
            return shard, False, f"Blob not found: {blob_name}"

        # Stream download directly to file
        temp_zip = systems_dir / f".tmp_{shard}.zip"
        blob.download_to_filename(str(temp_zip))
        temp_zip.replace(zip_path)

        # Unpack zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=systems_dir)

        # Touch done marker
        done_marker.touch()

        # Cleanup zip if requested
        if not keep_zip:
            try:
                zip_path.unlink()
            except Exception:
                pass

        return shard, True, "downloaded_and_extracted"
    except Exception as e:
        if zip_path.is_file():
            try:
                zip_path.unlink()
            except Exception:
                pass
        return shard, False, str(e)


def run_scaling(
    plinder_dir: Path,
    target_train: int = 2500,
    target_test: int = 100,
    max_workers: int = 8,
    keep_zips: bool = False,
):
    systems_dir = plinder_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading and filtering annotations...")
    df = load_and_filter_annotations(plinder_dir)
    df = deduplicate_to_one_ligand_per_system(df)

    log.info(f"Determining shards for train={target_train}, test={target_test}...")
    shards, covered_df = get_target_shards(
        df,
        target_train=target_train,
        include_all_val=True,
        target_test=target_test,
    )

    # Check which are already done
    shards_to_download = [
        s for s in sorted(shards)
        if not (systems_dir / f"{s}_done").is_file()
    ]
    already_done = len(shards) - len(shards_to_download)
    log.info(f"Total target shards: {len(shards)}. Already extracted: {already_done}. To download: {len(shards_to_download)}.")

    if not shards_to_download:
        log.info("All target shards are already downloaded and extracted!")
        return

    log.info(f"Starting concurrent download and extraction with {max_workers} threads...")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_and_extract_shard,
                shard,
                systems_dir,
                keep_zips,
            ): shard
            for shard in shards_to_download
        }

        with tqdm(total=len(shards_to_download), desc="Downloading shards", unit="shard") as pbar:
            for future in as_completed(futures):
                shard, ok, msg = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    log.warning(f"Failed shard {shard}: {msg}")
                pbar.update(1)

    log.info(f"Download complete: {success_count} succeeded, {fail_count} failed.")

    # Final verification of available systems on disk
    disk_systems = set(os.listdir(systems_dir))
    final_covered = covered_df[covered_df["system_id"].isin(disk_systems)]
    log.info("=" * 60)
    log.info("PLINDER DATASET SCALING SUMMARY ON DISK:")
    for s in ["train", "val", "test"]:
        cnt = (final_covered["split"] == s).sum()
        log.info(f"  {s:>6}: {cnt:>6,} systems verified on disk")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and scale PLINDER dataset")
    parser.add_argument(
        "--plinder_dir",
        type=Path,
        default=Path("/home/venkatgb/.local/share/plinder/2024-06/v2"),
        help="Path to PLINDER root directory",
    )
    parser.add_argument(
        "--target_train",
        type=int,
        default=2500,
        help="Target number of clean training systems (e.g. 2500, 3500, 5000)",
    )
    parser.add_argument(
        "--target_test",
        type=int,
        default=100,
        help="Target number of clean test systems",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel download/extraction threads",
    )
    parser.add_argument(
        "--keep_zips",
        action="store_true",
        help="Keep downloaded zip files instead of deleting after extraction",
    )
    args = parser.parse_args()

    run_scaling(
        plinder_dir=args.plinder_dir,
        target_train=args.target_train,
        target_test=args.target_test,
        max_workers=args.max_workers,
        keep_zips=args.keep_zips,
    )
