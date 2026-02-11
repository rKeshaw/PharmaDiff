#!/usr/bin/env python3
"""Evaluation reporting utility.

Builds:
1) Standardized metrics table.
2) Pocket ablation comparison table (baseline vs cross-attention).
3) Quality-stratified pocket metric report.

Inputs are CSV/JSON artifacts produced by prior evaluation runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_METRIC_COLUMNS = [
    "validity",
    "uniqueness",
    "novelty",
    "rdkit_pharma_match",
    "pgmg_match_score",
    "pocket_contact_satisfaction",
    "pocket_ifp_similarity",
    "rdkit_QED",
    "rdkit_SAS",
]


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported input format for {path}. Use CSV or JSON.")


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_summary_table(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    avail = [c for c in metric_cols if c in df.columns]
    if not avail:
        raise ValueError("No requested metrics found in input table.")
    out = df[avail].mean(numeric_only=True).to_frame("mean").reset_index(names="metric")
    return out.sort_values("metric").reset_index(drop=True)


def build_ablation_table(
    baseline_df: pd.DataFrame,
    pocket_df: pd.DataFrame,
    metric_cols: List[str],
) -> pd.DataFrame:
    base = build_summary_table(baseline_df, metric_cols).rename(columns={"mean": "baseline_pooled"})
    pocket = build_summary_table(pocket_df, metric_cols).rename(columns={"mean": "pocket_cross_attention"})
    merged = base.merge(pocket, on="metric", how="outer")
    merged["delta"] = merged["pocket_cross_attention"] - merged["baseline_pooled"]
    return merged.sort_values("metric").reset_index(drop=True)


def _auto_quality_bin(df: pd.DataFrame) -> pd.Series:
    if "quality_weight" in df.columns:
        q = pd.to_numeric(df["quality_weight"], errors="coerce")
        return pd.cut(
            q,
            bins=[-float("inf"), 0.5, 0.8, float("inf")],
            labels=["low", "medium", "high"],
        )
    if "entry_resolution" in df.columns:
        r = pd.to_numeric(df["entry_resolution"], errors="coerce")
        return pd.cut(
            r,
            bins=[-float("inf"), 2.0, 2.8, float("inf")],
            labels=["high", "medium", "low"],
        )
    raise ValueError(
        "Quality stratification requires either `quality_weight` or `entry_resolution` column."
    )


def build_quality_stratified_table(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["quality_bin"] = _auto_quality_bin(df)
    avail = [c for c in metric_cols if c in df.columns]
    pocket_cols = [c for c in ["pocket_contact_satisfaction", "pocket_ifp_similarity"] if c in avail]
    if not pocket_cols:
        raise ValueError(
            "Need pocket metrics in input (`pocket_contact_satisfaction` and/or `pocket_ifp_similarity`)."
        )
    grouped = (
        df.dropna(subset=["quality_bin"])
        .groupby("quality_bin", observed=True)[pocket_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = (
        df.dropna(subset=["quality_bin"])
        .groupby("quality_bin", observed=True)
        .size()
        .to_frame("n_samples")
        .reset_index()
    )
    return counts.merge(grouped, on="quality_bin", how="left")


def _save(df: pd.DataFrame, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_stem.with_suffix(".csv"), index=False)
    out_stem.with_suffix(".md").write_text(df.to_markdown(index=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phase-5 evaluation reports.")
    parser.add_argument("--metrics", type=Path, required=True, help="CSV/JSON metrics input (cross-attn model).")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
        help="Optional CSV/JSON metrics input for pooled-pocket baseline ablation.",
    )
    parser.add_argument(
        "--quality-metrics",
        type=Path,
        default=None,
        help="Optional per-sample CSV/JSON including quality columns and pocket metrics.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/phase5"))
    parser.add_argument(
        "--metrics-cols",
        nargs="*",
        default=DEFAULT_METRIC_COLUMNS,
        help="Metric columns used for summary and ablation.",
    )
    args = parser.parse_args()

    metrics_df = _coerce_numeric(_load_table(args.metrics), args.metrics_cols)
    summary = build_summary_table(metrics_df, args.metrics_cols)
    _save(summary, args.out_dir / "standardized_metrics")

    if args.baseline_metrics is not None:
        baseline_df = _coerce_numeric(_load_table(args.baseline_metrics), args.metrics_cols)
        ablation = build_ablation_table(baseline_df, metrics_df, args.metrics_cols)
        _save(ablation, args.out_dir / "ablation_table")

    if args.quality_metrics is not None:
        quality_df = _coerce_numeric(_load_table(args.quality_metrics), args.metrics_cols + ["quality_weight", "entry_resolution"])
        stratified = build_quality_stratified_table(quality_df, args.metrics_cols)
        _save(stratified, args.out_dir / "quality_stratified")


if __name__ == "__main__":
    main()