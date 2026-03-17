#!/usr/bin/env python3
"""Run horizon feature-availability audit and export CSV report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import select_features
from src.utils.leakage import (
    DEFAULT_HORIZON_SAFE_FEATURES,
    audit_horizon_feature_availability,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit feature availability for horizon forecasting.")
    p.add_argument("--dataset", required=True, help="Path to parquet dataset (store/share/direct).")
    p.add_argument(
        "--model-type",
        required=True,
        choices=["xgb_total", "xgb_share", "xgb_direct"],
        help="Model type to derive selected feature columns.",
    )
    p.add_argument("--horizon-days", type=int, default=7, help="Forecast horizon in days.")
    p.add_argument(
        "--known-future",
        default="",
        help="Comma-separated additional known-future feature names.",
    )
    p.add_argument(
        "--output",
        default="results/horizon_feature_audit.csv",
        help="Output CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(Path(args.dataset))
    X, _ = select_features(df, args.model_type)

    extra_known = {c.strip() for c in args.known_future.split(",") if c.strip()}
    known_future = set(DEFAULT_HORIZON_SAFE_FEATURES) | extra_known

    report = audit_horizon_feature_availability(
        feature_cols=X.columns,
        horizon_days=args.horizon_days,
        known_future_cols=known_future,
        output_csv=args.output,
    )

    counts = report["status"].value_counts().to_dict()
    print(f"Wrote audit report: {args.output}")
    print(f"Feature count: {len(report)}")
    print(f"Status counts: {counts}")

    flagged = report[report["status"].isin(["blocked", "review", "unknown"])]
    if not flagged.empty:
        print("\nTop flagged features:")
        print(flagged[["feature", "status", "reason"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
