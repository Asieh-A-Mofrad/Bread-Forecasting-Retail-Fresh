#!/usr/bin/env python3
"""Quick compare: baseline vs pruned XGB feature sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import select_features
from src.models.evaluate import get_xgb_feature_importance
from src.utils.metrics import calculate_error_metrics
from src.utils.splitting import split_last_n_observations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs pruned XGB feature sets.")
    p.add_argument("--dataset", required=True, help="Path to parquet dataset (store/share/direct).")
    p.add_argument(
        "--model-type",
        default="xgb_total",
        choices=["xgb_total", "xgb_share", "xgb_direct"],
        help="Model type for select_features.",
    )
    p.add_argument("--horizon-days", type=int, default=7, help="Holdout length (last N dates).")
    p.add_argument("--top-k", type=int, default=20, help="Keep top-K important features.")
    p.add_argument("--threshold", type=float, default=None, help="Importance threshold (overrides top-k).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(Path(args.dataset))
    train_df, test_df = split_last_n_observations(df, args.horizon_days)

    X_base, y_base = select_features(train_df.copy(), args.model_type)

    params = {
        "random_state": 42,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }
    model_base = xgb.XGBRegressor(**params)
    model_base.fit(X_base, y_base)

    X_test_base = test_df[X_base.columns]
    y_test = test_df["quantity"].values
    y_pred_base = model_base.predict(X_test_base)
    metrics_base = calculate_error_metrics(y_test, y_pred_base, verbose=False)

    imp_df = get_xgb_feature_importance(model_base, feature_names=X_base.columns)
    if args.threshold is not None:
        keep_cols = imp_df[imp_df["importance"] > args.threshold]["feature"].tolist()
        mode = f"threshold>{args.threshold}"
    else:
        keep_cols = imp_df["feature"].head(args.top_k).tolist()
        mode = f"top_k={args.top_k}"

    if not keep_cols:
        raise SystemExit("No features selected. Lower threshold or increase top-k.")

    X_pruned = X_base[keep_cols]
    model_pruned = xgb.XGBRegressor(**params)
    model_pruned.fit(X_pruned, y_base)

    X_test_pruned = test_df[keep_cols]
    y_pred_pruned = model_pruned.predict(X_test_pruned)
    metrics_pruned = calculate_error_metrics(y_test, y_pred_pruned, verbose=False)

    summary = pd.DataFrame([metrics_base, metrics_pruned], index=["baseline", "pruned"]).round(4)
    print(f"Selection mode: {mode}")
    print(f"Baseline feature count: {X_base.shape[1]}")
    print(f"Pruned feature count: {len(keep_cols)}")
    print("Kept features:")
    print(pd.Series(keep_cols).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
