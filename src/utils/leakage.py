# src/utils/leakage.py
"""Leakage guard utilities.

These checks are intentionally strict and fail fast to prevent accidental
training-time leakage from derived targets or unavailable future information.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Columns that are predictions/ground-truth artifacts and should never be features.
GLOBAL_FORBIDDEN_FEATURES = {
    "y_true",
    "y_pred",
    "pred_total",
    "pred_share",
    "pred_quantity",
}

# Model-specific forbidden feature columns.
FORBIDDEN_BY_MODEL = {
    "xgb_total": {"quantity", "target", "store_total"},
    "xgb_share": {"store_total"},
    "xgb_direct": {"store_total", "target"},
    "prophet_total": {"quantity", "target", "store_total"},
}

# Conservative default set for known-at-forecast-time features.
DEFAULT_HORIZON_SAFE_FEATURES = {
    "day_of_week_num",
    "month",
    "quarter",
    "sin_dow",
    "cos_dow",
    "sin_day",
    "cos_day",
    "is_holiday",
    "days_to_nearest_holiday",
    "closed_in_next_4_days",
    # Planned business inputs (only safe when truly known/planned at forecast time).
    "on_promotion",
    "discount",
    "kronemarked",
    "promo_combined",
    "promotion_count",
}

HISTORY_DERIVED_TOKENS = (
    "lag",
    "rolling",
    "days_since_last",
    "_te",
    "mean_target",
)

BUSINESS_INPUT_TOKENS = (
    "promotion",
    "promo",
    "discount",
    "kronemarked",
    "closed_in_next",
)


def assert_columns_exist(df_columns: Iterable[str], required: Iterable[str], context: str) -> None:
    missing = sorted(set(required) - set(df_columns))
    if missing:
        raise ValueError(f"[{context}] Missing required columns: {missing}")


def assert_no_forbidden_features(feature_cols: Iterable[str], model_type: str, context: str = "") -> None:
    feature_cols = set(feature_cols)
    forbidden = set(FORBIDDEN_BY_MODEL.get(model_type, set())) | GLOBAL_FORBIDDEN_FEATURES
    leaked = sorted(feature_cols & forbidden)
    if leaked:
        prefix = f"[{context}] " if context else ""
        raise RuntimeError(f"{prefix}Leakage risk: forbidden feature columns present for {model_type}: {leaked}")


def assert_horizon_known_future_features(
        feature_cols: Iterable[str],
        allowed_known_future: Iterable[str] | None = None,
        context: str = "horizon",
) -> None:
    """Validate that feature set contains only known-at-origin covariates.

    This does not prove leakage absence, but blocks obvious future-unavailable columns.
    """
    allowed = set(allowed_known_future or DEFAULT_HORIZON_SAFE_FEATURES)
    cols = set(feature_cols)

    # Anything explicitly lag/rolling-like is assumed history-derived and allowed.
    history_like = {c for c in cols if
                    any(tok in c for tok in ("lag", "rolling", "days_since_last", "mean_target", "_te"))}

    unknown = sorted(cols - allowed - history_like)
    if unknown:
        raise RuntimeError(
            f"[{context}] Potential forecast-time availability risk. "
            f"Review these columns before horizon training: {unknown}"
        )


def audit_horizon_feature_availability(
        feature_cols: Iterable[str],
        horizon_days: int,
        known_future_cols: Iterable[str] | None = None,
        output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Create a feature availability audit table for horizon forecasting.

    Status definitions:
    - blocked: direct leakage or prediction artifact
    - known_future: explicitly safe at forecast origin
    - history_derived: likely safe if computed with proper shift
    - review: needs explicit confirmation (planned/forecasted availability)
    - unknown: not classified; requires manual review
    """
    known_future = set(known_future_cols or DEFAULT_HORIZON_SAFE_FEATURES)
    blocked = GLOBAL_FORBIDDEN_FEATURES | {"quantity", "target", "store_total"}

    rows = []
    for col in sorted(set(feature_cols)):
        if col in blocked:
            status = "blocked"
            reason = "direct target/prediction artifact"
            recommendation = "remove from feature set"
        elif col in known_future:
            status = "known_future"
            reason = "explicitly listed as forecast-time available"
            recommendation = "safe to keep"
        elif any(tok in col for tok in HISTORY_DERIVED_TOKENS):
            status = "history_derived"
            reason = "looks history-derived (must be shift-safe)"
            recommendation = "keep only if computed from t-1 and earlier"
        elif any(tok in col for tok in BUSINESS_INPUT_TOKENS):
            status = "review"
            reason = "business input may be unknown at forecast origin"
            recommendation = "keep only if planned/forecasted upstream"
        else:
            status = "unknown"
            reason = "not matched by known-safe or history-derived rules"
            recommendation = "manual feature-availability review"

        rows.append(
            {
                "feature": col,
                "status": status,
                "reason": reason,
                "recommendation": recommendation,
                "horizon_days": horizon_days,
            }
        )

    report = pd.DataFrame(rows).sort_values(["status", "feature"]).reset_index(drop=True)

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    return report
