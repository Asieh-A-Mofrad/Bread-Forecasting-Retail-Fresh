# src/models/prophet_total.py
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from src.utils.leakage import assert_no_forbidden_features

# Regressors available at train/predict time for store-level total forecasting.
PROPHET_REGRESSORS = [
    "discount",
    "on_promotion",
    "promotion_count",
    "closed_in_last_3_days",
    "closed_in_next_4_days",
]


def _normalize_group_key(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _fit_single_store_prophet(df_store: pd.DataFrame, regressor_cols: list[str]) -> Prophet:
    data = df_store.copy()
    data["ds"] = pd.to_datetime(data["date"])
    data["y"] = data["quantity"]

    holidays_df = data.loc[data.get("is_holiday", 0) == 1, ["ds"]].copy()
    if not holidays_df.empty:
        holidays_df["holiday"] = "holiday"
    else:
        holidays_df = None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="additive",
    )

    for reg in regressor_cols:
        model.add_regressor(reg)

    fit_cols = ["ds", "y"] + regressor_cols
    model.fit(data[fit_cols])

    return model


def train_model(
        df: pd.DataFrame,
        model_path,
):
    """Train Prophet total-demand model(s).

    If `gln` is present, trains one Prophet model per store and saves them in a
    single bundle. If `gln` is missing, trains a single global model.
    """

    start = time.time()
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    regressor_cols = [c for c in PROPHET_REGRESSORS if c in data.columns]
    assert_no_forbidden_features(
        feature_cols=regressor_cols,
        model_type="prophet_total",
        context="prophet_total_regressors",
    )

    models_by_gln = {}
    group_col = "gln" if "gln" in data.columns else None

    if group_col is None:
        models_by_gln["__global__"] = _fit_single_store_prophet(data, regressor_cols)
    else:
        unique_gln = data[group_col].nunique(dropna=True)
        if unique_gln <= 1:
            print(
                f"[PROPHET TOTAL] Warning: only {unique_gln} unique '{group_col}' "
                "value(s) found. Training will produce a single store model."
            )
        for gln, gdf in data.groupby(group_col):
            key = _normalize_group_key(gln)
            if key is None:
                continue
            models_by_gln[key] = _fit_single_store_prophet(gdf, regressor_cols)

    bundle = {
        "model_type": "prophet_total",
        "models_by_gln": models_by_gln,
        "feature_cols": regressor_cols,
        "group_col": group_col,
    }

    # Backward compatibility for older evaluation paths expecting "model".
    if "__global__" in models_by_gln:
        bundle["model"] = models_by_gln["__global__"]

    joblib.dump(bundle, Path(model_path))

    elapsed = time.time() - start
    print(f"[PROPHET TOTAL] Training completed in {elapsed:.2f}s")

    return {
        "training_time_sec": elapsed,
        "regressors": regressor_cols,
        "n_models": len(models_by_gln),
    }
