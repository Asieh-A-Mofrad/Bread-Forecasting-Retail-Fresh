# src/models/autogluon_product.py
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
from pathlib import Path

AG_MODELS_STABLE = {
    "SeasonalNaive": {},
    "AutoETS": {},
    "DynamicOptimizedTheta": {},

    # Strong tabular baselines
    "RecursiveTabular": {},
    "DirectTabular": {},

    # Deep models that usually finish
    "DeepAR": {},
    "TemporalFusionTransformer": {},
    "Chronos": {},
}

ID_COL = "item_id"
TIME_COL = "timestamp"
TARGET_COL = "target"


def build_ts_dataframe_product(df, feature_cols=None):
    df = df.copy()

    # Construct product-level item_id
    df[ID_COL] = df["gb_id"].astype(str) + "_" + df["gln"].astype(str)

    # Ensure datetime
    df[TIME_COL] = pd.to_datetime(df["timestamp"])

    # Sort
    df = df.sort_values([ID_COL, TIME_COL])

    if feature_cols is None:
        feature_cols = []

    feature_cols = [c for c in feature_cols if c in df.columns]

    keep_cols = [ID_COL, TIME_COL, TARGET_COL] + feature_cols

    ts_df = df[keep_cols]

    return ts_df, feature_cols


def train_autogluon_product(
        train_ts: TimeSeriesDataFrame,
        known_covariates: [],
        model_path: Path,
        prediction_length: int,
        num_val_windows: int = 1,
        random_seed: int = 123,
):
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=prediction_length,
        freq="D",
        path=model_path,
        verbosity=2,
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        known_covariates_names=known_covariates,
        eval_metric="WAPE",
    )

    predictor.fit(
        train_data=train_ts,
        presets="high_quality",
        time_limit=7200,
        hyperparameters=AG_MODELS_STABLE,
        num_val_windows=num_val_windows,
        random_seed=random_seed,
    )

    return predictor
