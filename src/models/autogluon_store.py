# src/models/autogluon_store.py
import time
import joblib
from pathlib import Path
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from src.utils.leakage import assert_horizon_known_future_features

ID_COL = "item_id"
TIME_COL = "timestamp"
TARGET_COL = "target"

# ----------------------------
# Known covariates (future-known)
# ----------------------------
KNOWN_COVARIATES = [
    # calendar
    "day_of_week_num",
    "sin_dow",
    "cos_dow",
    "sin_day",
    "cos_day",
    "month",
    "quarter",
    "is_holiday",
    "days_to_nearest_holiday",

    # promotions / business
    "on_promotion",
    "discount",
    "kronemarked",
    "promo_combined",
    "promotion_count",
    "closed_in_next_4_days",
]

KNOWN_COVARIATES_Calendar_only = [
    # calendar
    "day_of_week_num",
    "sin_dow",
    "cos_dow",
    "month",
    "quarter",
    "is_holiday",
]

KNOWN_COVARIATES_Promotion_only = [
    # promotions / business
    "on_promotion",
    "discount",
    "kronemarked",
    "promo_combined",
    "promotion_count",
]

KNOWN_COVARIATES_CP = [
    # calendar
    "day_of_week_num",
    "sin_dow",
    "cos_dow",
    "month",
    "quarter",
    "is_holiday",

    # promotions / business
    "on_promotion",
    "discount",
    "kronemarked",
    "promo_combined",
    "promotion_count",
]


def train_autogluon(
        train_df: pd.DataFrame,
        model_path: Path,
        prediction_length: int,
        use_features: bool = False,
        num_val_windows: int = 1,
):
    start = time.time()

    ts_df, feature_cols = build_ts_dataframe(
        train_df,
        use_features=use_features,
    )

    if use_features:
        assert_horizon_known_future_features(
            feature_cols=feature_cols,
            allowed_known_future=KNOWN_COVARIATES,
            context="autogluon_store_known_covariates",
        )

    predictor = TimeSeriesPredictor(
        target=TARGET_COL,
        prediction_length=prediction_length,
        freq="D",
        eval_metric="WAPE",
        known_covariates_names=feature_cols if use_features else [],
        path=model_path,
        verbosity=2,
        quantile_levels=[0.025, 0.05, 0.5, 0.95, 0.975]
    )

    predictor.fit(
        ts_df,
        presets="medium_quality",
        time_limit=3600,
        num_val_windows=num_val_windows,
    )

    joblib.dump(
        {
            "model_type": model_path.name,
            "use_features": use_features,
            "feature_cols": feature_cols,
            "prediction_length": prediction_length,
            "eval_metric": "WAPE",
        },
        model_path / "metadata.pkl",
    )

    print(f"[AUTOGLUON STORE | features={use_features}] Training completed in {time.time() - start:.2f}s")

    return {"train_time": time.time() - start}


# ---------------------
def predict_autogluon(
        test_df: pd.DataFrame,
        model_path: Path,
        store_col: str = "gln",  # "store_id"
):
    predictor = TimeSeriesPredictor.load(model_path)

    df = test_df.copy()
    df = df.rename(columns={
        store_col: "item_id",
        "date": "timestamp",
        "quantity": "target",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = (
        df
        .groupby(["item_id", "timestamp"], as_index=False)["target"]
        .sum()
    )

    df = (
        df
        .set_index("timestamp")
        .groupby("item_id", group_keys=False)
        .apply(lambda x: x.asfreq("D", fill_value=0))
        .reset_index()
    )

    ts_test = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    forecasts = predictor.predict(ts_test)

    return forecasts, df


# ---------------------------
def aggregate_store_forecasts(forecasts):
    return (
        forecasts
        .reset_index()
        .groupby("timestamp")["mean"]
        .sum()
        .values
    )


def aggregate_store_truth(df):
    return (
        df
        .groupby("timestamp")["target"]
        .sum()
        .values
    )


# -----
def build_ts_dataframe(
        raw_df: pd.DataFrame,
        use_features: bool,
):
    df = raw_df.copy()

    # ----------------------------
    # Rename columns
    # ----------------------------
    df = df.rename(columns={
        "gln": ID_COL,
        "date": TIME_COL,
        "quantity": TARGET_COL,
    })

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    # Defensive check
    n_stores = df[ID_COL].nunique()
    if n_stores != 11:
        raise ValueError(f"Expected 11 stores, but got {n_stores}. Check item_id normalization.")

    # ----------------------------
    # Aggregation rules
    # ----------------------------
    agg_dict = {
        TARGET_COL: "sum",

        # calendar
        "day_of_week_num": "first",
        "sin_dow": "first",
        "cos_dow": "first",
        "sin_day": "first",
        "cos_day": "first",
        "month": "first",
        "quarter": "first",
        "is_holiday": "first",
        "days_to_nearest_holiday": "first",

        # promotions / business
        "on_promotion": "max",
        "discount": "mean",
        "kronemarked": "max",
        "promo_combined": "max",
        "promotion_count": "sum",
        "closed_in_next_4_days": "max",
    }

    df = (
        df
        .groupby([ID_COL, TIME_COL], as_index=False)
        .agg({k: v for k, v in agg_dict.items() if k in df.columns})
    )

    # ----------------------------
    # Enforce DAILY frequency per store
    # ----------------------------
    df = (
        df
        .groupby(ID_COL, group_keys=False)
        .apply(
            lambda x: (
                x.set_index(TIME_COL)
                .asfreq("D")
                .ffill()
                .reset_index()
            )
        )
    )

    # ----------------------------
    # Sanity checks
    # ----------------------------
    expected_cols = {ID_COL, TIME_COL, TARGET_COL}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after resample: {missing}")

    assert df[ID_COL].nunique() == 11

    # ----------------------------
    # Select columns
    # ----------------------------
    keep_cols = [ID_COL, TIME_COL, TARGET_COL]

    if use_features:
        feature_cols = [c for c in KNOWN_COVARIATES if c in df.columns]
        keep_cols += feature_cols
    else:
        feature_cols = []

    print("Columns after aggregation & resample:")
    print(df[keep_cols].columns.tolist())

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df[keep_cols],
        id_column=ID_COL,
        timestamp_column=TIME_COL,
    )

    return ts_df, feature_cols
