# src/data/datasets.py
import pandas as pd

from src.features.store_temporal import add_store_temporal_features
from src.data.aggregation import aggregate_to_store_daily as _aggregate_to_store_daily
from src.utils.leakage import assert_columns_exist


def prepare_total_datasets(sales_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build store-level dataset for total sales modeling.

    Output columns:
      - gln, date, quantity (target)
      - store-level temporal + promo + calendar features
    """
    # 1. Aggregate product → store
    store_daily = _aggregate_to_store_daily(sales_daily)

    # 2. Add store-level temporal features
    store_daily = add_store_temporal_features(store_daily)

    # 3. Ensure datetime
    store_daily["date"] = pd.to_datetime(store_daily["date"])
    assert_columns_exist(
        store_daily.columns,
        required=["gln", "date", "quantity"],
        context="prepare_total_datasets",
    )

    return store_daily


def aggregate_to_store_daily(sales_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible re-export of store aggregation.
    Canonical implementation lives in `src.data.aggregation`.
    """
    return _aggregate_to_store_daily(sales_daily)


def prepare_product_daily_dataset(
        sales_daily: pd.DataFrame,
        target: str = "share",  # "share" or "quantity"
) -> pd.DataFrame:
    """
    Build product-level dataset for direct and share modeling.

    Target:
      share    = product_quantity / store_total
      quantity = product_quantity

    IMPORTANT:
    - store_total is ONLY used internally for share target
    - store_total is NEVER returned for quantity (direct model)
    """

    df = sales_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    assert_columns_exist(
        df.columns,
        required=["gln", "gb_id", "date", "quantity"],
        context="prepare_product_daily_dataset",
    )

    # --------------------------------------------------
    # 1. Store daily totals (INTERNAL USE ONLY)
    # --------------------------------------------------
    if target == "share":
        store_totals = (
            df.groupby(["gln", "date"])["quantity"]
            .sum()
            .rename("store_total")
            .reset_index()
        )

        df = df.merge(store_totals, on=["gln", "date"], how="left")
        df = df[df["store_total"] > 0].copy()

        # Target = share
        df["target"] = (df["quantity"] / df["store_total"]).clip(lower=0.0)

    elif target == "quantity":
        # Target = direct quantity
        df["target"] = df["quantity"]

    else:
        raise ValueError(f"Unknown target: {target}")

    # --------------------------------------------------
    # 2. Relative promotion & pricing features
    # --------------------------------------------------
    store_means = (
        df.groupby(["gln", "date"])
        .agg(
            store_mean_discount=("discount", "mean"),
            store_mean_price=("corrected_unit_price", "mean"),
            store_promo_rate=("on_promotion", "mean"),
        ).reset_index()
    )

    df = df.merge(store_means, on=["gln", "date"], how="left")

    df["discount_relative"] = df["discount"] - df["store_mean_discount"]
    df["price_relative"] = (df["corrected_unit_price"] - df["store_mean_price"]) / (df["store_mean_price"] + 1e-6)
    df["promo_boost"] = df["on_promotion"] * df["discount"]

    # --------------------------------------------------
    # 4. Drop columns that must never be used
    # --------------------------------------------------
    DROP_ALWAYS = [
        "netSalesKr",
        "grossMargin",
        "taxAmount",
        "price_error",
        "ref_store_price",
        "ref_global_price",
        "fallback_price",
        "ref_price",
        "price_corrected_flag",
        "price_correction_source",
    ]

    existing_drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    DROP_OBJECT_COLS = [
        "day_of_week",
        "last_holiday_type",
        "next_holiday_type",
    ]

    df = df.drop(columns=[c for c in DROP_OBJECT_COLS if c in df.columns])

    # --------------------------------------------------
    # 5. HARD SAFETY CHECK (recommended)
    # --------------------------------------------------
    if target == "quantity" and "store_total" in df.columns:
        raise RuntimeError("Leakage detected: store_total present in direct model")

    if target == "share":
        assert_columns_exist(
            df.columns,
            required=["target", "gln", "gb_id", "date"],
            context="prepare_product_daily_dataset:share",
        )
    elif target == "quantity":
        assert_columns_exist(
            df.columns,
            required=["target", "gln", "gb_id", "date", "quantity"],
            context="prepare_product_daily_dataset:quantity",
        )

    return df
