# src/models/utils.py
from src.utils.leakage import assert_columns_exist, assert_no_forbidden_features


def select_features(df, model_type):
    """
    Centralized feature selection logic per model type.
    """

    if model_type == "xgb_total":
        drop_cols = ["date", "gln", "gb_id", "keep", "year_month", "quantity"]
        target_col = "quantity"

    elif model_type == "xgb_share":
        drop_cols = ["date", "gln", "gb_id", "quantity", "store_total"]
        target_col = "target"

    elif model_type == "xgb_direct":
        drop_cols = ["date", "gln", "gb_id", "store_total", "quantity", "target"]
        target_col = "quantity"

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    assert_columns_exist(df.columns, [target_col], context=f"select_features:{model_type}")

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    # Minimal feature de-duplication to reduce collinearity without breaking models.
    redundant = []
    if "promo_combined" in X.columns and "on_promotion" in X.columns:
        redundant.append("on_promotion")
    if "month" in X.columns and "quarter" in X.columns:
        redundant.append("quarter")
    if "rolling_mean_7" in X.columns and "rolling_mean_30" in X.columns:
        redundant.append("rolling_mean_30")

    if redundant:
        X = X.drop(columns=redundant, errors="ignore")

    assert_no_forbidden_features(
        feature_cols=X.columns,
        model_type=model_type,
        context="select_features",
    )

    return X, y
