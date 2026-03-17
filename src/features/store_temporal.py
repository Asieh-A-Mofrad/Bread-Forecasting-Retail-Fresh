# src/features/store_temporal.py
import pandas as pd


def add_store_temporal_features(store_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag and rolling temporal features computed directly from store-level totals.
    These features operate on daily store-level quantity.

    Added features:
        - sales_lag_1
        - sales_lag_7
        - rolling_mean_7, rolling_std_7
        - rolling_mean_30, rolling_std_30
        - days_since_last_kronemarked
        - days_since_last_promotion

    Parameters
    ----------
    store_daily : pd.DataFrame
        Output of aggregate_to_store_daily()

    Returns
    -------
    df : pd.DataFrame
        Store-level dataset enriched with temporal features.
    """

    df = store_daily.copy()
    df = df.sort_values(['gln', 'date'])

    # Group by store for lag/rolling operations.
    grp = df.groupby("gln", sort=False)

    # Lags
    df["sales_lag_1"] = grp["quantity"].shift(1)
    df["sales_lag_7"] = grp["quantity"].shift(7)

    # Rolling means/stds (shifted to avoid leakage)
    df["rolling_mean_7"] = grp["quantity"].transform(lambda s: s.rolling(7, min_periods=1).mean().shift(1))
    df["rolling_std_7"] = grp["quantity"].transform(lambda s: s.rolling(7, min_periods=1).std().shift(1))
    df["rolling_mean_30"] = grp["quantity"].transform(lambda s: s.rolling(30, min_periods=1).mean().shift(1))
    df["rolling_std_30"] = grp["quantity"].transform(lambda s: s.rolling(30, min_periods=1).std().shift(1))

    # Days since events
    df["days_since_last_kronemarked"] = grp["kronemarked"].transform(_days_since_last_true)
    df["days_since_last_promotion"] = grp["on_promotion"].transform(_days_since_last_positive)

    # Replace early-day NaNs
    df = df.fillna({
        'sales_lag_1': 0,
        'sales_lag_7': 0,
        'rolling_mean_7': 0,
        'rolling_std_7': 0,
        'rolling_mean_30': 0,
        'rolling_std_30': 0,
        'days_since_last_kronemarked': 999,
        'days_since_last_promotion': 999
    })

    return df


def _days_since_last_true(s: pd.Series) -> pd.Series:
    """Days since the last True value in a boolean-like series."""
    mask = s.fillna(0).astype(bool)
    idx = pd.Series(range(len(s)), index=s.index, dtype=float)
    last_true_idx = idx.where(mask).ffill()
    out = idx - last_true_idx
    return out


def _days_since_last_positive(s: pd.Series) -> pd.Series:
    """Days since the last strictly positive value."""
    mask = s.fillna(0) > 0
    idx = pd.Series(range(len(s)), index=s.index, dtype=float)
    last_pos_idx = idx.where(mask).ffill()
    out = idx - last_pos_idx
    return out
