# src/features/temporal.py
import pandas as pd


def add_lag_features(df, lag_days=7):
    """
    Adds lag-based rolling features for each (store, product) pair:
    - 'avg_last_{lag_days}_sales': Average quantity sold over the past lag_days (excluding current day).
    - 'std_last_{lag_days}_sales': Standard deviation of quantity sold over the past lag_days (optional volatility feature).
    - 'promotion_days_count_last_{lag_days}': Number of promotion days in the past lag_days.

    Args:
        df (pd.DataFrame): Input dataframe with columns ['gln', 'gb_id', 'date', 'quantity', 'on_promotion'].
        lag_days (int): Number of past days to include in the rolling window.

    Returns:
        pd.DataFrame: Dataframe with new lag-based features added.
    """
    df = df.sort_values(by=['gln', 'gb_id', 'date']).copy()

    # Lag quantity and on_promotion to avoid data leakage
    df['lagged_quantity'] = df.groupby(['gln', 'gb_id'])['quantity'].shift(1)
    df['lagged_promotion'] = df.groupby(['gln', 'gb_id'])['on_promotion'].shift(1)

    # Rolling calculations within each group
    group = df.groupby(['gln', 'gb_id'])

    # Rolling sum, count, mean, std
    rolling_sum = (
        group['lagged_quantity']
        .rolling(window=lag_days, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    rolling_count = (
        group['lagged_quantity']
        .rolling(window=lag_days, min_periods=1)
        .count()
        .reset_index(level=[0, 1], drop=True)
    )
    rolling_mean = rolling_sum / rolling_count

    rolling_std = (
        group['lagged_quantity']
        .rolling(window=lag_days, min_periods=1)
        .std()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0)

    # Rolling sum of promotion days
    rolling_promo_days = (
        group['lagged_promotion']
        .rolling(window=lag_days, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0)

    # Assign features with lag_days in their names
    df[f'avg_last_{lag_days}_sales'] = rolling_mean.fillna(0)
    df[f'std_last_{lag_days}_sales'] = rolling_std
    df[f'promotion_days_count_last_{lag_days}'] = rolling_promo_days

    # Drop intermediate columns
    df = df.drop(columns=['lagged_quantity', 'lagged_promotion'])

    return df


def add_rolling_mean_features(
        df,
        time_window=5,
        add_all_days_mean=True,
        use_ema=False
):
    """
    Adds rolling mean features for sales (weekday-specific and optionally all-days),
    plus rolling counts of promotion days.

    Features:
        - rolling_mean: weekday-specific rolling mean of sales
        - rolling_mean_all (optional): all-days rolling mean of sales
        - promotion_days_count_rolling: weekday-specific rolling promo count

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['date', 'gln', 'gb_id', 'quantity', 'on_promotion'].
    time_window : int
        Number of past observations to use.
    add_all_days_mean : bool
        Add general rolling mean per (gln, gb_id).
    use_ema : bool
        If True, use EMA instead of simple rolling mean.

    Returns
    -------
    pd.DataFrame
        Updated with new rolling features.
    """

    df = df.sort_values(by=['gln', 'gb_id', 'date']).copy()
    df['day_of_week'] = df['date'].dt.day_name()

    # Select rolling function
    if use_ema:
        def roll_func(x):
            return x.shift(1).ewm(span=time_window, adjust=False).mean()
    else:
        def roll_func(x):
            return x.shift(1).rolling(window=time_window, min_periods=1).mean()

    # --- Weekday-specific rolling mean ---
    df['rolling_mean'] = (
        df.groupby(['gln', 'gb_id', 'day_of_week'])['quantity']
        .transform(roll_func)
    )

    # --- Weekday-specific rolling promotion count ---
    df['promotion_days_count_rolling'] = (
        df.groupby(['gln', 'gb_id', 'day_of_week'])['on_promotion']
        .transform(lambda x: x.shift(1).rolling(window=time_window, min_periods=1).sum())
    )

    # --- All-days rolling mean (optional) ---
    if add_all_days_mean:
        df['rolling_mean_all'] = (
            df.groupby(['gln', 'gb_id'])['quantity']
            .transform(roll_func)
        )

    # --- Fill missing values ---
    mean_per_product = df.groupby(['gln', 'gb_id'])['quantity'].transform('mean')

    df['rolling_mean'] = df['rolling_mean'].fillna(mean_per_product)
    df['promotion_days_count_rolling'] = df['promotion_days_count_rolling'].fillna(0)

    if add_all_days_mean:
        df['rolling_mean_all'] = df['rolling_mean_all'].fillna(mean_per_product)

    return df


# src/features/store_temporal.py
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

    # Group by store for rolling operations
    df = df.groupby('gln', group_keys=False).apply(_add_temporal_features_store)

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


# --------
# Helper
# -------
def _add_temporal_features_store(g):
    """Compute store-level lag and rolling windows."""
    g = g.sort_values('date')

    # Lags
    g['sales_lag_1'] = g['quantity'].shift(1)
    g['sales_lag_7'] = g['quantity'].shift(7)

    # Rolling means & stds (shifted to avoid leakage)
    g['rolling_mean_7'] = g['quantity'].rolling(7, min_periods=1).mean().shift(1)
    g['rolling_std_7'] = g['quantity'].rolling(7, min_periods=1).std().shift(1)
    g['rolling_mean_30'] = g['quantity'].rolling(30, min_periods=1).mean().shift(1)
    g['rolling_std_30'] = g['quantity'].rolling(30, min_periods=1).std().shift(1)

    # Days since events
    g['days_since_last_kronemarked'] = (
            (~g['kronemarked'].astype(bool)).cumsum() -
            (~g['kronemarked'].astype(bool)).cumsum().where(g['kronemarked'].astype(bool)).ffill().fillna(0)
    )

    g['days_since_last_promotion'] = (
            (g['on_promotion'] == 0).cumsum() -
            (g['on_promotion'] == 0).cumsum().where(g['on_promotion'] > 0).ffill().fillna(0)
    )

    return g
