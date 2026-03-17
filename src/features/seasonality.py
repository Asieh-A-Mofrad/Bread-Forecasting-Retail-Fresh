# src/features/seasonality.py
import numpy as np


def add_seasonality_features(df):
    """
    Adds cyclical and seasonal time features to capture weekly and annual patterns.
    """
    df = df.copy()
    df['day_of_week_num'] = df['date'].dt.weekday  # Monday=0, Sunday=6

    # Cyclical encoding for weekly seasonality
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)

    # Yearly seasonality features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Categorical month & quarter (for tree models)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    return df
