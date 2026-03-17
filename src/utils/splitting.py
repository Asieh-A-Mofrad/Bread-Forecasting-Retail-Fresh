# src/utils/splitting.py

def split_by_time(df, test_start, test_end):
    train = df[df["date"] < test_start].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


def split_share_dataset(share_df, test_start, test_end):
    train = share_df[share_df["date"] < test_start].copy()
    test = share_df[(share_df["date"] >= test_start) & (share_df["date"] <= test_end)].copy()
    return train, test


def split_last_n_observations(df, n):
    """
    Split dataframe so that test set contains the last n observed timestamps.
    Works even if some calendar days are missing.
    """
    last_dates = sorted(df["date"].unique())[-n:]
    test_start = last_dates[0]

    train = df[df["date"] < test_start].copy()
    test = df[df["date"].isin(last_dates)].copy()
    return train, test
