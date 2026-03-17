# src/features/holidays.py
import pandas as pd
import numpy as np


def add_holiday_features(df, holiday_obj, past_days=3, forward_days=4):
    """
    Fast, vectorized holiday + closure feature engineering without
    last/next holiday name fields.

    Adds:
        - closed_in_last_X_days
        - closed_in_next_Y_days
        - last_holiday_type
        - next_holiday_type
        - days_to_nearest_holiday
        - is_holiday
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------
    # 1. Build Holiday Table
    # -----------------------------
    holiday_df = pd.DataFrame([(pd.to_datetime(d), name) for d, name in holiday_obj.items()],
                              columns=["holiday_date", "holiday_name"]).sort_values("holiday_date")

    # --- Holiday type mapping using pattern rules ---
    def classify_holiday(name):
        name_low = name.lower()
        if "påske" in name_low or "langfredag" in name_low or "skjærtorsdag" in name_low:
            return "Easter"
        if "pinse" in name_low:
            return "Pentecost"
        if "jul" in name_low or "juledag" in name_low:
            return "Christmas"
        if "nyttår" in name_low:
            return "NewYear"
        if "grunnlov" in name_low:
            return "NationalDay"
        if "arbeider" in name_low:
            return "LaborDay"
        return "Other"

    holiday_df["holiday_type"] = holiday_df["holiday_name"].apply(classify_holiday)

    holiday_dates = holiday_df["holiday_date"].values

    # -----------------------------
    # 2. Add Sunday Closures (except last 4 weeks of the year)
    # -----------------------------
    df["day_of_week_num"] = df["date"].dt.weekday  # Monday=0, Sunday=6
    df["iso_week"] = df["date"].dt.isocalendar().week

    sundays = df.loc[(df["day_of_week_num"] == 6) & (df["iso_week"] < 49), "date"].values

    # Combine holiday + Sunday closures
    closed_dates = np.unique(np.concatenate([holiday_dates, sundays]))

    # -----------------------------
    # 3. Closed days before/after (vectorized)
    # -----------------------------
    date_values = df["date"].values.astype("datetime64[D]")

    closed_days = closed_dates.astype("datetime64[D]")

    # Matrix diff (broadcast)
    diff = date_values[:, None] - closed_days[None, :]
    diff_days = diff.astype("timedelta64[D]").astype(int)

    df[f"closed_in_last_{past_days}_days"] = np.sum((diff_days < 0) & (diff_days >= -past_days), axis=1)
    df[f"closed_in_next_{forward_days}_days"] = np.sum((diff_days > 0) & (diff_days <= forward_days), axis=1)

    # -----------------------------
    # 4. Last / Next Holiday Type (Vectorized)
    # -----------------------------
    holiday_dates_only = holiday_df["holiday_date"].values.astype("datetime64[D]")
    diff_holiday = date_values[:, None] - holiday_dates_only[None, :]
    diff_holiday_days = diff_holiday.astype("timedelta64[D]").astype(int)

    # Last holiday before date
    last_idx = np.where(diff_holiday_days >= 0, diff_holiday_days, np.inf).argmin(axis=1)
    has_last = np.isfinite(np.where(diff_holiday_days >= 0, diff_holiday_days, np.inf).min(axis=1))
    df["last_holiday_type"] = np.where(
        has_last,
        holiday_df.iloc[last_idx]["holiday_type"].values,
        "None"
    )

    # Next holiday after date
    next_idx = np.where(diff_holiday_days <= 0, -diff_holiday_days, np.inf).argmin(axis=1)
    has_next = np.isfinite(np.where(diff_holiday_days <= 0, -diff_holiday_days, np.inf).min(axis=1))
    df["next_holiday_type"] = np.where(
        has_next,
        holiday_df.iloc[next_idx]["holiday_type"].values,
        "None"
    )

    # -----------------------------
    # 5. Days to nearest holiday (vectorized)
    # -----------------------------
    abs_diff = np.abs(diff_holiday_days)
    df["days_to_nearest_holiday"] = abs_diff.min(axis=1)

    # -----------------------------
    # 6. Binary holiday indicator
    # -----------------------------
    df["is_holiday"] = df["date"].isin(holiday_df["holiday_date"]).astype(int)

    # Cleanup
    df = df.drop(columns=["iso_week"], errors="ignore")

    return df
