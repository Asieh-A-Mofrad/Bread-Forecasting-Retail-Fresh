# src/analysis/preprocessing_diagnostics.py
"""Diagnostics for preprocessing outputs (EDA + QA checks)."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def summarize_price_anomalies(
        price_dispersion: pd.DataFrame,
        suspicious_daily_prices: pd.DataFrame,
        store_col: str = "gln",
) -> dict:
    """Return compact summary of hourly/daily price anomaly sets."""
    out = {
        "n_hourly_dispersion_cases": int(len(price_dispersion)),
        "n_suspicious_daily_rows": int(len(suspicious_daily_prices)),
    }

    if not price_dispersion.empty and store_col in price_dispersion.columns:
        out["stores_with_hourly_dispersion"] = int(price_dispersion[store_col].nunique())

    if not suspicious_daily_prices.empty and store_col in suspicious_daily_prices.columns:
        out["stores_with_suspicious_daily_prices"] = int(suspicious_daily_prices[store_col].nunique())

    return out


def price_anomaly_cases_by_store(
        price_dispersion: pd.DataFrame,
        suspicious_daily_prices: pd.DataFrame,
        store_col: str = "gln",
) -> pd.DataFrame:
    """Aggregate anomaly counts by store for quick triage."""
    hourly = pd.DataFrame(columns=[store_col, "n_hourly_dispersion"])
    daily = pd.DataFrame(columns=[store_col, "n_suspicious_daily"])

    if not price_dispersion.empty and store_col in price_dispersion.columns:
        hourly = (
            price_dispersion.groupby(store_col)
            .size()
            .rename("n_hourly_dispersion")
            .reset_index()
        )

    if not suspicious_daily_prices.empty and store_col in suspicious_daily_prices.columns:
        daily = (
            suspicious_daily_prices.groupby(store_col)
            .size()
            .rename("n_suspicious_daily")
            .reset_index()
        )

    out = hourly.merge(daily, on=store_col, how="outer").fillna(0)
    for c in ["n_hourly_dispersion", "n_suspicious_daily"]:
        if c in out.columns:
            out[c] = out[c].astype(int)

    if out.empty:
        return out

    out["n_total_price_anomalies"] = out[["n_hourly_dispersion", "n_suspicious_daily"]].sum(axis=1)
    return out.sort_values("n_total_price_anomalies", ascending=False).reset_index(drop=True)


def plot_price_anomaly_examples(
        sales_hourly: pd.DataFrame,
        sales_daily: pd.DataFrame,
        price_dispersion: pd.DataFrame,
        suspicious_daily_prices: pd.DataFrame,
        sample_size: int = 5,
) -> None:
    """Visual examples of hourly dispersion and suspicious daily price trajectories."""
    sns.set_theme(style="whitegrid")

    if not price_dispersion.empty:
        sample = price_dispersion.sample(n=min(sample_size, len(price_dispersion)), random_state=42)
        for _, row in sample.iterrows():
            subset = sales_hourly[
                (sales_hourly["gln"] == row["gln"])
                & (sales_hourly["gb_id"] == row["gb_id"])
                & (pd.to_datetime(sales_hourly["date"]) == pd.to_datetime(row["date"]))
                ].copy()
            if subset.empty:
                continue
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=subset.sort_values("fromHour"), x="fromHour", y="unit_price", marker="o")
            plt.title(
                f"Hourly price variation | gln={row['gln']} gb_id={row['gb_id']} date={pd.to_datetime(row['date']).date()}")
            plt.tight_layout()
            plt.show()

    if not suspicious_daily_prices.empty:
        sample = suspicious_daily_prices.sample(
            n=min(sample_size, len(suspicious_daily_prices)), random_state=42
        )
        for _, row in sample.iterrows():
            subset = sales_daily[(sales_daily["gln"] == row["gln"]) & (sales_daily["gb_id"] == row["gb_id"])].copy()
            if subset.empty:
                continue
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=subset.sort_values("date"), x="date", y="unit_price", marker="o", color="C1")
            plt.title(f"Daily price history | gln={row['gln']} gb_id={row['gb_id']}")
            plt.tight_layout()
            plt.show()


def product_record_counts(
        sales_daily: pd.DataFrame,
        store_col: str = "gln",
        product_col: str = "gb_id",
        date_col: str = "date",
) -> pd.DataFrame:
    """Count rows and active dates per store-product; useful for sparsity screening."""
    required = [store_col, product_col, date_col]
    missing = [c for c in required if c not in sales_daily.columns]
    if missing:
        raise ValueError(f"Missing required columns for product_record_counts: {missing}")

    df = sales_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    out = (
        df.groupby([store_col, product_col], as_index=False)
        .agg(
            n_rows=(date_col, "size"),
            n_active_days=(date_col, "nunique"),
            first_date=(date_col, "min"),
            last_date=(date_col, "max"),
        )
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )

    out["span_days"] = (out["last_date"] - out["first_date"]).dt.days + 1
    out["coverage_ratio"] = out["n_active_days"] / out["span_days"].replace(0, pd.NA)

    return out
