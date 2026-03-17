# src/analysis/campaign_diagnostics.py
"""Reusable diagnostics for campaign/promotions data consistency."""

from __future__ import annotations

import pandas as pd


def expand_campaign_dates(
        campaign_df: pd.DataFrame,
        gb_col: str = "gb_id",
        from_col: str = "fromDate",
        to_col: str = "toDate",
        keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Expand campaign intervals into daily rows."""
    keep_cols = keep_cols or []

    c = campaign_df.copy()
    c[from_col] = pd.to_datetime(c[from_col], errors="coerce")
    c[to_col] = pd.to_datetime(c[to_col], errors="coerce")
    c = c.dropna(subset=[gb_col, from_col, to_col])

    rows = []
    for _, r in c.iterrows():
        days = pd.date_range(r[from_col], r[to_col], freq="D")
        base = {gb_col: r[gb_col]}
        for k in keep_cols:
            if k in r.index:
                base[k] = r[k]
        for d in days:
            row = base.copy()
            row["date"] = d
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[gb_col, "date", *keep_cols])

    return pd.DataFrame(rows)


def campaign_discount_consistency(
        campaign_df: pd.DataFrame,
        standard_col: str = "standardPrice",
        campaign_price_col: str = "campaignPrice",
        campaign_discount_col: str = "discount",
        tolerance: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare declared discount with price-implied discount.

    Returns:
    - full comparison dataframe
    - mismatches above tolerance
    """
    c = campaign_df.copy()

    required = [standard_col, campaign_price_col]
    missing = [x for x in required if x not in c.columns]
    if missing:
        raise ValueError(f"Missing required campaign columns: {missing}")

    c[standard_col] = pd.to_numeric(c[standard_col], errors="coerce")
    c[campaign_price_col] = pd.to_numeric(c[campaign_price_col], errors="coerce")

    c["discount_implied"] = -(1 - (c[campaign_price_col] / c[standard_col].replace(0, pd.NA)))

    if campaign_discount_col in c.columns:
        c[campaign_discount_col] = pd.to_numeric(c[campaign_discount_col], errors="coerce")
        c["discount_diff_abs"] = (c[campaign_discount_col] - c["discount_implied"]).abs()
        mismatches = c[c["discount_diff_abs"] > tolerance].copy()
    else:
        c["discount_diff_abs"] = pd.NA
        mismatches = pd.DataFrame(columns=c.columns)

    return c, mismatches


def compare_campaign_to_sales_promotions(
        sales_daily: pd.DataFrame,
        expanded_campaign_daily: pd.DataFrame,
        gb_col: str = "gb_id",
        date_col: str = "date",
        promo_flag_col: str = "on_promotion",
) -> pd.DataFrame:
    """Compare campaign-derived promotion days with observed promotion flags in sales data."""
    s = sales_daily.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")

    c = expanded_campaign_daily.copy()
    c[date_col] = pd.to_datetime(c[date_col], errors="coerce")
    c["on_campaign"] = 1

    merged = s.merge(c[[gb_col, date_col, "on_campaign"]], on=[gb_col, date_col], how="left")
    merged["on_campaign"] = merged["on_campaign"].fillna(0).astype(int)

    if promo_flag_col not in merged.columns:
        merged[promo_flag_col] = 0

    summary = (
        merged.groupby(["on_campaign", promo_flag_col], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values("n_rows", ascending=False)
    )

    return summary
