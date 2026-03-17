# src/analysis/eda.py
"""Reusable EDA helpers for forecasting datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dataset_overview(
        df: pd.DataFrame,
        date_col: str = "date",
        store_col: str = "gln",
        product_col: str | None = "gb_id",
) -> dict:
    """Return a compact dataset summary dict."""
    out = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }

    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        out["date_min"] = str(d.min())
        out["date_max"] = str(d.max())
        out["n_dates"] = int(d.nunique(dropna=True))

    if store_col in df.columns:
        out["n_stores"] = int(df[store_col].nunique(dropna=True))

    if product_col and product_col in df.columns:
        out["n_products"] = int(df[product_col].nunique(dropna=True))

    return out


def missingness_table(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Return missing-value table sorted descending."""
    miss = df.isna().sum()
    pct = (miss / max(len(df), 1)) * 100
    out = pd.DataFrame(
        {
            "column": miss.index,
            "missing_count": miss.values,
            "missing_pct": pct.values,
        }
    )
    out = out.sort_values(["missing_count", "column"], ascending=[False, True])
    return out.head(top_n).reset_index(drop=True)


def high_correlation_pairs(
        df: pd.DataFrame,
        threshold: float = 0.90,
        exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return highly correlated numeric feature pairs above threshold."""
    exclude = set(exclude_cols or [])
    num = df.select_dtypes(include=[np.number]).drop(columns=list(exclude), errors="ignore")
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr_abs"])

    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    rows = []
    for col in upper.columns:
        vals = upper[col].dropna()
        for idx, v in vals.items():
            if v >= threshold:
                rows.append((idx, col, float(v)))

    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "corr_abs"])
    if out.empty:
        return out
    return out.sort_values("corr_abs", ascending=False).reset_index(drop=True)


def sales_coverage(
        df: pd.DataFrame,
        store_col: str = "gln",
        date_col: str = "date",
        product_col: str = "gb_id",
) -> pd.DataFrame:
    """Coverage summary by store: date span, active days, and products."""
    if store_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"Expected columns '{store_col}' and '{date_col}'.")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    agg_dict = {
        "date_min": (date_col, "min"),
        "date_max": (date_col, "max"),
        "n_days": (date_col, "nunique"),
        "n_rows": (date_col, "size"),
    }

    if product_col in d.columns:
        agg_dict["n_products"] = (product_col, "nunique")

    out = d.groupby(store_col, as_index=False).agg(**agg_dict)
    out["span_days"] = (out["date_max"] - out["date_min"]).dt.days + 1
    out["coverage_ratio"] = out["n_days"] / out["span_days"].replace(0, np.nan)

    return out.sort_values(store_col).reset_index(drop=True)
