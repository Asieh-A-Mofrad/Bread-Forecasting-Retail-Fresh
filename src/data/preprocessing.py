# src/data/preprocessing.py
from pathlib import Path
import pandas as pd
import holidays

from src.data.io import read_data
from src.features.pricing import (
    correct_daily_unit_price,
    calculate_standard_price_and_discount
)
from src.features.promotions import (
    prepare_promotion_data,
    add_promotion_count
)
from src.features.temporal import (
    add_lag_features,
    add_rolling_mean_features
)
from src.features.holidays import add_holiday_features
from src.features.seasonality import add_seasonality_features
from src.features.product import add_gb_id_previous_day_feature

from src.data.raw_processing import preprocess_sales

bread_path = "../data/Bread/"


def prepare_sales_data_combined(bread_path, return_diagnostics: bool = False):
    """
    Full preprocessing pipeline from raw hourly sales to feature-rich daily product data.
    """
    bread_path = Path(bread_path)

    # --------------------------------------------------
    # 1️⃣ Load raw data
    # --------------------------------------------------
    sales_hourly = read_data(bread_path / "sales_hourly")

    eanCode_to_gb_id = pd.read_csv(bread_path / "production_product" / "eanCode_to_gb_id.csv")
    sales_hourly_gb, sales_daily, anomalies, suspicious = preprocess_sales(sales_hourly, eanCode_to_gb_id)

    # --------------------------------------------------
    # 2️⃣ Promotions & pricing
    # --------------------------------------------------
    campaign = pd.read_excel(bread_path / "promotions" / "bread_campaigns.xlsx")
    kronemarked = pd.read_csv(bread_path / "promotions" / "kronemarked_timeline.csv")

    sales_daily = correct_daily_unit_price(sales_daily)
    sales_daily = calculate_standard_price_and_discount(sales_daily)
    sales_daily, mismatch = prepare_promotion_data(sales_daily, campaign, kronemarked)

    # --------------------------------------------------
    # 3️⃣ Temporal & calendar features
    # --------------------------------------------------
    sales_daily = add_promotion_count(sales_daily)
    sales_daily = add_lag_features(sales_daily)
    sales_daily = add_rolling_mean_features(sales_daily)

    holiday_years = range(sales_daily["date"].min().year, sales_daily["date"].max().year + 1)
    holiday_obj = holidays.Norway(years=holiday_years)
    sales_daily = add_holiday_features(sales_daily, holiday_obj)

    sales_daily = add_gb_id_previous_day_feature(sales_daily)
    sales_daily = add_seasonality_features(sales_daily)

    if return_diagnostics:
        diagnostics = {
            "sales_hourly": sales_hourly,
            "price_dispersion": anomalies,
            "suspicious_daily_prices": suspicious,
        }
        return sales_daily, mismatch, diagnostics

    return sales_daily, mismatch
