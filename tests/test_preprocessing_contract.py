# tests/test_preprocessing_contract.py
import pandas as pd

from src.data import preprocessing as prep


def test_prepare_sales_data_combined_returns_expected_contract(monkeypatch, tmp_path):
    raw_sales = pd.DataFrame({"date": ["2024-01-01"], "gln": [1], "eanCode": [111], "quantity": [5]})
    sales_daily_seed = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "gln": [1], "gb_id": [10], "quantity": [5.0]})
    mismatch_df = pd.DataFrame({"dummy": [1]})
    anomalies_df = pd.DataFrame({"issue": ["dispersion"]})
    suspicious_df = pd.DataFrame({"issue": ["suspicious"]})

    monkeypatch.setattr(prep, "read_data", lambda _p: raw_sales)
    monkeypatch.setattr(prep.pd, "read_csv", lambda _p: pd.DataFrame({"eanCode": [111], "gb_id": [10]}))
    monkeypatch.setattr(prep.pd, "read_excel", lambda _p: pd.DataFrame({"campaign_id": [1]}))
    monkeypatch.setattr(
        prep,
        "preprocess_sales",
        lambda sales_hourly, mapping: (sales_hourly, sales_daily_seed.copy(), anomalies_df, suspicious_df),
    )
    monkeypatch.setattr(prep, "correct_daily_unit_price", lambda df: df.assign(step_price=1))
    monkeypatch.setattr(prep, "calculate_standard_price_and_discount", lambda df: df.assign(step_discount=1))
    monkeypatch.setattr(prep, "prepare_promotion_data",
                        lambda df, campaign, kronemarked: (df.assign(step_promo=1), mismatch_df))
    monkeypatch.setattr(prep, "add_promotion_count", lambda df: df.assign(step_promo_count=1))
    monkeypatch.setattr(prep, "add_lag_features", lambda df: df.assign(quantity_lag_1=4.0))
    monkeypatch.setattr(prep, "add_rolling_mean_features", lambda df: df.assign(quantity_rolling_mean_7=4.5))
    monkeypatch.setattr(prep, "add_holiday_features", lambda df, holiday_obj: df.assign(is_holiday=0))
    monkeypatch.setattr(prep, "add_gb_id_previous_day_feature", lambda df: df.assign(gb_id_prev=9))
    monkeypatch.setattr(prep, "add_seasonality_features", lambda df: df.assign(day_of_week_num=1))

    sales_daily, mismatch = prep.prepare_sales_data_combined(tmp_path)
    assert "day_of_week_num" in sales_daily.columns
    assert mismatch.equals(mismatch_df)

    sales_daily_d, mismatch_d, diagnostics = prep.prepare_sales_data_combined(tmp_path, return_diagnostics=True)
    assert sales_daily_d.shape[0] == 1
    assert mismatch_d.equals(mismatch_df)
    assert set(diagnostics.keys()) == {"sales_hourly", "price_dispersion", "suspicious_daily_prices"}
