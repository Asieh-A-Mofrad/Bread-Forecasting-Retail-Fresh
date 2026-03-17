# src/total.py
from src.data.preprocessing import prepare_sales_data_combined
from src.data.aggregation import aggregate_to_store_daily
from src.features.store_temporal import add_store_temporal_features
from src.utils.splitting import split_by_time
from src.models.train_xgb import train_xgb_model, load_model_and_test
from src.models.evaluate import plot_error_analysis

sales_daily, mismatch = prepare_sales_data_combined(bread_path)

store_daily = aggregate_to_store_daily(sales_daily)
store_daily = add_store_temporal_features(store_daily)

train_data, test_data = split_by_time(store_daily, test_start="2024-01-01", test_end="2024-03-31")

drop_cols = ["date", "gln", "gb_id", "keep", "year_month"]
train_data = train_data.drop(columns=drop_cols, errors="ignore")
test_data = test_data.drop(columns=drop_cols, errors="ignore")

feature_cols = [c for c in train_data.columns if c != "quantity"]

train_xgb_model(train_data, feature_cols, model_path="Models/XGBoost_total.pkl")

results = load_model_and_test(test_data, feature_cols, model_path="Models/XGBoost_total.pkl")

plot_error_analysis(results["y_test"], results["y_pred"])
