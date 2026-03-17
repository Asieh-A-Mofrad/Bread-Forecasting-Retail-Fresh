# Feature Catalog

This file documents major feature groups, expected formulas, and forecast-time availability.

## 1) Temporal Features

Store-level (`store_daily`):

- `sales_lag_1`, `sales_lag_7`
- `rolling_mean_7`, `rolling_std_7`
- `rolling_mean_30`, `rolling_std_30`
- `days_since_last_kronemarked`, `days_since_last_promotion`

Product-level (`share_daily`, `direct_daily`):

- `avg_last_7_sales`, `std_last_7_sales`, `promotion_days_count_last_7`
- `rolling_mean` (weekday-conditioned)
- `rolling_mean_all`
- `promotion_days_count_rolling`

Leakage note:

- all rolling/lag features must use shifted history only (no current-day target leakage).

## 2) Promotion and Price Features

- `standard_price`, `corrected_unit_price`
- `discount`
- `on_promotion`, `kronemarked`, `promo_combined`
- `promotion_count`
- `discount_relative`
- `price_relative = (corrected_unit_price - store_mean_price) / (store_mean_price + 1e-6)`
- `promo_boost = on_promotion * discount`

Forecast-time availability:

- only include features that can be known or forecasted at prediction time for the chosen horizon.

## 3) Target Encodings

- `gln_te`: mean target by store (`gln`) from training data.
- `gb_id_mean_target`: mean target by product (`gb_id`) from training data.

Leakage note:

- encodings must be fit within train folds only.
- do not fit encoding maps using validation/test or full data.

## 4) Calendar and Seasonality

- `day_of_week_num`, `month`, `quarter`
- `sin_dow`, `cos_dow`, `sin_day`, `cos_day`
- `is_holiday`, `days_to_nearest_holiday`

These are forecast-safe if generated from known calendar tables.

## 5) Horizon-Specific Feature Policy

For horizon-based notebooks:

- ensure each feature is available at origin for each D+H prediction
- avoid direct use of realized future promotions/prices unless they are planned inputs
- keep per-horizon evaluation tables to detect leakage or drift by horizon bucket

## 6) Minimal Feature De-duplication (XGB via `select_features`)
To reduce highly correlated features without major behavior changes, the training
pipeline drops a small set of redundant columns when both are present:
- `on_promotion` (kept: `promo_combined`)
- `quarter` (kept: `month`)
- `rolling_mean_30` (kept: `rolling_mean_7`)

Rationale: these pairs show very high correlation in `notebooks/00_eda.ipynb`.

## 7) Feature-Pruning Compare (Importance-Based)
To experiment with smaller feature sets based on importance:
- Notebook cells: `notebooks/02_train_models.ipynb` and `notebooks/02_train_models_horizon.ipynb`
- CLI: `scripts/feature_prune_compare.py`

The compare supports `TOP_K` or `IMPORTANCE_THRESHOLD` to select features,
then reports baseline vs pruned metrics side by side.
