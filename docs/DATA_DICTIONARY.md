# Data Dictionary

This file documents dataset-level schemas used by notebook workflows.

## `sales_daily.parquet`

Grain: one row per `gln` x `gb_id` x `date`.

Core identifiers and targets:

- `gln`: store identifier
- `gb_id`: product identifier
- `date`: calendar date
- `quantity`: observed daily sold quantity

Common pricing and promotion fields:

- `unit_price`: observed unit price
- `corrected_unit_price`: cleaned/corrected unit price
- `standard_price`: estimated reference price
- `discount`: relative discount vs standard price
- `on_promotion`: derived promotion flag
- `kronemarked`: kronemarked period flag
- `promo_combined`: merged promotion indicator from sources
- `promotion_count`: count of promoted products per store-day

Common calendar/store status fields:

- `day_of_week_num`, `month`, `quarter`
- `sin_dow`, `cos_dow`, `sin_day`, `cos_day`
- `is_holiday`, `days_to_nearest_holiday`
- `closed_in_last_3_days`, `closed_in_next_4_days`

## `store_daily.parquet`

Grain: one row per `gln` x `date`.

Target:

- `quantity`: store total daily quantity (sum of product quantity)

Aggregated features (selected):

- price/promo aggregates: `unit_price`, `discount`, `on_promotion`, `kronemarked`, `promo_combined`, `promotion_count`
- holiday/calendar: `is_holiday`, `days_to_nearest_holiday`, `day_of_week_num`, `sin_dow`, `cos_dow`, `sin_day`,
  `cos_day`, `month`, `quarter`
- temporal features: `sales_lag_1`, `sales_lag_7`, `rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`,
  `rolling_std_30`, `days_since_last_kronemarked`, `days_since_last_promotion`
- encoding feature used in notebooks: `gln_te`

## `share_daily.parquet`

Grain: one row per `gln` x `gb_id` x `date`.

Target:

- `target`: product share of store total (`quantity / store_total`)

Model feature examples:

- relative pricing/promo: `discount_relative`, `price_relative`, `promo_boost`
- temporal/rolling: `avg_last_7_sales`, `std_last_7_sales`, `promotion_days_count_last_7`, `rolling_mean`,
  `rolling_mean_all`, `promotion_days_count_rolling`
- encodings: `gln_te`, `gb_id_mean_target`

## `direct_daily.parquet`

Grain: one row per `gln` x `gb_id` x `date`.

Target:

- `target`: direct product quantity (typically equals `quantity`)

Model feature examples:

- relative pricing/promo: `discount_relative`, `price_relative`, `promo_boost`
- temporal/rolling: same family as `share_daily`
- encodings: `gln_te`, `gb_id_mean_target`

## Leakage Guardrails

- `store_total` may be used internally to compute share target, but must not be a model feature for direct quantity
  forecasting.
- Target encodings (`gln_te`, `gb_id_mean_target`) should be fit on train folds only and then applied to
  validation/test.
- Temporal rolling features must be shifted to use only historical observations.
