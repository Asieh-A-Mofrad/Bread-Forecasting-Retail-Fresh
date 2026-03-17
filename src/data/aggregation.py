# src/data/aggregation.py
import pandas as pd


def aggregate_to_store_daily(sales_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates product-level daily sales to store-level daily totals.

    After aggregation, the resulting columns contain:
      - total store quantity per day (target)
      - average/combined promotional signals
      - average price/discount
      - store-level closure & holiday indicators
      - calendar features taken once per store/day

    Parameters
    ----------
    sales_daily : pd.DataFrame
        Must contain product-level daily features including:
        ['gln', 'date', 'quantity', 'unit_price', 'discount',
         'on_promotion', 'kronemarked', 'promo_combined',
         'promotion_count', 'closed_in_last_3_days', 'closed_in_next_4_days',
         'is_holiday', 'days_to_nearest_holiday', 'day_of_week_num',
         'sin_dow', 'cos_dow', 'sin_day', 'cos_day', 'month', 'quarter']

    Returns
    -------
    store_daily : pd.DataFrame
        Store-level daily aggregated dataframe.
    """

    # --- Aggregation dictionary ---
    agg_dict = {
        # Target: total store sales
        'quantity': 'sum',

        # Store-level average pricing signals
        'unit_price': 'mean',
        'discount': 'mean',

        # Promotion indicators (fraction of products on promotion)
        'on_promotion': 'mean',
        'kronemarked': 'max',  # whether store had any kronemarked item
        'promo_combined': 'mean',
        'promotion_count': 'mean',

        # Store closures
        'closed_in_last_3_days': 'max',
        'closed_in_next_4_days': 'max',

        # Holiday features
        'is_holiday': 'max',
        'days_to_nearest_holiday': 'min',

        # Calendar features — same for all products that day; take first
        'day_of_week_num': 'first',
        'sin_dow': 'first',
        'cos_dow': 'first',
        'sin_day': 'first',
        'cos_day': 'first',
        'month': 'first',
        'quarter': 'first'
    }

    # --- Perform aggregation ---
    store_daily = sales_daily.groupby(['gln', 'date'], as_index=False).agg(agg_dict)

    # Convert promotion fractions to float (0–1)
    store_daily['on_promotion'] = store_daily['on_promotion'].astype(float)

    # Ensure boolean-like features are integer
    store_daily['kronemarked'] = store_daily['kronemarked'].astype(int)
    store_daily['is_holiday'] = store_daily['is_holiday'].astype(int)

    # Fill missing discount/promotion fields
    store_daily = store_daily.fillna({
        'discount': 0,
        'promotion_count': 0,
        'days_to_nearest_holiday': 0
    })

    # Sanity: ensure date is datetime
    store_daily['date'] = pd.to_datetime(store_daily['date'])

    return store_daily
