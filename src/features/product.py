# src/features/product.py

def add_gb_id_previous_day_feature(df):
    """
    Adds a feature 'gb_id_previous_day' to indicate the quantity of
    the product with gb_id 655629783174417F92A38D18897A69EE that was carried
    over from the previous day.

    This feature helps analyze the effect of varying quantities of discounted
    carried-over products on the current day's sales.

    Method:
    - Sorts data by store ('gln') and date to ensure proper chronological order.
    - Filters for the target product (carried-over item).
    - Shifts the quantity of the carried-over product by one day within each store.
    - Fills missing values with 0 (no carryover from the previous day).

    Parameters:
    df (DataFrame): Input sales data with 'date', 'gln', 'gb_id', and 'quantity'.

    Returns:
    pd.DataFrame: Updated dataframe with the new feature.
    """

    target_gb_id = '655629783174417F92A38D18897A69EE'

    # Sort by store and date to ensure proper shifting
    df = df.sort_values(by=['gln', 'date']).copy()

    # Keep quantity only for the target product, otherwise NaN
    carried_over_qty = df['quantity'].where(df['gb_id'] == target_gb_id)

    # Shift within each store
    df['gb_id_previous_day'] = carried_over_qty.groupby(df['gln']).shift(1).fillna(0)

    return df
