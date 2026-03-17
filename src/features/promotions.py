# src/features/promotions.py
import pandas as pd


def add_promotion_features_combined(sales_df, campaign_df, kronemarked_df):
    """
    Adds promotion-related features to the sales data.

    Parameters:
        sales_df (DataFrame): Sales data.
        campaign_df (DataFrame): Campaign data.
        kronemarked_df (DataFrame): Kronemarked periods.

    Returns:
        DataFrame: Updated sales data with promotion and kronemarked features.
    """

    required_sales_cols = {'gb_id', 'date', 'gln', 'quantity', 'discount', 'standard_price'}
    required_campaign_cols = {'gb_id', 'fromDate', 'toDate', 'discount', 'standardPrice', 'campaignPrice'}
    required_kronemarked_cols = {'fromDate', 'toDate', 'kronemarked'}

    if not required_sales_cols.issubset(sales_df.columns):
        raise ValueError("Missing required columns in sales_df.")
    if not required_campaign_cols.issubset(campaign_df.columns):
        raise ValueError("Missing required columns in campaign_df.")
    if not required_kronemarked_cols.issubset(kronemarked_df.columns):
        raise ValueError("Missing required columns in kronemarked_df.")

    # Normalize date columns
    sales_df['date'] = pd.to_datetime(sales_df['date']).dt.normalize()
    campaign_df['fromDate'] = pd.to_datetime(campaign_df['fromDate']).dt.normalize()
    campaign_df['toDate'] = pd.to_datetime(campaign_df['toDate']).dt.normalize()
    kronemarked_df['fromDate'] = pd.to_datetime(kronemarked_df['fromDate']).dt.normalize()
    kronemarked_df['toDate'] = pd.to_datetime(
        kronemarked_df['toDate']).dt.normalize()  # Ensure date columns are datetime and normalized
    campaign_df['fromDate'] = pd.to_datetime(campaign_df['fromDate']).dt.normalize()
    campaign_df['toDate'] = pd.to_datetime(campaign_df['toDate']).dt.normalize()
    sales_df['date'] = pd.to_datetime(sales_df['date']).dt.normalize()

    valid_campaigns = campaign_df.dropna(subset=['fromDate', 'toDate']).copy()

    expanded_campaigns = valid_campaigns.apply(
        lambda row: pd.DataFrame({
            'gb_id': [row['gb_id']] * ((row['toDate'] - row['fromDate']).days + 1),
            'date': pd.date_range(row['fromDate'], row['toDate']),
            'Discount': row['discount'],
            'standardPrice': row['standardPrice'],
            'campaignPrice': row['campaignPrice'],
        }), axis=1
    )

    expanded_campaigns = pd.concat(expanded_campaigns.values, ignore_index=True)

    # Get the unique combinations of gb_id and date in sales_df
    unique_combinations = sales_df[['gb_id', 'date']].drop_duplicates()

    # Merge with expanded_campaigns to filter
    filtered_campaigns = expanded_campaigns.merge(unique_combinations, on=['gb_id', 'date'])

    # Merge sales_daily with filtered_campaigns
    merged_data = sales_df.merge(
        filtered_campaigns,
        on=['gb_id', 'date'],
        how='left',  # Left merge keeps all rows from sales_df
        suffixes=('', '_campaign')  # Add suffix for columns from filtered_campaigns
    )

    # Add 'on_promotion' column
    merged_data['OnPromotion'] = ~merged_data['Discount'].isna()  # True if 'discount' is not NaN

    # Replace NaN discount with 0 where there is no campaign
    merged_data['Discount'] = merged_data['Discount'].fillna(0)
    merged_data['standardPrice'] = merged_data['standardPrice'].fillna(0)
    merged_data['campaignPrice'] = merged_data['campaignPrice'].fillna(0)

    # Optionally, ensure on_promotion is boolean
    merged_data['OnPromotion'] = merged_data['OnPromotion'].astype(bool)

    # Expand kronemarked dates
    kronemarked_df = kronemarked_df.dropna(subset=['fromDate', 'toDate']).copy()
    expanded_kronemarked = kronemarked_df.apply(
        lambda row: pd.DataFrame({
            'date': pd.date_range(row['fromDate'], row['toDate']),
            'kronemarked': row['kronemarked']
        }), axis=1
    )
    expanded_kronemarked = pd.concat(expanded_kronemarked.values, ignore_index=True)

    # Merge with sales data
    merged_data = merged_data.merge(expanded_kronemarked, on='date', how='left')

    # Fill NaNs in kronemarked with 0 and ensure boolean
    merged_data['kronemarked'] = merged_data['kronemarked'].fillna(0).astype(bool)

    return merged_data


def adjust_sales_discount(
        sales_df,
        discount_tolerance=1e-6,
        conflict_threshold=0.05,
        min_discount_for_promo=0.01,
):
    """
    Adjusts and reconciles discounts using both recorded (OnPromotion) and derived (on_promotion) flags.

    Logic:
        - If either OnPromotion OR on_promotion is True → treat as promotion.
        - Replace standard_price only if price < standardPrice.
        - Remove floating-point noise and ensure consistent discount calculations.

    Parameters:
        sales_df (pd.DataFrame): DataFrame with:
            ['gln', 'gb_id', 'date', 'price', 'Discount',
             'standard_price', 'standardPrice', 'campaignPrice',
             'OnPromotion', 'on_promotion']
        discount_tolerance (float): Threshold for rounding residuals (default=1e-6)
        conflict_threshold (float): Threshold for detecting discount mismatches (default=0.05)
        min_discount_for_promo (float): Minimum discount magnitude to consider a valid promotion (default=1%)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            (processed_df, mismatch_cases)
    """

    df = sales_df.copy()

    # --- Combine both promotion indicators ---
    df['promo_combined'] = df['OnPromotion'] | df['on_promotion']

    # Round relevant numeric columns
    for col in ['unit_price', 'standard_price', 'standardPrice']:
        df[col] = df[col].round(2)

    # Identify mismatched standard prices during promotions
    mismatch_cases = df.loc[(df['promo_combined']) & (df['standard_price'] != df['standardPrice']),
    ['gln', 'gb_id', 'date', 'standard_price', 'standardPrice']]

    # Replace standard_price only if current price < standardPrice (actual discount)
    mask_replace = (df['promo_combined']
                    & (df['standard_price'] != df['standardPrice'])
                    & (df['unit_price'] < df['standardPrice']))
    df.loc[mask_replace, 'standard_price'] = df['standardPrice']

    # Compute discount relative to standard price
    df['discount'] = (df['unit_price'] - df['standard_price']) / df['standard_price']

    # Remove tiny floating-point errors
    df.loc[df['discount'].abs() < discount_tolerance, 'discount'] = 0.0

    # Recompute derived promotion indicator based on price difference
    df['on_promotion_recomputed'] = df['discount'] <= -min_discount_for_promo

    # Ensure discount = 0 only if neither source nor computed says it's a promotion
    df.loc[~df['promo_combined'] & ~df['on_promotion_recomputed'], 'discount'] = 0.0

    # Detect major mismatches between calculated and reported discount
    df['conflict'] = (df['discount'] - df['Discount']).abs() > conflict_threshold

    # Round for readability
    df['discount'] = df['discount'].round(4)

    # Drop unnecessary columns
    df = df.drop(columns=['Discount', 'standardPrice', 'campaignPrice', 'OnPromotion']).reset_index(drop=True)

    return df, mismatch_cases


# ------------------------
def prepare_promotion_data(sales_df, campaign_df, kronemarked_df):
    merged = add_promotion_features_combined(sales_df, campaign_df, kronemarked_df)
    cleaned, mismatch = adjust_sales_discount(merged)
    return cleaned, mismatch


# --------------------------
def add_promotion_count(df):
    """
    Adds a column 'promotion_count' that reflects the number of products on promotion for each store and date.

    Parameters:
        df (DataFrame): Input sales data containing 'date', 'gln', 'discount', and 'on_promotion' columns.

    Returns:
        DataFrame: Updated DataFrame with the 'promotion_count' column added.
    """
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Count products on promotion (where on_promotion is True) per (date, gln)
    promotion_counts = df[df['on_promotion']].groupby(['date', 'gln']).size()

    # Merge the count back to the original dataframe
    df = df.merge(promotion_counts.rename('promotion_count'), on=['date', 'gln'], how='left')

    # Fill missing values with 0 (if no product had a promotion on that date for that store)
    df['promotion_count'] = df['promotion_count'].fillna(0).astype(int)

    return df
