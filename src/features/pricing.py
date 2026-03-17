# src/features/pricing.py
import pandas as pd
import numpy as np


def correct_daily_unit_price(
        sales_daily,
        low_price_threshold=5,
        high_price_threshold=65,
        days_window=30
):
    """
    Simple correction of daily unit prices that fall outside allowed range.
    Correction = replace with a reference price (median over window).

    Reference price hierarchy:
        1. median price for (gln, gb_id) over last X days
        2. median price for (gb_id) across all stores
        3. global product median over entire dataset
        4. if unavailable → leave as is but mark as 'uncorrected'

    Parameters
    ----------
    sales_daily : DataFrame
        Must include ['date','gln','gb_id','unit_price'].
    low_price_threshold : float
        Minimum acceptable price (default 5).
    high_price_threshold : float
        Maximum acceptable price (default 65).
    days_window : int
        How many days back to search for reference prices.

    Returns
    -------
    DataFrame with:
        - corrected_unit_price
        - price_corrected_flag (bool)
        - price_correction_source (str)
    """

    df = sales_daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Identify outliers
    df["price_error"] = (df["unit_price"] < low_price_threshold) | (df["unit_price"] > high_price_threshold)

    if df["price_error"].sum() == 0:
        # No correction needed
        df["corrected_unit_price"] = df["unit_price"]
        df["price_corrected_flag"] = False
        df["price_correction_source"] = None
        return df

    # -----------------------------
    # Build reference price tables
    # -----------------------------
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=days_window)
    recent = df[df["date"] >= cutoff]

    # 1) Store-product median
    ref_store = (recent.groupby(["gln", "gb_id"])["unit_price"]
                 .median()
                 .rename("ref_store_price"))

    # 2) Global-product median
    ref_global = (df.groupby("gb_id")["unit_price"]
                  .median()
                  .rename("ref_global_price"))

    # Merge references
    df = df.merge(ref_store, on=["gln", "gb_id"], how="left")
    df = df.merge(ref_global, on="gb_id", how="left")

    # Final fallback: product-wide median computed earlier
    df["fallback_price"] = df.groupby("gb_id")["unit_price"].transform("median")

    # Assemble a final reference price
    df["ref_price"] = (df["ref_store_price"]
                       .combine_first(df["ref_global_price"])
                       .combine_first(df["fallback_price"]))

    # -----------------------------
    # Apply correction only to errors
    # -----------------------------
    df["corrected_unit_price"] = df["unit_price"]

    mask = df["price_error"] & df["ref_price"].notna()

    df.loc[mask, "corrected_unit_price"] = df.loc[mask, "ref_price"]
    df["price_corrected_flag"] = mask

    # Label correction source
    df["price_correction_source"] = np.select(
        [mask & df["ref_store_price"].notna(),
         mask & df["ref_store_price"].isna() & df["ref_global_price"].notna(),
         mask & df["ref_global_price"].isna() & df["fallback_price"].notna(),
         df["price_error"],  # error but no correction
         ],
        [
            "store_median",
            "global_median",
            "fallback_median",
            "uncorrected_missing_reference"
        ],
        default=None
    )

    return df


# --------------
def calculate_standard_price_and_discount(
        sales_daily,
        initial_rolling_window=30,
        promo_start_threshold=0.90,  # start promo if price < 90% of standard
        promo_end_threshold=0.97,  # end promo if price > 97% of standard
        small_fluctuation=0.05  # ignore changes smaller than 5%
):
    """
    Estimate standard price dynamically and identify promotions using hysteresis logic.

    Parameters
    ----------
    sales_daily : pd.DataFrame
        Input dataframe with at least ['gb_id', 'date', 'price'] columns.
    initial_rolling_window : int, default=30
        Rolling window size (in days) for initial standard price estimation.
    promo_start_threshold : float, default=0.94
        Fraction of standard price below which a promotion starts.
    promo_end_threshold : float, default=0.97
        Fraction of standard price above which a promotion ends.
    small_fluctuation : float, default=0.01
        Ignore very small price variations within ±1%.

    Returns
    -------
    pd.DataFrame
        Updated dataframe with columns:
        ['standard_price', 'discount', 'on_promotion']
    """

    sales_daily = sales_daily.copy()
    sales_daily.sort_values(by=['gb_id', 'date'], inplace=True)

    # --- Step 1: Estimate a baseline standard price per product using rolling max ---
    sales_daily['rolling_max_price'] = (
        sales_daily.groupby('gb_id')['unit_price']
        .transform(lambda x: x.rolling(initial_rolling_window, min_periods=1)
                   .apply(lambda w: np.mean(sorted(w, reverse=True)[:5]), raw=False))
    )

    # Initialize columns
    sales_daily['standard_price'] = np.nan
    sales_daily['discount'] = 0.0
    sales_daily['on_promotion'] = False

    # --- Step 2: Apply hysteresis logic per product ---
    for gb_id, group in sales_daily.groupby('gb_id', sort=False):
        group = group.copy()
        in_promo = False
        standard_price = None

        for i in range(len(group)):
            idx = group.index[i]
            price = group.loc[idx, 'unit_price']
            roll_max = group.loc[idx, 'rolling_max_price']

            # Initialize standard price if missing
            if standard_price is None:
                standard_price = roll_max

            price_ratio = price / standard_price if standard_price > 0 else 1.0

            # --- Hysteresis conditions ---
            if not in_promo:
                # Possible promo start
                if price_ratio < promo_start_threshold:
                    in_promo = True
                    group.at[idx, 'on_promotion'] = True
                    group.at[idx, 'discount'] = price_ratio - 1.0
                elif abs(price_ratio - 1) > small_fluctuation:
                    # update standard price if normal price change
                    standard_price = price
                    group.at[idx, 'discount'] = 0.0
                else:
                    group.at[idx, 'discount'] = 0.0
                group.at[idx, 'standard_price'] = standard_price

            else:
                # Currently in promo period
                group.at[idx, 'on_promotion'] = True
                group.at[idx, 'discount'] = price_ratio - 1.0

                # Check for promo end
                if price_ratio > promo_end_threshold:
                    in_promo = False
                    standard_price = price
                    group.at[idx, 'on_promotion'] = False
                    group.at[idx, 'discount'] = 0.0

                group.at[idx, 'standard_price'] = standard_price

        sales_daily.loc[group.index, ['standard_price', 'discount', 'on_promotion']] = \
            group[['standard_price', 'discount', 'on_promotion']]

    # --- Step 3: Clean up ---
    sales_daily['on_promotion'] = sales_daily['on_promotion'].astype(bool)
    sales_daily.drop(columns=['rolling_max_price'], inplace=True)

    # Remove floating point noise — treat very small values as zero
    tolerance = 1e-6  # acceptable rounding error threshold
    sales_daily.loc[np.abs(sales_daily['discount']) < tolerance, 'discount'] = 0.0

    # Also enforce rule: if not on promotion, discount must be exactly 0
    sales_daily.loc[~sales_daily['on_promotion'], 'discount'] = 0.0
    sales_daily['discount'] = sales_daily['discount'].round(4)

    return sales_daily
