# src/data/raw_processing.py

def preprocess_sales(
        sales_hourly,
        eanCode_to_gb_id,
        price_dispersion_threshold=0.15):
    """
    Aggregate hourly sales to daily per store/product (gb_id).
    Adds gb_id mapping to hourly data
    Detects data quality issues:
    - Missing mappings,
    - Price dispersion anomalies within a day,
    - Suspiciously high or low daily prices.

    It return:
    sales_hourly: cleaned hourly data with gb_id and unit_price
    sales_daily: aggregated daily sales data
    anomalies: intra-day price dispersion cases
    suspicious: daily rows with outlier prices
    """

    sales_hourly = sales_hourly.copy()
    sales_hourly.drop_duplicates(inplace=True)
    sales_hourly['quantity'] = sales_hourly['quantity'].round().astype(int)

    # --- Step 1: Ensure mapping consistency ---
    mapping = eanCode_to_gb_id[['eanCode', 'gb_id']].drop_duplicates()
    dup_check = (
        eanCode_to_gb_id.groupby('eanCode')['gb_id']
        .nunique()
        .reset_index(name='gb_id_count')
        .query('gb_id_count > 1')
    )
    if not dup_check.empty:
        print(f"Warning: {len(dup_check)} eanCodes map to multiple gb_ids.")
        print(dup_check.head(10))

    # --- Step 2: Merge gb_id into hourly data ---
    sales_hourly = sales_hourly.merge(mapping, on='eanCode', how='left', validate='m:1')

    # --- Step 3: Handle unmapped eanCodes ---
    unmapped = sales_hourly['gb_id'].isna().sum()
    if unmapped:
        missing = set(sales_hourly.loc[sales_hourly['gb_id'].isna(), 'eanCode'])
        print(f"{unmapped} rows have no gb_id mapping.")
        overlap = missing.intersection(
            set(eanCode_to_gb_id['gb_eanCode'])) if 'gb_eanCode' in eanCode_to_gb_id.columns else set()
        if overlap:
            print(f"{len(overlap)} missing eanCodes appear in mapping['gb_eanCode'].")
        else:
            print("None of the missing eanCodes appear in mapping['gb_eanCode'].")
        print(f"Dropped {unmapped} unmapped rows.")

        missing_eans = sales_hourly.loc[~sales_hourly['eanCode'].isin(eanCode_to_gb_id['eanCode']), 'eanCode'].unique()
        print("Unmapped eanCodes:", missing_eans)
        sales_hourly = sales_hourly.dropna(subset=['gb_id'])

    # --- Step 4: Aggregate hourly per gb_id ---
    # Combine all eanCodes belonging to the same gb_id within the same store/hour/date
    sales_hourly = (
        sales_hourly.groupby(['gln', 'gb_id', 'date', 'fromHour', 'toHour'], as_index=False)
        .agg({
            'quantity': 'sum',
            'netSalesKr': 'sum',
            'taxAmount': 'sum',
            'grossMargin': 'sum'
        })
    )

    # Compute unit price after hourly aggregation
    sales_hourly['unit_price'] = (
            (sales_hourly['netSalesKr'] + sales_hourly['taxAmount']) / sales_hourly['quantity']).round(3)

    # --- Step 5: Detect intra-day price dispersion (per product/date/store) ---
    price_dispersion = (
        sales_hourly.groupby(['gln', 'gb_id', 'date'])
        .agg(mean_price=('unit_price', 'mean'),
             std_price=('unit_price', 'std'))
        .reset_index()
    )
    price_dispersion['rel_diff'] = price_dispersion['std_price'] / price_dispersion['mean_price']

    # Round dispersion metrics
    price_dispersion = price_dispersion.round({
        'mean_price': 3,
        'std_price': 3,
        'rel_diff': 3
    })

    anomalies = price_dispersion.query('rel_diff > @price_dispersion_threshold')

    if len(anomalies):
        print(
            f"Found {len(anomalies)} intra-day price dispersion cases (> {price_dispersion_threshold * 100:.0f}% difference).")

    # --- Step 6: Aggregate to daily ---
    sales_daily = (
        sales_hourly.groupby(['gln', 'gb_id', 'date'], as_index=False)
        .agg({
            'quantity': 'sum',
            'netSalesKr': 'sum',
            'grossMargin': 'sum',
            'taxAmount': 'sum'
        })
    )

    for df in [sales_daily, sales_hourly]:
        df[['netSalesKr', 'taxAmount', 'grossMargin']] = df[['netSalesKr', 'taxAmount', 'grossMargin']].round(3)

    sales_daily['unit_price'] = (
            (sales_daily['netSalesKr'] + sales_daily['taxAmount']) / sales_daily['quantity']).round(3)

    # --- Step 7: Flag suspicious daily prices ---
    suspicious = sales_daily.query('unit_price < 5 or unit_price > 65')
    if len(suspicious):
        print(f"Found {len(suspicious)} suspicious price rows.")

    return sales_hourly, sales_daily, anomalies, suspicious
