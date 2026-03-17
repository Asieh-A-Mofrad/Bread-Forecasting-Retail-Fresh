# src/features/diagnostics.py
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_price_diagnostics(
        sales_hourly,
        sales_daily,
        price_dispersion,
        suspicious,
        sample_size=5
):
    """
    Visualize pricing anomalies:
      1️⃣ Intra-day price dispersion (hourly variation)
      2️⃣ Suspicious daily price levels (too high/low)

    Parameters:
        sales_hourly (DataFrame): Hourly sales data with
            ['gln', 'gb_id', 'date', 'fromHour', 'unit_price'].
        sales_daily (DataFrame): Aggregated daily sales data with columns
            ['gln', 'gb_id', 'date', 'unit_price'].
        price_dispersion (DataFrame): Output of preprocess_sales() with
            ['gln', 'gb_id', 'date', 'rel_diff'].
        suspicious (DataFrame): Daily data flagged as having implausible prices.
        sample_size (int): Number of random examples to visualize.
    """

    sns.set(style="whitegrid")

    # --- 1️⃣ Plot intra-day dispersion examples ---
    if not price_dispersion.empty:
        high_dispersion = price_dispersion.sample(n=min(sample_size, len(price_dispersion)), random_state=42)
        print(f"📊 Plotting {len(high_dispersion)} intra-day dispersion examples...")

        for _, row in high_dispersion.iterrows():
            subset_hourly = sales_hourly[
                (sales_hourly['gln'] == row['gln']) &
                (sales_hourly['gb_id'] == row['gb_id']) &
                (sales_hourly['date'] == row['date'])
                ].copy()

            if subset_hourly.empty or subset_hourly['unit_price'].isna().all():
                continue

            plt.figure(figsize=(8, 4))
            sns.lineplot(data=subset_hourly.sort_values('fromHour'), x='fromHour', y='unit_price', marker='o',
                         color='C0')
            plt.title(f"Intra-day Price Variation\n"
                      f"Store: {row['gln']} | Product: {row['gb_id']} | Date: {row['date'].date()}", fontsize=11)
            plt.suptitle(f"Relative Dispersion: {row['rel_diff'] * 100:.1f}%", fontsize=9, y=0.94)
            plt.xlabel("Hour of Day")
            plt.ylabel("Unit Price (NOK)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    else:
        print("✅ No intra-day price dispersion cases to visualize.")

    # --- 2️⃣ Plot suspicious price examples ---
    if not suspicious.empty:
        suspicious_sample = suspicious.sample(n=min(sample_size, len(suspicious)), random_state=42)
        print(f"⚠️ Plotting {len(suspicious_sample)} suspicious price examples...")

        for _, row in suspicious_sample.iterrows():
            subset = sales_daily[(sales_daily['gln'] == row['gln']) &
                                 (sales_daily['gb_id'] == row['gb_id'])
                                 ].copy()

            if subset.empty or subset['unit_price'].isna().all():
                continue

            plt.figure(figsize=(8, 4))
            sns.lineplot(data=subset.sort_values('date'), x='date', y='unit_price', marker='o', color='C1')
            plt.title(f"Daily Price History\nStore: {row['gln']} | Product: {row['gb_id']}", fontsize=11)
            plt.ylabel("Daily Unit Price (NOK)")
            plt.xlabel("Date")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    else:
        print("✅ No suspicious daily price rows to visualize.")


def summarize_price_dispersion(price_anomaly_cases):
    summary = (
        price_anomaly_cases.groupby('gln')
        .size()
        .reset_index(name='n_cases')
        .sort_values('n_cases', ascending=False)
    )
    print("Stores with most intra-day price variation:")
    print(summary)
