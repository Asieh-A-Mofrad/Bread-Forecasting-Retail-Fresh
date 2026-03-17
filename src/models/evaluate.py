# src/models/evaluate.py
import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from src.utils.metrics import calculate_error_metrics


def _normalize_group_key(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def plot_sales_per_month(sales_daily):
    """
    Plots total sales per month per store using a line plot.
    
    Parameters:
        sales_daily (DataFrame): Data containing 'date', 'gln', and 'quantity' columns.
    """
    sales_daily = sales_daily.copy()

    # --- Ensure date and clean NaN/inf ---
    sales_daily['date'] = pd.to_datetime(sales_daily['date'], errors='coerce')
    sales_daily.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    sales_daily.dropna(subset=['date', 'quantity'], inplace=True)

    # --- Aggregate monthly ---
    sales_daily['month'] = sales_daily['date'].dt.to_period('M')
    sales_per_month = (
        sales_daily.groupby(['month', 'gln'])['quantity']
        .sum()
        .reset_index()
    )
    sales_per_month['month'] = sales_per_month['month'].astype(str)

    # --- Distinct color palette for up to 11 stores ---
    n_stores = sales_per_month['gln'].nunique()
    if n_stores <= 10:
        palette = sns.color_palette("tab10", n_stores)
    elif n_stores <= 12:
        palette = sns.color_palette("Set3", n_stores)
    else:
        # Fallback for many stores
        palette = sns.color_palette("husl", n_stores)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=sales_per_month, x='month', y='quantity', hue='gln', marker='o', palette=palette)

    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Total Sales Quantity", fontsize=12)
    plt.title("Total Sales per Month per Store", fontsize=14, pad=10)
    plt.legend(title="Store (gln)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_sales_share_per_product(sales_daily, time_unit='M', start_date=None):
    """
    Plots sales share by product as a stacked area plot per store.

    Parameters:
        sales_daily (DataFrame): Data containing 'date', 'gln', 'gb_id', and 'quantity' columns.
        time_unit (str): 'M' for monthly aggregation, 'D' for daily aggregation.
        start_date (str, optional): If time_unit='D', specify a start date (YYYY-MM-DD).

    Returns:
        None: Displays separate plots for each store.
    """
    # Ensure date column is datetime
    sales_daily['date'] = pd.to_datetime(sales_daily['date'])

    # Get unique stores
    stores = sales_daily['gln'].unique()

    for store in stores:
        df_store = sales_daily[sales_daily['gln'] == store].copy()

        if time_unit == 'M':  # Monthly aggregation
            df_store['time_period'] = df_store['date'].dt.to_period('M')
        elif time_unit == 'D' and start_date:  # Daily aggregation for 1 month
            start_date = pd.to_datetime(start_date)
            end_date = start_date + pd.Timedelta(days=30)
            df_store = df_store[(df_store['date'] >= start_date) & (df_store['date'] < end_date)]
            df_store['time_period'] = df_store['date'].dt.to_period('D')  # Keep daily resolution
        else:
            raise ValueError("Invalid time_unit. Use 'M' for monthly or 'D' with a valid start_date.")

        # Aggregate quantity per product and time period
        df_pivot = df_store.groupby(['time_period', 'gb_id'])['quantity'].sum().unstack().fillna(0)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.stackplot(df_pivot.index.astype(str), df_pivot.T, labels=df_pivot.columns, alpha=0.7)
        plt.title(f"Sales Share per Product - Store {store} ({'Monthly' if time_unit == 'M' else 'Daily'})")
        plt.xlabel("Time Period")
        plt.ylabel("Sales Quantity")
        # plt.legend(title="Product (gb_id)", loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


def plot_discount_trends(sales_daily):
    """
    Plots the trend of discount categories over time using a line plot.

    Parameters:
        sales_daily (DataFrame): Data containing 'date', 'gln', and 'discount' columns.

    Returns:
        None: Displays the plot.
    """
    # Ensure 'date' is in datetime format
    sales_daily['date'] = pd.to_datetime(sales_daily['date'])

    # Create discount categories
    sales_daily['discount_category'] = pd.cut(
        sales_daily['discount'],
        bins=[-1, -0.30, -0.15, -0.05],  # Adjust bin ranges if needed
        labels=['Large Discount (-30%)', 'Medium Discount (-15%)', 'Small Discount']
    )

    # Count number of discounts per store and category
    discount_counts = sales_daily.groupby(['date', 'gln', 'discount_category']).size().reset_index(name='count')

    # Plot using sns.lineplot to show trends over time
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=discount_counts, x='date', y='count', hue='discount_category', marker="o")

    plt.title("Number of Discounts Per Period (Per Store)")
    plt.xlabel("Date")
    plt.ylabel("Number of Discounts")
    plt.legend(title="Discount Level")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_monthly_sales_share(df, month_str, store_col='gln', product_col='gb_id', quantity_col='quantity'):
    """
    Plots stacked area chart of product sales share per store for a chosen month.
    
    Parameters:
        df (pd.DataFrame): Sales dataframe with 'date', store, product, and quantity.
        month_str (str): Month string in 'YYYY-MM' format.
        store_col (str): Column representing store.
        product_col (str): Column representing product.
        quantity_col (str): Column representing quantity sold.
    """
    df['date'] = pd.to_datetime(df['date'])
    df_month = df[df['date'].dt.to_period('M') == month_str]

    if df_month.empty:
        print(f"No data for the month: {month_str}")
        return

    # Group by date, store, and product to get daily sales
    grouped = df_month.groupby(['date', store_col, product_col])[quantity_col].sum().reset_index()

    # Plot per store
    stores = grouped[store_col].unique()
    for store in stores:
        df_store = grouped[grouped[store_col] == store]

        pivot = df_store.pivot(index='date', columns=product_col, values=quantity_col).fillna(0)

        # Convert to share
        pivot_share = pivot  # .div(pivot.sum(axis=1), axis=0)

        plt.figure(figsize=(12, 6))
        pivot_share.plot.area(stacked=True, ax=plt.gca(), cmap='tab20')
        plt.title(f"Product Sales Share - Store {store} - {month_str}")
        plt.xlabel("Date")
        plt.ylabel("Sales Share")
        plt.legend(title="Product", bbox_to_anchor=(1.05, 1), loc='upper left').remove()
        plt.tight_layout()
        plt.grid(True)
        plt.show()


def plot_error_analysis(y_test, y_pred):
    """
    Generates error analysis plots for model evaluation.
    
    Parameters:
        y_test (array-like): Actual values.
        y_pred (array-like): Predicted values.
    """
    errors = y_test - y_pred  # Residuals

    #  1. Scatter Plot (Actual vs. Predicted)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs. Actual')
    plt.plot(y_test, y_test, color='red', linestyle='dashed', label='Perfect Fit')

    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('Comparison of Actual vs. Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()

    #  2. Histogram of Errors
    plt.figure(figsize=(6, 4))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='dashed', label='Zero Error')

    plt.xlabel('Prediction Error (y_test - y_pred)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.grid()
    plt.show()

    # 3. Error distribution using log transformation.
    log_errors = np.sign(errors) * np.log1p(np.abs(errors))  # Log transformation while keeping sign

    plt.figure(figsize=(7, 5))
    sns.histplot(log_errors, bins=30, kde=True, color="blue", edgecolor="black")
    plt.xlabel("Log-Transformed Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Log-Transformed Errors")
    plt.axvline(0, color='red', linestyle='dashed', label="No Error")
    plt.legend()
    plt.grid()
    plt.show()

    # 4. Q-Q Plot (Quantile-Quantile)
    plt.figure(figsize=(6, 6))
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.grid()
    plt.show()

    # 5. Residuals vs. Fitted Values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, errors, alpha=0.5, color='blue')
    plt.axhline(0, color='red', linestyle='dashed')  # Reference line at zero

    plt.xlabel('Fitted Values (Predicted y)')
    plt.ylabel('Residuals (y_test - y_pred)')
    plt.title('Residuals vs. Fitted Values')
    plt.grid()
    plt.show()


# -----------
def _predict_prophet_bundle(df: pd.DataFrame, bundle: dict):
    """Predict Prophet totals for a dataframe using bundle metadata."""
    feature_cols = bundle.get("feature_cols", [])
    group_col = bundle.get("group_col")
    models_by_gln = bundle.get("models_by_gln")

    # Backward compatibility: old bundles had a single "model".
    if models_by_gln is None:
        models_by_gln = {"__global__": bundle["model"]}
        group_col = None

    pred = pd.Series(index=df.index, dtype=float)

    # Per-store Prophet models
    if group_col and group_col in df.columns:
        for gln, gdf in df.groupby(group_col):
            key = _normalize_group_key(gln)
            model = models_by_gln.get(key) or models_by_gln.get("__global__")
            if model is None:
                raise KeyError(f"No Prophet model found for group '{gln}'")

            future = pd.DataFrame({"ds": pd.to_datetime(gdf["date"])})
            for col in feature_cols:
                future[col] = gdf[col].values if col in gdf.columns else 0.0

            pred.loc[gdf.index] = model.predict(future)["yhat"].values
    else:
        model = models_by_gln.get("__global__")
        if model is None:
            # Fallback to first stored model when no explicit global model exists.
            model = next(iter(models_by_gln.values()))

        future = pd.DataFrame({"ds": pd.to_datetime(df["date"])})
        for col in feature_cols:
            future[col] = df[col].values if col in df.columns else 0.0
        pred.loc[df.index] = model.predict(future)["yhat"].values

    return pred.values


def load_model_and_predict(df, model_path):
    """
    Load a trained model bundle and generate predictions.

    Returns:
        np.ndarray of predictions
    """
    bundle = joblib.load(model_path)
    model_type = bundle.get("model_type", "")

    if model_type.startswith("prophet"):
        return _predict_prophet_bundle(df, bundle)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    X = df[feature_cols]
    return model.predict(X)


# -----------
def load_model_and_test(df, model_path):
    """
    Load a saved model and evaluate it on test data.
    Supports XGB / RF / Prophet models.
    """

    bundle = joblib.load(model_path)
    model_type = bundle.get("model_type", "")
    model = bundle.get("model")
    feature_cols = bundle.get("feature_cols", [])

    if model_type.startswith("prophet"):
        y_pred = _predict_prophet_bundle(df, bundle)
        y_test = df["quantity"].values
        metrics = calculate_error_metrics(y_test, y_pred)
        return {
            "model": model,
            "n_models": len(bundle.get("models_by_gln", {}) or {}),
            "group_col": bundle.get("group_col"),
            "feature_cols": feature_cols,
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": metrics,
        }

    if model_type == "autogluon_total":
        from autogluon.timeseries import TimeSeriesDataFrame

        predictor = bundle["predictor"]
        use_features = bundle["use_features"]
        feature_cols = bundle.get("feature_cols", [])

        df_test = df.copy()
        df_test["date"] = pd.to_datetime(df_test["date"])
        df_test["series_id"] = "total_sales"

        keep_cols = ["series_id", "date", "quantity"]
        if use_features:
            keep_cols += feature_cols

        ts_test = TimeSeriesDataFrame.from_data_frame(
            df_test[keep_cols],
            id_column="series_id",
            timestamp_column="date",
        )

        forecasts = predictor.predict(ts_test)
        y_pred = forecasts["mean"].values
        y_test = df_test["quantity"].values
        metrics = calculate_error_metrics(y_test, y_pred)
        return {
            "model": predictor,
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": metrics,
        }

    if model is None:
        raise ValueError(f"Model bundle at {model_path} is missing 'model'.")

    X_test = df[feature_cols]

    if model_type in {"xgb_total", "rf_total", "rf"}:
        y_test = df["quantity"].values
    elif model_type == "xgb_share":
        y_test = df["target"].values
    elif model_type == "xgb_direct":
        y_test = df["quantity"].values
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    y_pred = model.predict(X_test)
    metrics = calculate_error_metrics(y_test, y_pred)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def get_xgb_feature_importance(
        model,
        feature_names,
        importance_type="gain",
        top_k=20,
):
    booster = model.get_booster()
    scores = booster.get_score(importance_type=importance_type)

    imp_df = (
        pd.DataFrame.from_dict(scores, orient="index", columns=["importance"])
        .rename_axis("feature")
        .reset_index()
        .sort_values("importance", ascending=False)
    )

    # Handle missing features (XGB drops unused ones)
    imp_df = imp_df.merge(
        pd.DataFrame({"feature": feature_names}),
        on="feature",
        how="right",
    ).fillna(0)

    return imp_df.sort_values("importance", ascending=False).head(top_k)


def plot_xgb_feature_importance(imp_df, title="XGB Feature Importance"):
    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -------------------
def load_prophet_and_predict(df, model_path):
    bundle = joblib.load(model_path)
    return _predict_prophet_bundle(df, bundle)
