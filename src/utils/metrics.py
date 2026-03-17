# src/utils/metrics.py
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def calculate_error_metrics(y_test, y_pred, verbose: bool = True):
    """
    Calculate and print various error metrics for regression models.

    Parameters:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: A dictionary containing the calculated error metrics.
    """
    # Calculate MSE, MAE, and R²
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE (Mean Absolute Percentage Error)
    # Handle division by zero by replacing zeros with a small epsilon
    epsilon = np.finfo(float).eps  # Smallest positive float value
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, epsilon))) * 100

    # Calculate RMSLE (Root Mean Squared Logarithmic Error)
    # Ensure non-negative values for log1p
    y_test_non_neg = np.maximum(y_test, 0)
    y_pred_non_neg = np.maximum(y_pred, 0)
    sle = (np.log1p(y_pred_non_neg) - np.log1p(y_test_non_neg)) ** 2
    rmsle = np.sqrt(np.mean(sle))

    # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
    wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test) * 100

    if verbose:
        print(f"Test Mean Squared Error: {mse:.2f}")
        print(f"Test Mean Absolute Error: {mae:.2f}")
        print(f"Test R-Squared: {r2:.2f}")
        print(f"Test Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Test Root Mean Squared Logarithmic Error (RMSLE): {rmsle:.2f}")
        print(f"Test Weighted Mean Absolute Percentage Error (WMAPE): {wmape:.2f}%")

    # Return metrics as a dictionary
    return {
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "MAPE (%)": mape,
        "RMSLE": rmsle,
        "WMAPE (%)": wmape
    }
