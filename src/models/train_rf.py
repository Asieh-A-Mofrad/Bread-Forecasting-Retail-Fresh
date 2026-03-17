# src/models/train_rf.py
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import optuna

from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

import joblib
import pickle


# -----------------------
# Model: Random Forest
# -----------------------
def train_model(df, feature_cols, model_path, n_trials=50, verbose=True):
    """
    Trains a Random Forest using expanding-window TimeSeriesSplit CV + Optuna.
    Saves the final model trained on the full training set.
    """

    # Split
    train_data = df.copy()

    if train_data.empty:
        raise ValueError("Train data is empty. Check date ranges.")

    X_train, y_train = train_data[feature_cols], train_data["quantity"]

    # Ensure clean data
    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("NaN values detected in training data.")

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=5, test_size=90)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }

        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

        cv_scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=-1)

        return -cv_scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    if verbose:
        print("\nBest Parameters:", best_params)
        print("Best CV MAE:", study.best_value)

    # Retrain on all training data
    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)

    # Save model
    joblib.dump({
        "model": final_model,
        "feature_cols": feature_cols,
        "model_type": "rf"
    }, model_path)

    print(f"\nSaved model as: {model_path}")

    # Feature importances
    feature_importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "best_params": best_params,
        "cv_score": study.best_value,
        "feature_importances": feature_importances
    }
