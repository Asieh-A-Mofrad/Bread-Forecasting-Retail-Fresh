# src/models/train_xgb.py
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from src.models.utils import select_features


# Split the data
def train_model(
        df,
        model_path,
        n_trials=50,
        verbose=True
):
    """
    Trains an XGBoost model using expanding-window TimeSeriesSplit CV + Optuna.
    Saves the final model trained on the full training set.    
    """
    model_type = "xgb_total"

    X_train, y_train = select_features(df, model_type)

    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("NaN values detected in training data.")

    tscv = TimeSeriesSplit(n_splits=5, test_size=90)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }

        model = xgb.XGBRegressor(**params)

        cv_errors = []

        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_t, y_t, verbose=False)
            preds = model.predict(X_v)

            cv_errors.append(mean_absolute_error(y_v, preds))

        return np.mean(cv_errors)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    if verbose:
        print("\nBest Parameters:", best_params)
        print("Best CV RSME:", study.best_value)

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train, verbose=False)

    joblib.dump({
        "model": final_model,
        "feature_cols": list(X_train.columns),
        "model_type": model_type
    }, model_path)
    print(f"\nSaved model as: {model_path}")

    feature_importances = pd.DataFrame({
        "Feature": list(X_train.columns),
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "best_params": best_params,
        "cv_score": study.best_value,
        "feature_importances": feature_importances
    }
