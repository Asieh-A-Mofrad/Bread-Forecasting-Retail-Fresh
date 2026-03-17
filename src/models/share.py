# src/models/share.py
import pandas as pd
import numpy as np

import xgboost as xgb
import optuna

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import joblib

from src.models.utils import select_features


def train_model(
        train_df: pd.DataFrame,
        model_path,
        n_trials: int = 40
):
    if "target" not in train_df.columns:
        raise ValueError("Expected column 'target' not found. Did you prepare the dataset correctly?")

    model_type = "xgb_share"

    # 1. Select features (NO date inside)
    X_train, y_train = select_features(train_df, model_type)

    # 2. Get dates from ORIGINAL dataframe
    dates = pd.to_datetime(train_df["date"]).sort_values().unique()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

        if model_type != "xgb_share":
            raise ValueError(f"Unknown model_type: {model_type}")

        scores = []

        for split_date in np.array_split(dates, 5)[1:]:
            train_idx = train_df["date"] < split_date.min()
            val_idx = train_df["date"].isin(split_date)

            X_t = X_train.loc[train_idx]
            y_t = y_train.loc[train_idx]

            X_v = X_train.loc[val_idx]
            y_v = y_train.loc[val_idx]

            model = xgb.XGBRegressor(
                **params,
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
            )

            assert all(X_train.dtypes.apply(lambda x: x.kind in "ifb")), "Non-numeric columns leaked into X_train"
            model.fit(X_t, y_t)
            preds = model.predict(X_v)

            weights = np.maximum(y_v, 0.05)
            mae = np.average(np.abs(y_v - preds), weights=weights)
            scores.append(mae)

        return np.mean(scores)

    print("🔎 Optuna search for SHARE model")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    final_model = xgb.XGBRegressor(
        **study.best_params,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )

    final_model.fit(X_train, y_train)

    joblib.dump({
        "model": final_model,
        "feature_cols": X_train.columns.tolist(),
        "model_type": "xgb_share"
    }, model_path)

    print(f"Saved SHARE model to {model_path}")

    return {
        "best_params": study.best_params,
        "cv_score": study.best_value
    }
