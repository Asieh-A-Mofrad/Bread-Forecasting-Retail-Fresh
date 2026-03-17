# src/models/registry.py
from src.models.train_xgb import train_model as train_xgb
from src.models.train_rf import train_model as train_rf
from src.models.share import train_model as train_share
from src.models.direct import train_model as train_direct
from src.models.prophet_total import train_model as train_prophet_total

try:
    # Optional dependency: only available in the AutoGluon environment.
    from src.models.autogluon_store import train_autogluon
except Exception:
    train_autogluon = None

MODEL_REGISTRY = {
    "xgb_total": train_xgb,
    "rf_total": train_rf,
    "xgb_share": train_share,
    "xgb_direct": train_direct,
    "prophet_total": train_prophet_total,
}

if train_autogluon is not None:
    MODEL_REGISTRY.update({
        # prediction_length defaults to 7 for total-demand setup.
        "autogluon_total_basic": lambda df, model_path: train_autogluon(
            df,
            model_path=model_path,
            prediction_length=7,
            use_features=False,
        ),
        "autogluon_total_rich": lambda df, model_path: train_autogluon(
            df,
            model_path=model_path,
            prediction_length=7,
            use_features=True,
        ),
    })
