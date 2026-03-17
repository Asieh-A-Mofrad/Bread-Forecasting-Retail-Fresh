# src/model/hierarchical.py
from src.utils.metrics import calculate_error_metrics
from src.models.evaluate import load_model_and_predict, plot_error_analysis


def evaluate_hierarchical(
        store_test,
        share_test,
        total_model_path,
        share_model_path,
):
    # ----------------------------------
    # Predict totals
    # ----------------------------------
    store_test = store_test.copy()
    store_test["pred_total"] = load_model_and_predict(
        store_test,
        total_model_path
    )

    # ----------------------------------
    # Predict shares
    # ----------------------------------
    share_test = share_test.copy()
    share_test["pred_share"] = load_model_and_predict(
        share_test,
        share_model_path
    )

    # ----------------------------------
    # Merge totals into product-level df
    # ----------------------------------
    df = share_test.merge(
        store_test[["gln", "date", "pred_total"]],
        on=["gln", "date"],
        how="left"
    )

    # ----------------------------------
    # Hierarchical reconstruction
    # ----------------------------------
    df["pred_quantity"] = df["pred_share"] * df["pred_total"]

    # ----------------------------------
    # Metrics (TRUE product quantity!)
    # ----------------------------------
    metrics = calculate_error_metrics(
        df["quantity"],  # <-- from share_test
        df["pred_quantity"]
    )

    plot_error_analysis(
        df["quantity"],
        df["pred_quantity"]
    )

    return metrics, df
