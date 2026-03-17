# src/pipeline/hierarchical.py

def evaluate_hierarchical(
        sales_daily,
        total_model_path,
        share_model_path,
        total_features,
        share_features,
        test_start,
        test_end,
        plot=True
):
    # ------------------
    # Build datasets
    # ------------------
    store_df = prepare_total_datasets(sales_daily)
    share_df = prepare_share_dataset(sales_daily)

    # ------------------
    # Time split
    # ------------------
    store_train, store_test = split_by_time(store_df, test_start, test_end)
    share_train, share_test = split_share_dataset(share_df, test_start, test_end)

    # ------------------
    # Load models
    # ------------------
    total_model = joblib.load(total_model_path)
    share_model = joblib.load(share_model_path)

    # ------------------
    # TOTAL prediction
    # ------------------
    store_test["pred_total"] = total_model.predict(store_test[total_features])

    # ------------------
    # SHARE prediction
    # ------------------
    share_test["pred_share"] = share_model.predict(share_test[share_features])

    # Normalize shares per (store, date)
    share_test["pred_share"] = (
        share_test.groupby(["gln", "date"])["pred_share"]
        .transform(lambda x: x / x.sum())
    )

    # ------------------
    # Final allocation
    # ------------------
    share_test = share_test.merge(
        store_test[["gln", "date", "pred_total"]],
        on=["gln", "date"],
        how="left"
    )

    share_test["pred_quantity"] = share_test["pred_share"] * share_test["pred_total"]

    # ------------------
    # Metrics (FINAL)
    # ------------------
    metrics = calculate_error_metrics(share_test["quantity"], share_test["pred_quantity"])

    if plot:
        plot_error_analysis(share_test["quantity"], share_test["pred_quantity"])

    return share_test, metrics
