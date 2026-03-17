# src/features/encoding.py

def fit_gln_target_encoding(store_train, group_col=None, target_col=None):
    """
    Fit target encoding for store identifier.

    Defaults:
    - group_col: auto-detect from ["gln", "store_id", "item_id"]
    - target_col: "quantity" if present, else "target"
    """
    if group_col is None:
        for cand in ("gln", "store_id", "item_id"):
            if cand in store_train.columns:
                group_col = cand
                break
    if group_col is None:
        raise KeyError("No group column found for target encoding. Expected one of: gln, store_id, item_id.")

    if target_col is None:
        if "quantity" in store_train.columns:
            target_col = "quantity"
        elif "target" in store_train.columns:
            target_col = "target"
        else:
            raise KeyError("No target column found for target encoding. Expected 'quantity' or 'target'.")

    mapping = store_train.groupby(group_col)[target_col].mean()
    global_mean = store_train[target_col].mean()
    return mapping, global_mean


def apply_gln_te(df, mapping, global_mean, group_col=None):
    if group_col is None:
        for cand in ("gln", "store_id", "item_id"):
            if cand in df.columns:
                group_col = cand
                break
    if group_col is None:
        raise KeyError("No group column found for applying target encoding.  Expected one of: gln, store_id, item_id.")

    df["gln_te"] = df[group_col].map(mapping).fillna(global_mean)
    return df


def fit_gb_id_target_encoding(train_df, target_col="target"):
    mapping = train_df.groupby("gb_id")[target_col].mean()
    global_mean = train_df[target_col].mean()
    return mapping, global_mean


def apply_gb_id_target_encoding(df, mapping, global_mean):
    df = df.copy()
    df["gb_id_mean_target"] = (df["gb_id"].map(mapping).fillna(global_mean))
    return df
