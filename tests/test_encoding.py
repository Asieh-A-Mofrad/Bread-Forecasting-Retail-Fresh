# tests/test_encoding.py
import pandas as pd
import pytest

from src.features.encoding import (
    apply_gb_id_target_encoding,
    apply_gln_te,
    fit_gb_id_target_encoding,
    fit_gln_target_encoding,
)


def test_fit_gln_target_encoding_uses_detected_columns():
    df = pd.DataFrame(
        {
            "gln": [1, 1, 2],
            "quantity": [10.0, 20.0, 30.0],
        }
    )
    mapping, global_mean = fit_gln_target_encoding(df)
    assert mapping.loc[1] == 15.0
    assert mapping.loc[2] == 30.0
    assert global_mean == 20.0


def test_fit_gln_target_encoding_raises_when_group_col_missing():
    df = pd.DataFrame({"quantity": [1.0, 2.0]})
    with pytest.raises(KeyError):
        fit_gln_target_encoding(df)


def test_apply_gln_te_fills_unseen_groups_with_global_mean():
    train = pd.DataFrame({"gln": [1, 1, 2], "quantity": [10.0, 20.0, 30.0]})
    mapping, global_mean = fit_gln_target_encoding(train)

    test = pd.DataFrame({"gln": [1, 3]})
    out = apply_gln_te(test, mapping, global_mean)
    assert out["gln_te"].tolist() == [15.0, 20.0]


def test_gb_id_target_encoding_roundtrip():
    train = pd.DataFrame({"gb_id": [100, 100, 200], "target": [2.0, 4.0, 8.0]})
    mapping, global_mean = fit_gb_id_target_encoding(train)
    out = apply_gb_id_target_encoding(
        pd.DataFrame({"gb_id": [100, 999]}),
        mapping,
        global_mean,
    )
    assert out["gb_id_mean_target"].tolist() == [3.0, pytest.approx(14.0 / 3.0)]
