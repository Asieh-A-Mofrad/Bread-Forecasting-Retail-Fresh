# tests/test_leakage.py
from pathlib import Path

import pandas as pd
import pytest

from src.utils.leakage import (
    assert_horizon_known_future_features,
    assert_no_forbidden_features,
    audit_horizon_feature_availability,
)


def test_assert_no_forbidden_features_raises_for_model_specific_column():
    with pytest.raises(RuntimeError):
        assert_no_forbidden_features(["day_of_week_num", "store_total"], model_type="xgb_share")


def test_assert_horizon_known_future_features_allows_history_derived_tokens():
    assert_horizon_known_future_features(["day_of_week_num", "quantity_lag_1", "rolling_mean_7", "gln_te"],
                                         context="unit-test", )


def test_assert_horizon_known_future_features_raises_on_unknown_column():
    with pytest.raises(RuntimeError):
        assert_horizon_known_future_features(["day_of_week_num", "mystery_feature"], context="unit-test")


def test_audit_horizon_feature_availability_statuses_and_csv(tmp_path: Path):
    output_path = tmp_path / "audit.csv"
    report = audit_horizon_feature_availability(
        feature_cols=["y_pred", "on_promotion", "quantity_lag_1", "unknown_col"],
        horizon_days=7,
        output_csv=output_path,
    )
    assert isinstance(report, pd.DataFrame)
    assert output_path.exists()
    status_map = dict(zip(report["feature"], report["status"]))
    assert status_map["y_pred"] == "blocked"
    assert status_map["on_promotion"] == "known_future"
    assert status_map["quantity_lag_1"] == "history_derived"
    assert status_map["unknown_col"] == "unknown"
