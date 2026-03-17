from src.models.prophet_total import _normalize_group_key


def test_normalize_group_key_integer_like():
    assert _normalize_group_key(101) == "101"
    assert _normalize_group_key(101.0) == "101"


def test_normalize_group_key_string():
    assert _normalize_group_key("001") == "001"


def test_normalize_group_key_none():
    assert _normalize_group_key(None) is None
