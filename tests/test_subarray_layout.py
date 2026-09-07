import pytest

from rag.subarray_layout import get_subarray_layout


def test_get_subarray_layout_returns_positive_baseline_and_station_count():
    layout = get_subarray_layout("Low_full_AA2")
    assert layout["max_baseline_m"] > 0
    assert layout["n_stations"] > 0
    assert layout["n_baselines"] > 0


def test_get_subarray_layout_unknown_template_raises_value_error():
    with pytest.raises(ValueError):
        get_subarray_layout("not_a_real_template")
