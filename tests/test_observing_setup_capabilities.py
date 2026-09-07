import pytest

from rag.observing_setup_capabilities import (
    describe_rules,
    describe_schema,
    get_context_defaults,
    list_capability_schemas,
)


def test_list_capability_schemas_includes_subarray_config():
    schemas = list_capability_schemas("SKA-Low")
    assert "subarray_config" in schemas
    assert "odps/calibrated_visibilities" in schemas


def test_list_capability_schemas_unknown_telescope_raises():
    with pytest.raises(ValueError):
        list_capability_schemas("ska_huge")


def test_describe_schema_returns_allowed_values_for_aa2_low_subarrays():
    described = describe_schema("subarray_config", context="SV-AA2", telescope="SKA-Low")
    assert described["template"]["type"] == "oneof"
    assert described["template"]["allowed_values"] == [
        "Low_full_AA2", "Low_core_AA2", "Low_outer_AA2",
    ]


def test_describe_schema_returns_minmax_bounds():
    described = describe_schema("subarray_config", context="base", telescope="ska_mid")
    n_beam = described["n_beam"]
    assert n_beam["type"] == "minmax"
    assert "min" in n_beam and "max" in n_beam


def test_describe_schema_returns_equals_fixed_value():
    described = describe_schema("continuum_settings", context="base", telescope="ska_mid")
    assert described["channelWidth"]["type"] == "equals"
    assert described["channelWidth"]["ref_value"] == "13.4 kHz"


def test_describe_rules_includes_pst_beam_limit_rule():
    rules = describe_rules("obs_config")
    assert "check_sum_pst_beams" in rules
    assert rules["check_sum_pst_beams"]["func"].endswith("check_sum_pst_beams_is_valid")


def test_get_context_defaults_returns_per_band_and_low_defaults():
    defaults = get_context_defaults("SV-AA2")
    assert "Band 1" in defaults["skaMidDefaults"]
    assert defaults["skaMidDefaults"]["Band 1"]["maxContinuumBandwidth"] > 0
    assert defaults["skaLowDefaults"]["maxContinuumBandwidth"] > 0
