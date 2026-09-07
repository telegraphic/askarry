"""
Unit tests for rag.subarray_names use captured example data (real
sensitivity-calculator/ska_ost_array_config/schema output observed while
building this module) via monkeypatch, following the mocking pattern in
tests/test_sensitivity_calculator.py — no network or private-index package
needed to run these. A couple of tests hit the real live APIs to guard
against the mapping rules drifting from reality; skip them offline.
"""

import pytest
import requests

import rag.subarray_names as sn

FAKE_LOW_SUBARRAYS = [
    {"name": "LOW_AAstar_all", "label": "AA*", "n_stations": 307},
    {"name": "LOW_inner_r10km_aastar", "label": "AA* (LOW_inner_r10km)", "n_stations": 271},
    {"name": "LOW_AAstar_core_only", "label": "AA* (core only)", "n_stations": 199},
    {"name": "LOW_AA1_all", "label": "AA1", "n_stations": 16},
    {"name": "LOW_AA2_all", "label": "AA2", "n_stations": 68},
]
FAKE_MID_SUBARRAYS = [
    {"name": "MID_AAstar_all", "label": "AA*", "n_ska": 80, "n_meer": 64},
    {"name": "MID_AA4_MeerKAT_only", "label": "AA*/AA4 (13.5-m antennas only)", "n_ska": 0, "n_meer": 64},
]

FAKE_OST_TEMPLATES = frozenset(
    {"LOW_FULL_AASTAR", "LOW_INNER_R10KM_AASTAR", "LOW_FULL_AA2", "MID_FULL_AASTAR"}
)

FAKE_SCHEMA_ROWS = (
    ("sv_aa2", "ska_low", "Low_full_AA2"),
    ("sv_aa2", "ska_low", "Low_core_AA2"),
    ("cycle_0", "ska_low", "Low_full_AAstar"),
    ("cycle_0", "ska_low", "Low_inner_r10km_AAstar"),
)


def _patch_sources(monkeypatch):
    monkeypatch.setattr(
        sn,
        "_sensitivity_subarrays",
        lambda: {"low": FAKE_LOW_SUBARRAYS, "mid": FAKE_MID_SUBARRAYS},
    )
    monkeypatch.setattr(sn, "_ska_ost_templates", lambda: FAKE_OST_TEMPLATES)
    monkeypatch.setattr(sn, "_all_schema_templates", lambda: FAKE_SCHEMA_ROWS)


def test_all_alias_resolves_to_full_template(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("LOW_AAstar_all")

    assert result["canonical_ska_ost_array_config_name"] == "LOW_FULL_AASTAR"
    assert result["sensitivity_calculator_matches"]["low"][0]["name"] == "LOW_AAstar_all"
    assert {"context": "cycle_0", "telescope": "ska_low", "template": "Low_full_AAstar"} in (
        result["setup_validator_matches"]
    )


def test_region_name_matches_directly_case_insensitively(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("low_inner_r10km_AASTAR")

    assert result["canonical_ska_ost_array_config_name"] == "LOW_INNER_R10KM_AASTAR"
    assert result["sensitivity_calculator_matches"]["low"][0]["name"] == "LOW_inner_r10km_aastar"


def test_schema_template_resolves_back_to_sensitivity_calculator_all_alias(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("Low_full_AA2")

    assert result["sensitivity_calculator_matches"]["low"][0]["name"] == "LOW_AA2_all"


def test_calculator_only_alias_has_no_ska_ost_template(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("LOW_AAstar_core_only")

    assert result["canonical_ska_ost_array_config_name"] is None
    assert result["sensitivity_calculator_matches"]["low"][0]["name"] == "LOW_AAstar_core_only"
    assert result["setup_validator_matches"] == []


def test_all_alias_release_not_in_ska_ost_array_config_has_no_canonical_name(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("LOW_AA1_all")

    assert result["canonical_ska_ost_array_config_name"] is None


def test_unrecognised_name_matches_nothing(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("not_a_real_name")

    assert result["canonical_ska_ost_array_config_name"] is None
    assert result["sensitivity_calculator_matches"] == {}
    assert result["setup_validator_matches"] == []


def test_telescope_filter_restricts_matches(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_subarray_name("LOW_AAstar_all", telescope="mid")

    assert result["sensitivity_calculator_matches"] == {}
    assert result["setup_validator_matches"] == []


def test_unknown_telescope_raises():
    with pytest.raises(ValueError):
        sn.resolve_subarray_name("LOW_AAstar_all", telescope="west")


def test_resolve_context_subarrays_chains_schema_to_sensitivity_calculator(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_context_subarrays("sv_aa2", "low")

    assert result[0]["template"] == "Low_full_AA2"
    assert result[0]["sensitivity_calculator_entry"]["name"] == "LOW_AA2_all"
    assert result[1] == {"template": "Low_core_AA2", "sensitivity_calculator_entry": None}


def test_resolve_context_subarrays_accepts_ska_prefixed_telescope(monkeypatch):
    _patch_sources(monkeypatch)

    result = sn.resolve_context_subarrays("sv_aa2", "ska_low")

    assert result[0]["sensitivity_calculator_entry"]["name"] == "LOW_AA2_all"


def test_live_sensitivity_calculator_all_alias_maps_to_ska_ost_array_config():
    """Live end-to-end check: sensitivity-calculator "_all" entries whose
    release exists in ska_ost_array_config resolve to the matching FULL
    template with the same station count. Requires network + the private
    ska_ost_array_config package; not mocked, so it can flake offline."""
    pytest.importorskip("ska_ost_array_config")
    try:
        entries = sn._sensitivity_subarrays()["low"]
    except requests.exceptions.RequestException:
        pytest.skip("sensitivity-calculator API unreachable")

    aa2_all = next(e for e in entries if e["name"] == "LOW_AA2_all")

    from rag.subarray_layout import get_subarray_layout

    result = sn.resolve_subarray_name("LOW_AA2_all", telescope="low")
    canonical = result["canonical_ska_ost_array_config_name"]

    assert canonical == "LOW_FULL_AA2"
    assert get_subarray_layout(canonical)["n_stations"] == aa2_all["n_stations"]
