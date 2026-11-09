"""Tests for the vendored SKA setup-validator core engine and its thin wrapper.

Ported from ska-sci-ops-setup-validator-backend's tests/unit/ska_sci_ops_setup_validator/
test_validators.py and test_frontend_utils.py, adapted to the vendored module
paths under rag/vendor/setup_validator/ and to rag.observing_setup_validator's
wrapper function.
"""

import importlib.util
from pathlib import Path

import pytest
import simplejson

from rag.observing_setup_validator import validate_observing_setup
from rag.vendor.setup_validator.obs_config import create_obs_config
from rag.vendor.setup_validator.unit_utils import to_base_value
from rag.vendor.setup_validator.validators import (
    ValidationError,
    apply_rule,
    apply_rule_expr,
    apply_rule_func,
    apply_rules,
    validate_equals,
    validate_minmax,
    validate_minmax2,
    validate_oneof,
    validate_param,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "setup_validator"

# The subarray-overlap/template rules (rag/vendor/setup_validator/rules/subarrays.py)
# delegate to the SKA-internal ska_ost_array_config package for antenna-layout
# lookups. That package isn't vendored here (heavy, unrelated dependency tree,
# not on public PyPI) - when it's absent those two rules no-op (log + pass)
# instead of crashing every validate() call. See rules/subarrays.py.
HAS_SKA_OST_ARRAY_CONFIG = importlib.util.find_spec("ska_ost_array_config") is not None


def test_validators():
    """Test basic parameter validation functions."""
    validate_param("cont", "1000 MHz", "oneof", {"allowed_values": ["1000 MHz", "2000 MHz"]})
    validate_param("cont", "1000 MHz", "equals", {"ref_value": "1000 MHz"})

    # oneof
    validate_oneof("cont", "1000 MHz", allowed_values=["1000 MHz", "1001 MHz"])
    with pytest.raises(ValidationError):
        validate_oneof("cont", "90 MHz", allowed_values=["1000 MHz", "1001 MHz"])

    # equals
    validate_equals("cbw", "13.44 kHz", ref_value="13.44 kHz")
    with pytest.raises(ValidationError):
        validate_equals("cbw", "13.44 kHz", ref_value="13.5 kHz")

    # minmax
    validate_minmax("cont", "1000 MHz", min="350 MHz", max="1050 MHz")
    with pytest.raises(ValidationError):
        validate_minmax("cont", "1000 MHz", min="350 MHz", max="900 MHz")
    with pytest.raises(ValidationError):
        validate_minmax("cont", "1000 MHz", min="1010 MHz", max="1020 MHz")

    # minmax2
    validate_minmax2("bw", "1000 MHz", min="350 MHz", max="1050 MHz", max2="1001 MHz")
    with pytest.raises(ValidationError):
        validate_minmax2("bw", "1002 MHz", min="350 MHz", max="1050 MHz", max2="1001 MHz")

    # minmax2 - with step
    validate_minmax2("bw", "1002 MHz", min="350 MHz", max="1050 MHz", max2="1020 MHz", step="1 MHz")
    with pytest.raises(ValidationError):
        validate_minmax2("bw", "1002 MHz", min="350 MHz", max="1050 MHz", max2="1020 MHz", step="10 MHz")
    with pytest.raises(ValidationError):
        validate_minmax2("bw", "1002.1 MHz", min="350 MHz", max="1050 MHz", max2="1020 MHz", step="1 MHz")


def test_rules():
    """Test apply_rule_expr / apply_rule_func / apply_rule / apply_rules."""
    params = {"bw": "100 MHz", "cbw": "1 kHz"}
    for k in params:
        params[k] = to_base_value(params[k])
    rule = {"name": "test_basic", "description": "", "expr": "bw / cbw == 100000", "errmsg": "test error"}
    assert apply_rule_expr(rule, params)

    with pytest.raises(ValidationError):
        rule["expr"] = "bw < cbw"
        apply_rule_expr(rule, params)

    apply_rule_expr(rule, params, raise_error=False)  # no raise when flag set

    # apply_rule_func
    params = {"a": "100 MHz", "b": "100 MHz"}
    for k in params:
        params[k] = to_base_value(params[k])
    rule = {"name": "test_basic", "description": "", "func": "numpy.isclose", "errmsg": "test error"}
    assert apply_rule_func(rule, params)

    # apply_rule_func with template substitution
    def rulefunc(a, b):
        return (False, {"var1": "template substitution"})

    rule = {"name": "test_template", "description": "", "func": rulefunc, "errmsg": "test $var1"}
    result = apply_rule_func(rule, {"a": 0, "b": 0}, raise_error=False)
    assert result == "test template substitution"
    assert apply_rule(rule, {"a": 0, "b": 0}, raise_error=False) == "test template substitution"

    with pytest.raises(ValidationError):
        apply_rule_func(rule, {"a": "100 MHz", "b": "100.1 MHz"})

    # apply_rules
    params = {"a": "100 MHz", "b": "0.1 GHz"}
    rules = {
        "rule1": {"name": "rule1", "description": "", "expr": "a  == b", "errmsg": "rule1 error"},
        "rule2": {"name": "rule2", "description": "", "func": "numpy.isclose", "errmsg": "rule2 error"},
    }
    apply_rules(rules, params)

    params = {"a": "100 MHz", "b": "100.1 MHz"}
    rules = {
        "rule1": {"name": "rule1", "description": "", "expr": "a  > b", "errmsg": "rule1 error"},
        "rule2": {"name": "rule2", "description": "", "func": "numpy.isclose", "errmsg": "rule2 error"},
    }
    with pytest.raises(ValidationError):
        apply_rules(rules, params, raise_on_first=False, raise_error=True)
    with pytest.raises(ValidationError):
        apply_rules(rules, params, raise_on_first=True, raise_error=True)

    apply_rules(rules, params, raise_error=False)
    apply_rules(rules, params, raise_error=False, raise_on_first=True)


def test_validate_obs_config_with_no_subarrays_fails():
    """An ObsConfig with no subarrays defined must fail validation."""
    obs = create_obs_config(telescope="SKA-Mid", context="SV-AA*")
    with pytest.raises(ValidationError):
        obs.validate()


def _load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / name) as fh:
        return simplejson.load(fh)


def test_validate_observing_setup_success():
    """A valid frontend-schema config validates successfully end-to-end."""
    obs_config = _load_fixture("frontend_pass.json")

    response = validate_observing_setup(obs_config)

    assert isinstance(response, dict)
    assert response["status"] == "success"


def test_validate_observing_setup_bad_context():
    """An invalid context value is rejected before any rules run."""
    obs_config = _load_fixture("frontend_pass.json")
    obs_config["context"] = ""

    response = validate_observing_setup(obs_config)

    assert response["status"] == "error"
    assert response["alerts"][0].startswith("Selector parameter 'ObsConfig.context' does not accept ''")


@pytest.mark.skipif(
    not HAS_SKA_OST_ARRAY_CONFIG,
    reason="subarray-overlap check requires the (unvendored, SKA-internal) ska_ost_array_config package",
)
def test_validate_observing_setup_overlapping_subarrays_fails():
    """Two subarrays with the same layout are rejected by the overlap rule."""
    obs_config = _load_fixture("frontend_pass.json")
    obs_config["context"] = "SV-AA*"
    obs_config["subarrays"]["2"] = obs_config["subarrays"]["1"]

    response = validate_observing_setup(obs_config)

    assert response["status"] == "error"
    assert response["alerts"][0] == "Subarrays overlap. Please check the subarray configuration."


# --- Per-context, per-dimension verdicts for a requirement record -----------
from rag.observing_setup_validator import DEFAULT_CONTEXTS, requirement_verdicts


def test_default_contexts_include_aa4():
    assert DEFAULT_CONTEXTS == ["SV-AA2", "SV-AA*", "Cycle 0", "Cycle 1", "AA4"]


def test_vlbi_is_mode_not_offered_not_aa4_only():
    out = requirement_verdicts({"telescope": {"value": "mid"}, "mode": {"value": "VLBI"}})
    assert out["verdict"] == "mode_not_offered"
    assert {c["verdict"] for c in out["by_context"].values()} == {"mode_not_offered"}


@pytest.mark.skipif(not HAS_SKA_OST_ARRAY_CONFIG, reason="ska_ost_array_config not installed")
def test_band5b_0p1arcsec_fails_on_resolution_not_just_passes_on_band():
    req = {"telescope": {"value": "mid"}, "band": {"value": "Band 5b"},
           "sensitivity": {"value": 1, "unit": "uJy/beam"}, "angular_resolution": {"value": 0.1, "unit": "arcsec"},
           "hours": {"value": 10, "unit": "h"}}
    out = requirement_verdicts(req, check_sensitivity=False)
    aastar = out["by_context"]["SV-AA*"]["dimensions"]
    assert aastar["band"]["verdict"] == "pass"
    assert aastar["angular_resolution"]["verdict"] == "fail"  # lambda/B_max = 0.145" on 36 km baselines
    assert out["by_context"]["AA4"]["dimensions"]["angular_resolution"]["verdict"] == "pass"
    assert out["verdict"] == "aa4_only" and out["failing_dimensions"] == ["angular_resolution"]


def test_zoom_unavailable_at_aa2_available_from_aastar():
    req = {"telescope": {"value": "mid"}, "mode": {"value": "zoom"}, "band": {"value": "Band 2"},
           "channel_width": {"value": 0.1, "unit": "km/s"}}
    out = requirement_verdicts(req, check_sensitivity=False)
    assert out["by_context"]["SV-AA2"]["dimensions"]["mode"]["verdict"] == "fail"
    assert out["verdict"] == "available_from SV-AA*"


def test_obs_config_path_reports_aa4_as_not_validated(monkeypatch):
    import rag.observing_setup_validator as osv
    monkeypatch.setattr(osv, "validate_observing_setup", lambda cfg: {"status": "success", "context": cfg["context"]})
    out = osv.validate_across_contexts({"telescopeType": "SKA-Mid", "subarrays": {}})
    assert list(out) == DEFAULT_CONTEXTS and out["AA4"]["status"] == "not_validated"
