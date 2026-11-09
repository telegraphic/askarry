import pytest
import requests

from rag.sensitivity_calculator import (
    continuum_calculate,
    get_subarrays,
    pss_calculate,
    query_sensitivity_calculator,
    zoom_calculate,
)


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._json_data


def test_query_builds_expected_url_and_params(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({"transformed_result": {}})

    monkeypatch.setattr(requests, "get", fake_get)

    query_sensitivity_calculator("low", "continuum/calculate", {"integration_time_h": 1})

    assert captured["url"] == "https://sensitivity-calculator.skao.int/api/v12/low/continuum/calculate"
    assert captured["params"] == {"integration_time_h": 1}


def test_query_returns_parsed_json(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse({"foo": "bar"}))

    result = query_sensitivity_calculator("mid", "subarrays")

    assert result == {"foo": "bar"}


def test_unknown_telescope_raises():
    with pytest.raises(ValueError):
        query_sensitivity_calculator("west", "subarrays")


def test_unknown_endpoint_raises():
    with pytest.raises(ValueError):
        query_sensitivity_calculator("low", "bogus/calculate")


def test_http_error_includes_response_body(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _FakeResponse({"detail": "bad param"}, status_code=400)
    )

    with pytest.raises(requests.HTTPError, match="bad param"):
        query_sensitivity_calculator("mid", "continuum/calculate", {"bandwidth_hz": -1})


@pytest.mark.parametrize(
    "func, telescope, kwargs, expected_endpoint",
    [
        (get_subarrays, "low", {}, "subarrays"),
        (continuum_calculate, "mid", {"integration_time_h": 1}, "continuum/calculate"),
        (zoom_calculate, "mid", {"freq_centres_hz": [1e9]}, "zoom/calculate"),
        (pss_calculate, "low", {"dm": 14}, "pss/calculate"),
    ],
)
def test_wrapper_dispatches_to_expected_endpoint(monkeypatch, func, telescope, kwargs, expected_endpoint):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({"transformed_result": {}})

    monkeypatch.setattr(requests, "get", fake_get)

    func(telescope, **kwargs)

    assert captured["url"] == f"https://sensitivity-calculator.skao.int/api/v12/{telescope}/{expected_endpoint}"
    assert captured["params"] == kwargs


# --- compare_to_calculator: strict / best-fit modes (recorded responses) ----
# tests/fixtures/senscalc/*.json are real v12 responses, recorded once and
# replayed here; a call that wasn't recorded fails the test (no network).
import json
import math
from pathlib import Path

from rag import sensitivity_calculator as sc

RECORDED = [json.loads(p.read_text()) for p in sorted((Path(__file__).parent / "fixtures/senscalc").glob("*.json"))]
P = "00:00:00 -30:00:00"


def _norm(params: dict) -> dict:
    return {k: str(v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in params.items()}


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setattr(sc, "_SUBARRAYS", {})

    def fake_get(url, params=None, timeout=None):
        tel, endpoint = url.split("/api/v12/")[1].split("/", 1)
        for rec in RECORDED:
            r = rec["request"]
            if (r["telescope"], r["endpoint"]) == (tel, endpoint) and _norm(r["params"]) == _norm(params or {}):
                return _FakeResponse(rec["body"], rec["status"])
        raise AssertionError(f"unrecorded calculator call: {tel} {endpoint} {params}")

    monkeypatch.setattr(requests, "get", fake_get)


def _rec(name):
    return json.loads((Path(__file__).parent / f"fixtures/senscalc/{name}.json").read_text())["body"]


def test_mid_subarray_name_and_label_both_accepted_and_echoed(replay):
    z = dict(rx_band="Band 2", freq_centres_hz=1.4e9, total_bandwidths_hz=3.2e6, spectral_resolutions_hz=13440,
             integration_time_s=3600, pointing_centre=P, weighting_mode="robust", robustness=0)
    for given in ("MID_inner_r2km_aastar", "AA* (MID_inner_r2km)"):
        out = sc.query_sensitivity_calculator("mid", "zoom/calculate", {**z, "subarray_configuration": given})
        assert out["subarray_used"] == {"as_given": given, "name": "MID_inner_r2km_aastar",
                                        "label": "AA* (MID_inner_r2km)", "sent": "AA* (MID_inner_r2km)"}


def test_mixed_array_refusal_flags_array_substituted(replay):
    # B-03: 810 MHz in Band 2 is more than the mixed AA* accepts.
    b03 = dict(rx_band="Band 2", freq_centre_hz=1.355e9, bandwidth_hz=810e6, integration_time_s=3600,
               pointing_centre=P, subarray_configuration="AA*", weighting_mode="uniform")
    out = sc.compare_to_calculator("mid", b03, 15.8, mode="strict")
    sub = out["array_substituted"]
    assert sub["from"] == "AA*" and sub["to"] == "AA* (15-m antennas only)"
    assert "band range" in sub["reason"]
    assert out["closest"]["pct_diff"] == pytest.approx(4.5, abs=0.1)


def test_strict_lists_defaults_applied(replay):
    c02 = dict(rx_band="Band 5b", freq_centre_hz=11.85e9, bandwidth_hz=1e9, integration_time_s=300 * 3600,
               pointing_centre=P, subarray_configuration="AA*")
    out = sc.compare_to_calculator("mid", c02, 1.0, mode="strict")
    assert out["strict"]["defaults_applied"] == {"weighting_mode": "robust", "robustness": 0}
    assert out["n_free_parameters"] == 0


def test_strict_refuses_to_guess_the_band():
    out = sc.compare_to_calculator("mid", {"integration_time_s": 3600}, 1.0, mode="strict")
    assert "rx_band is not stated" in out["strict"]["error"]


def test_best_fit_reports_free_parameters_and_strict_side_by_side(replay):
    # E-03: the stated 150 MHz centre misses by ~77%; 200 MHz matches.
    e03 = dict(freq_centre_mhz=150, bandwidth_mhz=200, integration_time_h=8, pointing_centre=P,
               subarray_configuration="LOW_AA4_all", weighting_mode="robust", robustness=-1)
    out = sc.compare_to_calculator("low", e03, 5, claimed_beam_arcsec=5, sweep={"freq_centre_mhz": [150, 200]})
    assert out["n_free_parameters"] == 1 and out["free_parameters"] == ["freq_centre_mhz"]
    assert out["closest"]["freq_centre_mhz"] == 200 and abs(out["closest"]["pct_diff"]) < 10
    assert out["strict"]["result"]["pct_diff"] > 50 and len(out["variants"]) == 2


def test_kelvin_claim_converted_with_beam(replay):
    # C-02: 0.02 K at 7.2 kHz with an 8.17 arcsec beam at 11.85 GHz.
    c02 = dict(rx_band="Band 5b", freq_centre_hz=11.85e9, bandwidth_hz=1e9, integration_time_s=300 * 3600,
               pointing_centre=P, subarray_configuration="AA*")
    out = sc.compare_to_calculator("mid", c02, 0.02, unit="K", quantity="spectral", claimed_beam_arcsec=8.17,
                                   claimed_channel_width_hz=7.2e3, mode="strict")
    res = out["strict"]["result"]
    # Rayleigh-Jeans: S = 2 k nu^2 T Omega / c^2, Omega = pi theta^2 / (4 ln 2).
    omega = math.pi * (8.17 / 206264.806) ** 2 / (4 * math.log(2))
    expected = 2 * 1.380649e-23 * (11.85e9) ** 2 * 0.02 * omega / (2.99792458e8) ** 2 / 1e-26
    assert res["converted_from_K"]["claimed_jy_per_beam"] == pytest.approx(expected, rel=1e-3)
    assert res["channel_width_scaled"]["to_hz"] == 7.2e3 and res["array_substituted"]["to"].startswith("AA* (15-m")


def test_sigma_level_scales_claim(replay):
    p = dict(rx_band="Band 2", freq_centre_hz=1.31e9, bandwidth_hz=700e6, integration_time_s=3600,
             pointing_centre=P, subarray_configuration="AA*", weighting_mode="robust", robustness=0)
    one = sc.compare_to_calculator("mid", p, 2.0, mode="strict")["closest"]
    five = sc.compare_to_calculator("mid", p, 10.0, mode="strict", sigma_level=5)["closest"]
    assert one["pct_diff"] == five["pct_diff"]


def test_stokes_v_uses_thermal_noise(replay):
    p = dict(rx_band="Band 2", freq_centre_hz=1.31e9, bandwidth_hz=700e6, integration_time_s=3600,
             pointing_centre=P, subarray_configuration="AA*", weighting_mode="robust", robustness=0)
    tr = _rec("mid_b2_robust0")["transformed_result"]
    i = sc.compare_to_calculator("mid", p, 1.0, mode="strict")["closest"]["sensitivity_ujy"]
    v = sc.compare_to_calculator("mid", p, 1.0, mode="strict", stokes="V")["closest"]["sensitivity_ujy"]
    assert i == pytest.approx(tr["total_continuum_sensitivity"]["value"] * 1e6, rel=1e-3)
    assert v == pytest.approx(tr["weighted_continuum_sensitivity"]["value"] * 1e6, rel=1e-3)


def test_half_integer_robust_interpolated_and_labelled(replay):
    p = dict(rx_band="Band 2", freq_centre_hz=1.31e9, bandwidth_hz=700e6, integration_time_s=3600,
             pointing_centre=P, subarray_configuration="MID_AAstar_all", weighting_mode="robust", robustness=0.5)
    row = sc.compare_to_calculator("mid", p, 1.0, mode="strict")["closest"]
    lo, hi = (_rec(f"mid_b2_robust{r}")["transformed_result"]["total_continuum_sensitivity"]["value"] for r in (0, 1))
    assert row["interpolated"] == {"between": [0, 1], "weight_of_upper": 0.5}
    assert row["sensitivity_ujy"] == pytest.approx((lo + hi) / 2 * 1e6, rel=1e-3)
    assert row["subarray_used"]["label"] == "AA*"  # the name was sent as the Mid label
