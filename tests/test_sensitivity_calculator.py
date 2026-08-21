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

    assert captured["url"] == "https://sensitivity-calculator.skao.int/api/v11/low/continuum/calculate"
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

    assert captured["url"] == f"https://sensitivity-calculator.skao.int/api/v11/{telescope}/{expected_endpoint}"
    assert captured["params"] == kwargs
