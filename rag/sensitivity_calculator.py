"""
Thin client for the SKAO sensitivity calculator REST API
(https://sensitivity-calculator.skao.int/api/v12/...), as documented in
pdfs/SKA_Key_Capabilities/Interacting with the SKAO sensitivity calculator.pdf.

Endpoints (per telescope, "low" or "mid"):
    subarrays           GET, no params — lists available subarray configs.
    continuum/calculate  GET, query params — continuum-mode sensitivity/integration-time.
    zoom/calculate       GET, query params — zoom/spectral-line-mode sensitivity.
    pss/calculate        GET, query params — pulsar search/timing sensitivity.
"""

from __future__ import annotations

import itertools
import math

import requests

# v11 was retired upstream in 2026-11 (404); v12 keeps the same query parameters and response shape.
BASE_URL = "https://sensitivity-calculator.skao.int/api/v12"
TELESCOPES = {"low", "mid"}
ENDPOINTS = {"subarrays", "continuum/calculate", "zoom/calculate", "pss/calculate"}
TIMEOUT_S = 30


def query_sensitivity_calculator(telescope: str, endpoint: str, params: dict | None = None) -> dict:
    """Query one SKAO sensitivity calculator REST endpoint and return its parsed JSON.

    Args:
        telescope: "low" or "mid".
        endpoint:  One of "subarrays", "continuum/calculate", "zoom/calculate", "pss/calculate".
        params:    Query parameters for the endpoint (see the Low/Mid REST API
                   query parameter references). None/omitted for "subarrays".

    Raises:
        ValueError: telescope or endpoint is not recognised.
        requests.HTTPError: the API returned a non-2xx status (400 = invalid
            params, 5xx = calculation error) — the response body (JSON error
            detail where available) is included in the exception message.
    """
    if telescope not in TELESCOPES:
        raise ValueError(f"Unknown telescope {telescope!r}; expected one of {sorted(TELESCOPES)}")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {sorted(ENDPOINTS)}")

    params = dict(params or {})
    used = None
    if endpoint != "subarrays" and params.get("subarray_configuration"):
        params["subarray_configuration"], used = _api_subarray(telescope, params["subarray_configuration"])

    url = f"{BASE_URL}/{telescope}/{endpoint}"
    response = requests.get(url, params=params, timeout=TIMEOUT_S)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise requests.HTTPError(f"{exc} — response body: {detail}", response=response) from exc
    result = response.json()
    if used and isinstance(result, dict):
        result["subarray_used"] = used
    return result


# Subarray lists per telescope, fetched once per process (only successful
# list responses are kept, so a failed call is retried next time).
_SUBARRAYS: dict[str, list[dict]] = {}


def subarray_entries(telescope: str) -> list[dict]:
    """The calculator's subarrays for *telescope*: [{"name", "label", ...}]."""
    if telescope not in _SUBARRAYS:
        try:
            got = query_sensitivity_calculator(telescope, "subarrays")
        except (requests.RequestException, ValueError):
            return []
        if not isinstance(got, list):
            return []
        _SUBARRAYS[telescope] = got
    return _SUBARRAYS[telescope]


def _find_subarray(telescope: str, value: str) -> dict | None:
    want = str(value).strip().lower()
    return next((e for e in subarray_entries(telescope)
                 if want in (str(e.get("name", "")).lower(), str(e.get("label", "")).lower())), None)


def _api_subarray(telescope: str, value: str) -> tuple[str, dict | None]:
    """The form of a subarray the API accepts, and what was used.

    The calculator lists every subarray with a name ("MID_inner_r2km_aastar")
    and a label ("AA* (MID_inner_r2km)"), but the Mid API accepts only labels
    and the Low API only names. Either is accepted here and translated; the
    returned dict echoes the name, label and the form the caller gave.
    """
    entry = _find_subarray(telescope, value)
    if entry is None:
        return value, None
    api = entry["label"] if telescope == "mid" else entry["name"]
    return api, {"as_given": value, "name": entry.get("name"), "label": entry.get("label"), "sent": api}


MID_EFFICIENCY_PARAMS = """
    Advanced overrides (all optional, all telescopes/rx_bands): taper (arcsec, Mid
    only), pwv (mm), el (deg, elevation of target — distinct from Low's
    elevation_limit), eta_system/eta_pointing/eta_coherence/eta_digitisation/
    eta_correlation/eta_bandpass/eta_ska/eta_meer (efficiencies, 0-1),
    t_sys_ska/t_rx_ska/t_spl_ska/t_sky_ska/t_gal_ska (K, for SKA-Mid 15m dishes),
    t_sys_meer/t_rx_meer/t_spl_meer/t_sky_meer/t_gal_meer (K, for MeerKAT 13.5m
    dishes), alpha (Galactic emission spectral index). Leave unset to use the
    calculator's own defaults."""


def get_subarrays(telescope: str) -> dict:
    """List available subarray configs for a telescope ("low" or "mid")."""
    return query_sensitivity_calculator(telescope, "subarrays")


def continuum_calculate(telescope: str, **params) -> dict:
    """Continuum-mode sensitivity/integration-time calculation.

    Common params (required unless noted):
        pointing_centre            "HH:MM:SS[.ss] [+|-]DD:MM:SS[.ss]"
        weighting_mode             "uniform", "natural", or "robust"
        robustness                 robust parameter; only used if weighting_mode="robust"
        subarray_configuration     e.g. "LOW_AA4_all" / "AA4" (from get_subarrays);
                                    alternative to num_stations (low) or
                                    n_ska + n_meer (mid, custom subarray)
        spectral_averaging_factor  optional; averages spectral resolution for the
                                    effective continuum bandwidth
        n_subbands                 optional; number of equal sub-bands to also report
        subband_freq_centres_mhz/_hz  optional; explicit per-sub-band centre frequencies

    Low-specific:
        freq_centre_mhz, bandwidth_mhz   define the spectral window (MHz)
        integration_time_h               hours; the only supported calc direction for Low
        elevation_limit                  optional; lowest elevation observed at (deg)
        num_stations                     optional alternative to subarray_configuration

    Mid-specific (required unless noted):
        rx_band                          receiver band, e.g. "Band 1" (5a/5b unavailable)
        freq_centre_hz, bandwidth_hz     define the spectral window (Hz)
        integration_time_s               seconds; OR supplied_sensitivity (+ sensitivity_unit,
                                          "Jy/beam" or "K") to instead solve for integration time
        subband_supplied_sensitivities   optional; per-sub-band sensitivities when solving
                                          for integration time per sub-band
    """ + MID_EFFICIENCY_PARAMS
    return query_sensitivity_calculator(telescope, "continuum/calculate", params)


def zoom_calculate(telescope: str, **params) -> dict:
    """Zoom/spectral-line-mode sensitivity calculation, for one or more zoom windows
    (pass one value per window in each array param).

    Common params (required unless noted):
        pointing_centre, weighting_mode, robustness, subarray_configuration
        (same meaning as in continuum_calculate)
        spectral_resolutions_hz     array; channel width per zoom window (Hz)
        spectral_averaging_factor   optional

    Low-specific:
        freq_centres_mhz            array; centre frequency per window (MHz)
        total_bandwidths_khz        array; total bandwidth per window (kHz)
        integration_time_h          hours
        elevation_limit              optional (deg)
        num_stations                 optional alternative to subarray_configuration

    Mid-specific (required unless noted; zoom only available for AA*/AA4/custom
    subarrays, the latter via n_ska + n_meer):
        rx_band                      receiver band, e.g. "Band 1"
        freq_centres_hz              array; centre frequency per window (Hz)
        total_bandwidths_hz          array; total bandwidth per window (Hz)
        integration_time_s           seconds
        supplied_sensitivities       optional array; per-window sensitivities for
                                      additional line-observation outputs
                                      (+ sensitivity_unit, "Jy/beam" or "K")
    """ + MID_EFFICIENCY_PARAMS
    return query_sensitivity_calculator(telescope, "zoom/calculate", params)


def pss_calculate(telescope: str, **params) -> dict:
    """Pulsar search/timing (PSS) sensitivity calculation. Only available for
    subarrays with max baseline < 20 km (check get_subarrays).

    Common params (required unless noted):
        pointing_centre       required for Mid, optional for Low
        pulsar_mode           "folded_pulse" or "single_pulse"
        dm                    optional; dispersion measure (pc/cm^3)
        intrinsic_pulse_width optional; ms — required if pulsar_mode="folded_pulse"
        pulse_period          optional; ms — ignored if pulsar_mode="single_pulse"

    Low-specific:
        subarray_configuration/num_stations   as in continuum_calculate
        freq_centre_mhz, bandwidth_mhz         spectral window (MHz); bandwidth_mhz
                                                ignored if pulsar_mode="folded_pulse"
        integration_time_h                     hours
        elevation_limit                        optional (deg)

    Mid-specific (required unless noted):
        rx_band, subarray_configuration
        freq_centre_hz                         Hz
        bandwidth_hz                           optional; Hz
        integration_time_s                     optional; seconds
    """ + MID_EFFICIENCY_PARAMS
    return query_sensitivity_calculator(telescope, "pss/calculate", params)


# Weighting schemes tried by compare_to_calculator when a paper doesn't say
# which it used (the most common reason a quoted figure fails to reproduce).
WEIGHTING_VARIANTS: list[dict] = [
    {"weighting_mode": "natural"},
    *({"weighting_mode": "robust", "robustness": r} for r in (-2, -1, 0, 1, 2)),
    {"weighting_mode": "uniform"},
]
# Claimed-figure units → Jy. "/beam" is ignored: continuum/zoom figures are
# per beam, pulsar-search figures are plain flux densities.
_JY_PER = {"jy": 1.0, "mjy": 1e-3, "ujy": 1e-6, "µjy": 1e-6, "μjy": 1e-6, "microjy": 1e-6, "njy": 1e-9}
_KELVIN = {"k", "mk", "uk", "µk", "μk"}
_K_PER = {"k": 1.0, "mk": 1e-3, "uk": 1e-6, "µk": 1e-6, "μk": 1e-6}

# What strict mode assumes when a parameter isn't stated. Echoed back as
# "defaults_applied" so a reader can see every assumption behind a match.
# Band centre and continuum bandwidth come from the setup-validator frontend
# defaults for STRICT_CONTEXT; the pointing only sets the elevation.
STRICT_CONTEXT = "sv_aastar"
STRICT_DEFAULTS = {
    "weighting": {"weighting_mode": "robust", "robustness": 0},
    "subarray": {"mid": "AA*", "low": "LOW_AAstar_all"},
    "pointing_centre": "00:00:00 -30:00:00",
}
# Ways the calculator says an array can't do a band or window. When one
# names a subarray that includes the 13.5-m MeerKAT dishes, the 15-m-only
# subarray at the same stage is tried instead, and the swap is reported.
_ARRAY_REFUSALS = ("Subarray configuration not allowed", "does not lie within the band range")


def _to_jy(value: float, unit: str) -> float:
    return value * _JY_PER[unit.lower().replace(" ", "").replace("/beam", "")]


def _is_kelvin(unit: str) -> bool:
    return unit.lower().replace(" ", "") in _KELVIN


def _kelvin_to_jy_per_beam(t_k: float, freq_hz: float, beam_arcsec: list[float]) -> float:
    """Rayleigh-Jeans brightness temperature to flux density per Gaussian beam."""
    from astropy import units as u

    maj, mnr = beam_arcsec
    beam_area = (2 * math.pi / (8 * math.log(2))) * (maj * u.arcsec) * (mnr * u.arcsec)
    return (t_k * u.K).to(u.Jy, equivalencies=u.brightness_temperature(freq_hz * u.Hz, beam_area)).value


def _read_result(response: dict, endpoint: str, quantity: str, window: int,
                 stokes: str = "I") -> tuple[float, list | None]:
    """(sensitivity in Jy, [beam_maj, beam_min] arcsec or None) from one response.

    Stokes I uses the total noise (thermal plus confusion) when the
    calculator gives it; Stokes V uses the thermal ("weighted") noise only,
    since circular polarisation has no confusion floor.
    """
    if endpoint == "pss":
        key = next(k for k in response if k.endswith("_sensitivity"))
        return _to_jy(response[key]["value"], response[key]["unit"]), None
    if endpoint == "zoom":
        result = response["transformed_result"][window]
        beam_deg = response["weighting"]["spectral_weighting"][window]["beam_size"]
        quantity = "spectral"
    else:
        result = response["transformed_result"]
        beam_deg = response["weighting"][f"{quantity}_weighting"]["beam_size"]
    # "total_*" (incl. confusion noise) is null for natural weighting.
    thermal = result[f"weighted_{quantity}_sensitivity"]
    got = thermal if stokes.upper() == "V" else (result[f"total_{quantity}_sensitivity"] or thermal)
    value = got["value"]
    if "unit" in got:  # e.g. "Jy / beam", "uJy / beam"
        value = _to_jy(value, got["unit"])
    # weighting.*.beam_size is always populated, in degrees.
    beam = [round(beam_deg["beam_maj_scaled"] * 3600, 3), round(beam_deg["beam_min_scaled"] * 3600, 3)]
    return value, beam


def _channel_width_hz(telescope: str, endpoint: str, params: dict, window: int) -> float | None:
    """Channel width the calculator's spectral figure refers to."""
    if endpoint == "zoom":
        res = params.get("spectral_resolutions_hz")
        return float(res[window] if isinstance(res, (list, tuple)) else res) if res else None
    from rag.observing_setup_capabilities import get_context_defaults

    d = get_context_defaults(STRICT_CONTEXT)
    khz = d["skaMidContChannelWidth"] if telescope == "mid" else d["skaLowContChannelWidth"]
    return float(khz) * 1e3 * float(params.get("spectral_averaging_factor", 1))


def _freq_hz(telescope: str, params: dict, window: int = 0) -> float | None:
    for key, scale in (("freq_centre_hz", 1), ("freq_centre_mhz", 1e6), ("freq_centres_hz", 1), ("freq_centres_mhz", 1e6)):
        if params.get(key) is not None:
            v = params[key]
            return float(v[window] if isinstance(v, (list, tuple)) else v) * scale
    return None


def _strict_params(telescope: str, endpoint: str, params: dict) -> tuple[dict, dict]:
    """The stated params plus declared defaults for what isn't stated.
    Returns (params to send, {param: default applied})."""
    from rag.observing_setup_capabilities import get_context_defaults

    out, applied = dict(params), {}
    if endpoint != "pss" and "weighting_mode" not in out:
        out.update(STRICT_DEFAULTS["weighting"])
        applied.update(STRICT_DEFAULTS["weighting"])
    if not out.get("subarray_configuration") and "num_stations" not in out and "n_ska" not in out:
        out["subarray_configuration"] = applied["subarray_configuration"] = STRICT_DEFAULTS["subarray"][telescope]
    if not out.get("pointing_centre"):
        out["pointing_centre"] = applied["pointing_centre"] = STRICT_DEFAULTS["pointing_centre"]
    d = get_context_defaults(STRICT_CONTEXT)
    if telescope == "mid":
        band = out.get("rx_band")
        if not band:
            raise ValueError("rx_band is not stated; strict mode does not guess a band")
        bd = d["skaMidDefaults"][band]
        centre, bw, f_key, bw_key, f_scale = bd["centralFrequency"], bd["continuumBandwidth"], "freq_centre_hz", "bandwidth_hz", 1e6
    else:
        ld = d["skaLowDefaults"]
        centre, bw, f_key, bw_key, f_scale = ld["centralFrequency"], ld["continuumBandwidth"], "freq_centre_mhz", "bandwidth_mhz", 1
    if endpoint == "continuum" and _freq_hz(telescope, out) is None:
        out[f_key] = applied[f_key] = float(centre) * f_scale
    if endpoint == "continuum" and bw_key not in out:
        out[bw_key] = applied[bw_key] = float(bw) * f_scale
    return out, applied


def _fifteen_m_equivalent(telescope: str, subarray: str) -> dict | None:
    """The 15-m-only subarray at the same stage as a mixed one, if any."""
    entry = _find_subarray(telescope, subarray)
    if telescope != "mid" or not entry or not entry.get("n_meer"):
        return None
    stage = str(entry.get("label", "")).split(" ")[0]  # "AA*", "AA4"
    return next((e for e in subarray_entries(telescope)
                 if not e.get("n_meer") and str(e.get("label", "")).startswith(f"{stage} (15-m")), None)


def _call(telescope: str, endpoint: str, params: dict) -> tuple[dict, dict | None]:
    """One calculation; on an array refusal, retry on the 15-m-only subarray.
    Returns (response, array_substituted or None). Raises HTTPError otherwise."""
    try:
        return query_sensitivity_calculator(telescope, f"{endpoint}/calculate", params), None
    except requests.HTTPError as exc:
        detail = str(exc)
        alt = _fifteen_m_equivalent(telescope, params.get("subarray_configuration", "")) \
            if any(r in detail for r in _ARRAY_REFUSALS) else None
        if alt is None:
            raise
        response = query_sensitivity_calculator(telescope, f"{endpoint}/calculate",
                                                {**params, "subarray_configuration": alt["label"]})
        reason = detail[detail.find("'detail'"):][:200] if "'detail'" in detail else detail[-200:]
        return response, {"from": params["subarray_configuration"], "to": alt["label"],
                          "reason": f"the calculator refused the requested array ({reason}); the 13.5-m "
                                    f"MeerKAT dishes don't cover this band or window, so the 15-m-only "
                                    f"subarray was used"}


def _evaluate(telescope: str, endpoint: str, params: dict, quantity: str, window: int,
              stokes: str) -> dict:
    """Sensitivity (Jy) and beam for one parameter set. A non-integer
    robustness is interpolated linearly between the neighbouring integers,
    which are all the calculator accepts."""
    r = params.get("robustness")
    if params.get("weighting_mode") == "robust" and r is not None and float(r) != int(float(r)):
        lo, hi = math.floor(float(r)), math.ceil(float(r))
        a = _evaluate(telescope, endpoint, {**params, "robustness": lo}, quantity, window, stokes)
        b = _evaluate(telescope, endpoint, {**params, "robustness": hi}, quantity, window, stokes)
        w = float(r) - lo
        beam = [round(x + w * (y - x), 3) for x, y in zip(a["beam"], b["beam"])] if a["beam"] and b["beam"] else None
        return {"jy": a["jy"] + w * (b["jy"] - a["jy"]), "beam": beam, "subarray_used": a["subarray_used"],
                "array_substituted": a["array_substituted"] or b["array_substituted"],
                "interpolated": {"between": [lo, hi], "weight_of_upper": round(w, 3)}}
    response, substituted = _call(telescope, endpoint, params)
    jy, beam = _read_result(response, endpoint, quantity, window, stokes)
    return {"jy": jy, "beam": beam, "subarray_used": response.get("subarray_used"),
            "array_substituted": substituted, "interpolated": None}


def _row(variant: dict, got: dict, claimed: dict, telescope: str, endpoint: str, params: dict,
         window: int, claimed_beam_arcsec, claimed_channel_width_hz) -> dict:
    """One compared variant: sensitivity, % difference, beam, and every
    conversion applied to the claim or the calculator's figure."""
    got_jy, notes = got["jy"], {}
    if claimed_channel_width_hz:
        calc_cw = _channel_width_hz(telescope, endpoint, params, window)
        if calc_cw:
            # Radiometer equation: noise scales as 1/sqrt(channel width).
            got_jy *= math.sqrt(calc_cw / claimed_channel_width_hz)
            notes["channel_width_scaled"] = {"from_hz": calc_cw, "to_hz": claimed_channel_width_hz}
    claimed_jy = claimed["jy"]
    if claimed["kelvin"] is not None:
        beam = claimed_beam_arcsec or got["beam"]
        beam = list(beam) if isinstance(beam, (list, tuple)) else [beam, beam]
        freq = _freq_hz(telescope, params, window)
        if not beam or not freq:
            raise ValueError("A claim in kelvin needs a frequency and a beam (claimed_beam_arcsec or the calculator's)")
        claimed_jy = _kelvin_to_jy_per_beam(claimed["kelvin"], freq, beam)
        notes["converted_from_K"] = {"beam_arcsec": beam, "freq_hz": freq, "claimed_jy_per_beam": claimed_jy}
    row = {**variant, "sensitivity_ujy": round(got_jy * 1e6, 4),
           "pct_diff": round(100 * (got_jy - claimed_jy) / claimed_jy, 1), **notes}
    if got["beam"]:
        row["beam_arcsec"] = got["beam"]
        if claimed_beam_arcsec:
            cb = sum(claimed_beam_arcsec) / 2 if isinstance(claimed_beam_arcsec, (list, tuple)) else claimed_beam_arcsec
            row["beam_pct_diff"] = round(100 * (sum(got["beam"]) / 2 - cb) / cb, 1)
    for key in ("interpolated", "array_substituted", "subarray_used"):
        if got.get(key):
            row[key] = got[key]
    return row


def compare_to_calculator(
    telescope: str,
    params: dict,
    claimed_sensitivity: float,
    unit: str = "uJy/beam",
    quantity: str = "continuum",
    claimed_beam_arcsec: float | list[float] | None = None,
    endpoint: str = "continuum",
    window_index: int = 0,
    mode: str = "best_fit",
    sweep: dict[str, list] | None = None,
    sigma_level: float = 1.0,
    stokes: str = "I",
    claimed_channel_width_hz: float | None = None,
) -> dict:
    """Re-run a paper's stated calculator parameters and report how far the
    result lands from the claimed figure.

    *mode* "strict" uses only the stated *params* plus the declared
    STRICT_DEFAULTS (robust 0, the AA* array, band centre and default
    bandwidth for the stated band), and lists every default it applied.
    "best_fit" (default, the old behaviour) also tries every weighting
    scheme when the params don't state one, and every combination of the
    values in *sweep* (e.g. {"freq_centre_hz": [1.31e9, 1.355e9]}); it
    returns all variants, the closest, the number of free parameters, and the
    strict result alongside, so a match that needed free parameters is easy
    to tell from one that didn't.

    *endpoint* "continuum", "zoom" or "pss" (pulsar search: one call, a plain
    flux density). For continuum, *quantity* picks "continuum" or "spectral"
    (per channel); for zoom, *window_index* picks the window. The claim may
    be in Jy (any prefix, per beam or not) or in K, converted to Jy/beam with
    the frequency and *claimed_beam_arcsec* (or the calculator's beam).
    *sigma_level* says what multiple of the rms the claim is (5 for a 5σ
    limit); *stokes* "V" compares with thermal noise only;
    *claimed_channel_width_hz* rescales a spectral figure to the paper's
    channel width. Non-integer robustness is interpolated between the
    neighbouring integers. When the calculator refuses an array for a band
    and the 15-m-only subarray works, the result says so in
    "array_substituted". Variants whose call fails are reported with their
    error rather than aborting.
    """
    if endpoint not in ("continuum", "zoom", "pss"):
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected continuum, zoom or pss")
    if mode not in ("strict", "best_fit"):
        raise ValueError(f"Unknown mode {mode!r}; expected strict or best_fit")
    if sigma_level <= 0:
        raise ValueError("sigma_level must be positive")
    kelvin = claimed_sensitivity * _K_PER[unit.lower().replace(" ", "")] if _is_kelvin(unit) else None
    claimed = {"jy": None if kelvin is not None else _to_jy(claimed_sensitivity, unit) / sigma_level,
               "kelvin": kelvin / sigma_level if kelvin is not None else None}
    q = "spectral" if endpoint == "zoom" else quantity

    def run(variant: dict, sent: dict) -> dict:
        try:
            got = _evaluate(telescope, endpoint, sent, q, window_index, stokes)
            return _row(variant, got, claimed, telescope, endpoint, sent, window_index,
                        claimed_beam_arcsec, claimed_channel_width_hz)
        except requests.HTTPError as exc:
            return {**variant, "error": str(exc)[-300:]}

    # Strict: stated params plus declared defaults, one call.
    try:
        strict_params, applied = _strict_params(telescope, endpoint, params)
        strict = {"params_used": strict_params, "defaults_applied": applied,
                  "result": run({}, strict_params)}
        if "error" in strict["result"]:
            strict["error"] = strict["result"].pop("error")
    except ValueError as exc:
        strict = {"error": str(exc), "defaults_applied": {}}

    claimed_out = {
        "claimed": claimed_sensitivity, "claimed_unit": unit, "sigma_level": sigma_level, "stokes": stokes.upper(),
        "claimed_1sigma_ujy": round(claimed["jy"] * 1e6, 4) if claimed["jy"] is not None else None,
    }
    if mode == "strict":
        res = strict.get("result") or {}
        return {"mode": "strict", "endpoint": endpoint, **claimed_out,
                "unit": "uJy" if endpoint == "pss" else "uJy/beam", "quantity": q, "strict": strict,
                "closest": res if "pct_diff" in res else None,
                "within_10pct": [res] if abs(res.get("pct_diff", 1e9)) <= 10 else [],
                "array_substituted": res.get("array_substituted"), "n_free_parameters": 0}

    weightings = [{}] if endpoint == "pss" or "weighting_mode" in params else WEIGHTING_VARIANTS
    sweep = sweep or {}
    combos = [dict(zip(sweep, values)) for values in itertools.product(*sweep.values())] or [{}]
    rows = [run({**combo, **w}, {**params, **combo, **w}) for combo in combos for w in weightings]
    ok = [r for r in rows if "pct_diff" in r]
    closest = min(ok, key=lambda r: abs(r["pct_diff"]) + abs(r.get("beam_pct_diff", 0)), default=None)
    return {
        "mode": "best_fit",
        "endpoint": endpoint,
        "claimed_ujy": claimed_out["claimed_1sigma_ujy"],  # kept for existing callers
        **claimed_out,
        "unit": "uJy" if endpoint == "pss" else "uJy/beam",
        "quantity": "folded/single pulse" if endpoint == "pss" else q,
        "n_free_parameters": len(sweep) + (len(weightings) > 1),
        "free_parameters": list(sweep) + (["weighting"] if len(weightings) > 1 else []),
        "closest": closest,
        "within_10pct": [r for r in ok if abs(r["pct_diff"]) <= 10],
        "array_substituted": (closest or {}).get("array_substituted"),
        "variants": rows,
        "strict": strict,
    }
