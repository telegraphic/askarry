"""
Thin client for the SKAO sensitivity calculator REST API
(https://sensitivity-calculator.skao.int/api/v11/...), as documented in
pdfs/SKA_Key_Capabilities/Interacting with the SKAO sensitivity calculator.pdf.

Endpoints (per telescope, "low" or "mid"):
    subarrays           GET, no params — lists available subarray configs.
    continuum/calculate  GET, query params — continuum-mode sensitivity/integration-time.
    zoom/calculate       GET, query params — zoom/spectral-line-mode sensitivity.
    pss/calculate        GET, query params — pulsar search/timing sensitivity.
"""

from __future__ import annotations

import requests

BASE_URL = "https://sensitivity-calculator.skao.int/api/v11"
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

    url = f"{BASE_URL}/{telescope}/{endpoint}"
    response = requests.get(url, params=params or {}, timeout=TIMEOUT_S)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise requests.HTTPError(f"{exc} — response body: {detail}", response=response) from exc
    return response.json()


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
_JY_PER = {"jy": 1.0, "mjy": 1e-3, "ujy": 1e-6, "µjy": 1e-6, "njy": 1e-9}


def _to_jy(value: float, unit: str) -> float:
    return value * _JY_PER[unit.lower().replace(" ", "").replace("/beam", "")]


def _read_result(response: dict, endpoint: str, quantity: str, window: int) -> tuple[float, list | None]:
    """(sensitivity in Jy, [beam_maj, beam_min] arcsec or None) from one response."""
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
    got = result[f"total_{quantity}_sensitivity"] or result[f"weighted_{quantity}_sensitivity"]
    # weighting.*.beam_size is always populated, in degrees.
    beam = [round(beam_deg["beam_maj_scaled"] * 3600, 3), round(beam_deg["beam_min_scaled"] * 3600, 3)]
    return got["value"], beam


def compare_to_calculator(
    telescope: str,
    params: dict,
    claimed_sensitivity: float,
    unit: str = "uJy/beam",
    quantity: str = "continuum",
    claimed_beam_arcsec: float | None = None,
    endpoint: str = "continuum",
    window_index: int = 0,
) -> dict:
    """Re-run a paper's stated calculator parameters and report how far the
    result lands from the claimed figure.

    *endpoint* "continuum" or "zoom" sweeps every weighting scheme (papers
    often don't state it); "pss" (pulsar search, beamformed, no weighting) is
    a single call compared as a plain flux density. *params* are that
    endpoint's params minus weighting_mode/robustness. For continuum,
    *quantity* picks "continuum" or "spectral" (per-channel); for zoom,
    *window_index* picks the zoom window. Variants whose call fails are
    reported with their error rather than aborting the sweep.
    """
    if endpoint not in ("continuum", "zoom", "pss"):
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected continuum, zoom or pss")
    claimed_jy = _to_jy(claimed_sensitivity, unit)
    rows = []
    for variant in [{}] if endpoint == "pss" else WEIGHTING_VARIANTS:
        try:
            response = query_sensitivity_calculator(
                telescope, f"{endpoint}/calculate", {**params, **variant}
            )
        except requests.HTTPError as exc:
            rows.append({**variant, "error": str(exc)[:300]})
            continue
        got_jy, beam = _read_result(response, endpoint, quantity, window_index)
        row = {
            **variant,
            "sensitivity_ujy": round(got_jy * 1e6, 4),
            "pct_diff": round(100 * (got_jy - claimed_jy) / claimed_jy, 1),
        }
        if beam:
            row["beam_arcsec"] = beam
            if claimed_beam_arcsec:
                row["beam_pct_diff"] = round(100 * (sum(beam) / 2 - claimed_beam_arcsec) / claimed_beam_arcsec, 1)
        rows.append(row)

    ok = [r for r in rows if "pct_diff" in r]
    closest = min(ok, key=lambda r: abs(r["pct_diff"]) + abs(r.get("beam_pct_diff", 0)), default=None)
    return {
        "endpoint": endpoint,
        "claimed_ujy": round(claimed_jy * 1e6, 4),
        "unit": "uJy" if endpoint == "pss" else "uJy/beam",
        "quantity": "folded/single pulse" if endpoint == "pss" else ("spectral" if endpoint == "zoom" else quantity),
        "closest": closest,
        "within_10pct": [r for r in ok if abs(r["pct_diff"]) <= 10],
        "variants": rows,
    }
