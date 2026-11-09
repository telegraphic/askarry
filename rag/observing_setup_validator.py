"""
Thin client for the SKAO observing-setup validator core engine, vendored from
ska-sci-ops-setup-validator-backend (see rag/vendor/setup_validator/).

Validates a telescope observing configuration (subarrays, beams, continuum/
zoom/PSS/PST mode settings, output data products) against the SKA validation
schema/rules for a given context (e.g. "Cycle 0", "SV-AA*") and telescope
("SKA-Low" or "SKA-Mid"). Same JSON shape as the ska-sci-ops-setup-validator
frontend GUI submits to its backend API.
"""

from __future__ import annotations


def validate_observing_setup(obs_config: dict) -> dict:
    """Validate an SKA telescope observing configuration.

    Args:
        obs_config: Dict with keys "context", "telescopeType", and "subarrays"
            (per-subarray template/band/mode settings and output data
            products) — see rag/vendor/setup_validator/schema/obs_config.yaml
            for allowed context/telescope values.

    Returns:
        dict with "status" ("success" or "error"), "message", and (on error)
        "alerts" — a list of human-readable validation failure messages.
    """
    from rag.vendor.setup_validator.frontend_utils import validate

    return validate(obs_config)


# Contexts accepted by the validator schema (schema/obs_config.yaml).
CONTEXTS = ["SV-AA2", "SV-AA*", "Cycle 0", "Cycle 1"]
# Evaluated by default: the validator's contexts plus the AA4 design
# baseline (the schema's "base" layer, which no observing context offers yet).
AA4 = "AA4"
DEFAULT_CONTEXTS = CONTEXTS + [AA4]

# Observing modes with no schema in any context, including the AA4 baseline:
# not offered at all, which is a different finding from "only at AA4".
MODE_SCHEMAS = {"continuum": "continuum_settings", "zoom": "zoom_settings", "pss": "pss_settings", "pst": "pst_settings"}
MODES_NOT_OFFERED = {"vlbi", "transient_buffer", "solar"}
_MODE_ALIASES = {"spectral line": "zoom", "spectral": "zoom", "zoom mode": "zoom", "pulsar search": "pss",
                 "pulsar timing": "pst", "vlbi": "vlbi", "ska-vlbi": "vlbi", "transient buffer": "transient_buffer",
                 "tb": "transient_buffer", "solar": "solar", "imaging": "continuum"}
# Worst first. An unchecked dimension doesn't block a pass; it is listed under "not_checked".
VERDICT_ORDER = ["mode_not_offered", "fail", "pass", "not_checked"]
C_LIGHT = 299_792_458.0
ARCSEC_PER_RAD = 206264.806


def validate_across_contexts(obs_config: dict | None = None, contexts: list[str] | None = None,
                             requirement: dict | None = None, check_sensitivity: bool = True) -> dict:
    """Judge an observing configuration or a requirement in each context.

    With *obs_config* only (the original use), validates it under each
    validator context and returns {context: validation result}; the AA4
    design baseline has no validator context, so it is reported as not
    validated. Default contexts: SV-AA2, SV-AA*, Cycle 0, Cycle 1, AA4.

    With *requirement* (a rag.requirements record), returns a verdict per
    context for each dimension the record states (observing mode, band and
    frequency, bandwidth, channel width, angular resolution, sensitivity,
    time resolution) and an overall summary. See requirement_verdicts.
    """
    contexts = contexts or DEFAULT_CONTEXTS
    if requirement is not None:
        out = requirement_verdicts(requirement, contexts, check_sensitivity)
        if obs_config is not None:
            for ctx in contexts:
                out["by_context"][ctx]["obs_config_validation"] = _validate_one(obs_config, ctx)
        return out
    if obs_config is None:
        raise ValueError("Give an obs_config, a requirement, or both")
    return {context: _validate_one(obs_config, context) for context in contexts}


def _validate_one(obs_config: dict, context: str) -> dict:
    if context == AA4:
        return {"status": "not_validated",
                "message": "The AA4 design baseline has no setup-validator context; pass a requirement for "
                           "per-dimension verdicts against the base schema."}
    return validate_observing_setup({**obs_config, "context": context})


def _schema_context(context: str) -> str:
    return "base" if context == AA4 else context.lower().replace("-", "_").replace(" ", "_").replace("*", "star")


def _stage_template(telescope: str, context: str) -> str:
    """The full-array template a context observes with (from the validator's
    own frontend defaults; AA4 is the design baseline)."""
    from rag.observing_setup_capabilities import get_context_defaults

    tel = "Mid" if telescope == "mid" else "Low"
    if context == AA4:
        return f"{tel}_full_AA4"
    return get_context_defaults(_schema_context(context))["defaultSubarrayTemplates"][f"SKA-{tel}"][context]


def _calculator_subarray(template: str) -> str:
    """The calculator's label for a full-array template (Mid_full_AAstar -> "AA*")."""
    stage = template.split("_")[-1]
    return {"AAstar": "AA*", "AA2": "AA2", "AA4": "AA4"}.get(stage, stage)


def _band_limits(telescope: str, context: str, band: str | None) -> dict | None:
    from rag.observing_setup_capabilities import get_context_defaults

    d = get_context_defaults("base" if context == AA4 else _schema_context(context))
    if telescope == "low":
        return d["skaLowDefaults"]
    return d["skaMidDefaults"].get(band) if band else None


def _mode(requirement: dict) -> str | None:
    from rag.requirements import value

    m = value(requirement, "mode")
    if not m:
        return None
    m = str(m).strip().lower()
    return _MODE_ALIASES.get(m, m.replace(" ", "_"))


def _row(verdict: str, required=None, available=None, note: str | None = None) -> dict:
    row = {"verdict": verdict}
    if required is not None:
        row["required"] = required
    if available is not None:
        row["available"] = available
    if note:
        row["note"] = note
    return row


def _band_for(telescope: str, freq_hz: float | None, band: str | None) -> str | None:
    if band or telescope != "mid" or not freq_hz:
        return band
    from rag.observing_setup_capabilities import get_context_defaults

    for b, v in get_context_defaults("base")["skaMidDefaults"].items():
        if v["minCentralFrequency"] * 1e6 <= freq_hz <= v["maxCentralFrequency"] * 1e6:
            return b
    return None


def _context_dimensions(req: dict, telescope: str, context: str, check_sensitivity: bool) -> dict:
    """Per-dimension verdicts for one context. Only dimensions the record states are judged."""
    from rag.observing_setup_capabilities import describe_schema
    from rag.requirements import value
    from rag.subarray_layout import get_subarray_layout

    tel_id = f"ska_{telescope}"
    ctx = _schema_context(context)
    dims: dict[str, dict] = {}
    freq = value(req, "frequency", "Hz")
    band = value(req, "band")
    band = f"Band {band}" if band and not str(band).lower().startswith("band") else band
    band = _band_for(telescope, freq, band.title().replace("Band ", "Band ") if band else None)
    if band:
        band = "Band " + band.split()[-1].lower() if band.split()[-1].lower() in ("5a", "5b") else band
    limits = _band_limits(telescope, context, band)
    if freq is None and limits:
        freq = float(limits["centralFrequency"]) * 1e6  # band centre, for resolution and conversions

    mode = _mode(req)
    if mode:
        if mode in MODES_NOT_OFFERED or mode not in MODE_SCHEMAS:
            dims["mode"] = _row("mode_not_offered", mode, note="no observing-mode schema in any context, "
                                                               "including the AA4 design baseline")
        else:
            enabled = describe_schema(MODE_SCHEMAS[mode], context=ctx, telescope=tel_id).get("enabled", {})
            ok = True in enabled.get("allowed_values", [True])
            dims["mode"] = _row("pass" if ok else "fail", mode, f"enabled={enabled.get('allowed_values')}")

    if telescope == "mid" and band:
        allowed = describe_schema("subarray_config", context=ctx, telescope=tel_id).get("band", {}).get("allowed_values", [])
        dims["band"] = _row("pass" if band in allowed else "fail", band, allowed)
    if value(req, "frequency") is not None and limits:
        lo, hi = limits["minCentralFrequency"], limits["maxCentralFrequency"]
        dims["frequency"] = _row("pass" if lo * 1e6 <= freq <= hi * 1e6 else "fail", f"{freq / 1e6:g} MHz",
                                 f"{lo}-{hi} MHz")
    elif value(req, "frequency") is not None:
        dims["frequency"] = _row("not_checked", note="no band limits for this frequency")

    bw = value(req, "bandwidth", "MHz")
    if bw is not None:
        if limits:
            top = float(limits["maxContinuumBandwidth"])
            dims["bandwidth"] = _row("pass" if bw <= top else "fail", f"{bw:g} MHz", f"<= {top:g} MHz")
        else:
            dims["bandwidth"] = _row("not_checked", f"{bw:g} MHz", note="band not stated")

    cw_entry = (req.get("channel_width") or {})
    if cw_entry.get("value") is not None:
        if cw_entry.get("unit") == "km/s":
            cw = cw_entry["value"] * 1e3 / C_LIGHT * freq if freq else None
        else:
            cw = value(req, "channel_width", "Hz")
        widths = []
        zoom = describe_schema("zoom_settings", context=ctx, telescope=tel_id)
        if True in zoom.get("enabled", {}).get("allowed_values", []):
            widths += [_hz(w) for w in zoom.get("channelWidth", {}).get("allowed_values", [])]
        from rag.observing_setup_capabilities import get_context_defaults
        dd = get_context_defaults(ctx)
        widths.append(float(dd["skaMidContChannelWidth" if telescope == "mid" else "skaLowContChannelWidth"]) * 1e3)
        finest = min(w for w in widths if w)
        if cw is None:
            dims["channel_width"] = _row("not_checked", note="velocity width with no frequency")
        else:
            dims["channel_width"] = _row("pass" if cw >= finest * 0.99 else "fail", f"{cw:.4g} Hz",
                                         f">= {finest:.4g} Hz (finest offered)")

    # Sensitivity (strict calculator), whose beam also judges resolution.
    calc_beam = None
    sens = value(req, "sensitivity")
    hours = value(req, "hours", "h")
    if sens is not None:
        if not check_sensitivity:
            dims["sensitivity"] = _row("not_checked", note="check_sensitivity=false")
        elif hours is None or (telescope == "mid" and not band):
            dims["sensitivity"] = _row("not_checked", note="needs hours" + (" and a band" if telescope == "mid" else ""))
        else:
            dims["sensitivity"], calc_beam = _sensitivity(req, telescope, context, band, freq, hours)

    res = value(req, "angular_resolution", "arcsec")
    if res is not None:
        template = _stage_template(telescope, context)
        if calc_beam:
            beam = sum(calc_beam) / 2
            dims["angular_resolution"] = _row("pass" if beam <= res * 1.1 else "fail", f"{res:g} arcsec",
                                              f"{beam:.3g} arcsec", "calculator beam at the weighting used for sensitivity")
        elif freq:
            bmax = get_subarray_layout(template)["max_baseline_m"]
            best = C_LIGHT / freq / bmax * ARCSEC_PER_RAD
            dims["angular_resolution"] = _row("pass" if best <= res else "fail", f"{res:g} arcsec", f"{best:.3g} arcsec",
                                              f"best case lambda/B_max, {template} (B_max {bmax / 1e3:.1f} km)")
        else:
            dims["angular_resolution"] = _row("not_checked", note="no frequency or band")

    if value(req, "time_resolution") is not None:
        dims["time_resolution"] = _row("not_checked", note="the validator schema has no time-resolution limits")
    return dims


def _hz(text: str) -> float:
    num, unit = str(text).split()
    return float(num) * {"Hz": 1, "kHz": 1e3, "MHz": 1e6}[unit]


def _sensitivity(req: dict, telescope: str, context: str, band: str | None, freq: float | None,
                 hours: float) -> tuple[dict, list | None]:
    """Strict calculator rms for the context's full array vs the required rms."""
    import requests

    from rag.requirements import value
    from rag.sensitivity_calculator import compare_to_calculator

    entry = req["sensitivity"]
    unit = entry.get("unit") or "uJy/beam"
    params: dict = {"subarray_configuration": _calculator_subarray(_stage_template(telescope, context))}
    bw = value(req, "bandwidth", "Hz")
    if telescope == "mid":
        params.update(rx_band=band, integration_time_s=hours * 3600)
        if freq:
            params["freq_centre_hz"] = freq
        if bw:
            params["bandwidth_hz"] = bw
    else:
        params.update(integration_time_h=hours)
        if freq:
            params["freq_centre_mhz"] = freq / 1e6
        if bw:
            params["bandwidth_mhz"] = bw / 1e6
    w = value(req, "weighting")
    if w:
        w = str(w).lower()
        params.update({"weighting_mode": "robust", "robustness": float(w.split()[-1])} if "robust" in w
                      else {"weighting_mode": w.split()[0]})
    try:
        out = compare_to_calculator(telescope, params, entry["value"], unit=unit, mode="strict",
                                    sigma_level=value(req, "sigma_level") or 1.0,
                                    claimed_beam_arcsec=value(req, "angular_resolution", "arcsec"))
    except (ValueError, requests.RequestException) as exc:
        return _row("not_checked", note=str(exc)[:200]), None
    res = out["strict"].get("result") or {}
    if "pct_diff" not in res:
        return _row("not_checked", note=out["strict"].get("error", "no result")[-200:]), None
    # pct_diff = (calculator - required) / required: negative means deeper than needed.
    row = _row("pass" if res["pct_diff"] <= 10 else "fail", f"{entry['value']} {unit}",
               f"{res['sensitivity_ujy']} uJy/beam", f"strict calculator, {params['subarray_configuration']}, "
               f"defaults {out['strict']['defaults_applied']}")
    if res.get("array_substituted"):
        row["array_substituted"] = res["array_substituted"]
    return row, res.get("beam_arcsec")


def requirement_verdicts(requirement: dict, contexts: list[str] | None = None, check_sensitivity: bool = True) -> dict:
    """Per-context, per-dimension verdicts for one requirement record.

    Dimension verdicts: "pass", "fail", "not_checked" (stated but not
    checkable here, with the reason), and for the observing mode
    "mode_not_offered" (no schema in any context: VLBI, transient buffer,
    solar). Each context's verdict is its worst dimension, ignoring
    unchecked ones, which are listed under "not_checked". The summary is:
    "mode_not_offered"; "available_from <context>" (passes every checked
    dimension in that context and, typically, later ones); "aa4_only"
    (passes only at the AA4 design baseline); or "not_met" (fails everywhere).
    A band that exists is not enough: resolution and depth are judged too.
    """
    from rag.requirements import validate_record, value

    problems = validate_record(requirement)
    if problems:
        raise ValueError("; ".join(problems))
    telescope = str(value(requirement, "telescope") or "").lower().replace("ska-", "").replace("ska_", "")
    if telescope not in ("low", "mid"):
        raise ValueError("requirement needs telescope 'low' or 'mid'")
    contexts = contexts or DEFAULT_CONTEXTS
    by_context = {}
    for ctx in contexts:
        dims = _context_dimensions(requirement, telescope, ctx, check_sensitivity)
        worst = min((d["verdict"] for d in dims.values()), key=VERDICT_ORDER.index, default="not_checked")
        by_context[ctx] = {"verdict": worst, "dimensions": dims,
                           "not_checked": sorted(d for d, x in dims.items() if x["verdict"] == "not_checked")}
    verdicts = {c: v["verdict"] for c, v in by_context.items()}
    if "mode_not_offered" in verdicts.values():
        summary = "mode_not_offered"
    else:
        passing = [c for c in contexts if verdicts[c] == "pass"]
        early = [c for c in passing if c != AA4]
        summary = (f"available_from {early[0]}" if early else "aa4_only" if passing else
                   "not_met" if "fail" in verdicts.values() else "not_checked")
    failing = sorted({d for v in by_context.values() for d, x in v["dimensions"].items() if x["verdict"] == "fail"})
    return {"verdict": summary, "failing_dimensions": failing, "by_context": by_context}
