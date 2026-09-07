"""
Bridges a validated SKA observing-setup config (rag.observing_setup_validator)
to real data-volume numbers (rag.data_product_estimator).

**Why some fields must be supplied by the caller, not derived here**

The setup-validator schema (rag/vendor/setup_validator/schema/base/*/odps/*.yaml
and */continuum|zoom_settings.yaml) tracks bandwidth, channel counts/widths,
polarizations, and SDP averaging *factors* for each output data product (ODP).
It does **not** track, anywhere: observation duration, raw (pre-averaging)
correlator dump time, or image resolution/field-of-view. These are not fields
we missed reading — they are genuine gaps in the schema (duration/dump-time
are scheduling-block concerns, resolution/fov depend on telescope+frequency+
baseline, all outside the validator's scope). Rather than guess values for
these (silently-wrong physics), this module always leaves them out of
``known_params`` and lists them in ``missing_params`` for the caller to fill
in via ``extra_params``.

Where the *data-size-tool* itself defines a class default for an optional
parameter (e.g. ``Image.n_bit=32``, ``CalibratedVisibilities.storage_model=
"msv2"``), we apply that same default here — that's the upstream library's own
stated default, not something invented by this module.

ODP type -> data-size-tool product_type ("ptype") support:
    calibrated_visibilities -> "Calibrated Vis"   (needs caller: t_int, duration)
    image_cube, mfs_image   -> "Image"            (needs caller: resolution, fov)
    pst_dynamic_spectrum    -> "Dynamic Spectrum"  (needs caller: duration)
    pst_flowthrough         -> "Flowthrough"       (needs caller: duration)
    pst_timing              -> unsupported (ptype None) - its schema only has
        centralFrequency/bandwidth, none of "PST Folded"'s required fields
        (n_bit, n_chan, n_stokes, n_beam, n_phase_bin, n_subint), so we do not
        attempt to auto-map it at all.
"""

from __future__ import annotations

from rag.data_product_estimator import estimate_data_product_size
from rag.observing_setup_validator import validate_observing_setup
from rag.subarray_layout import get_subarray_layout
from rag.vendor.setup_validator.frontend_utils import load_frontend_config

# ODP type -> data-size-tool ptype. None = not auto-mappable (see module docstring).
_ODP_TO_PTYPE = {
    "calibrated_visibilities": "Calibrated Vis",
    "image_cube": "Image",
    "mfs_image": "Image",
    "pst_dynamic_spectrum": "Dynamic Spectrum",
    "pst_flowthrough": "Flowthrough",
    "pst_timing": None,
}

# data-size-tool's own class defaults (rag/vendor/data_size/data_products.py).
_LIBRARY_DEFAULTS = {
    "Image": {
        "n_chan": 1, "n_stokes": 1, "n_products": 1, "n_beam": 1,
        "n_bit": 32, "n_timestep": 1, "n_psf_osamp": 3,
    },
    "Calibrated Vis": {
        "n_stokes": 1, "storage_model": "msv2",
        "include_weights": True, "include_uvw": True,
        "include_uncalibrated": False, "include_model": False,
    },
    "Dynamic Spectrum": {"n_beam": 1, "n_bit": 8},
    "Flowthrough": {"n_beam": 1, "osamp": 1.0},
}

# Required row_to_model() keys per ptype (rag/vendor/data_size/model_io.py).
_REQUIRED_PARAMS = {
    "Image": (
        "resolution", "fov", "n_chan", "n_stokes", "n_products",
        "n_beam", "n_bit", "n_timestep", "n_psf_osamp",
    ),
    "Calibrated Vis": (
        "n_station", "t_int", "bw", "cbw", "n_stokes", "duration",
        "storage_model", "include_uvw", "include_weights",
        "include_uncalibrated", "include_model",
    ),
    "Dynamic Spectrum": ("bw", "cbw", "t_int", "n_beam", "n_stokes", "n_bit", "duration"),
    "Flowthrough": ("bw", "n_beam", "osamp", "n_pol", "n_bit", "duration"),
}


def _parent_mode_bandwidth_hz(beam) -> tuple[float | None, float | None]:
    """Return (bandwidth_Hz, channel_width_Hz) from the beam's enabled continuum/zoom mode.

    calibrated_visibilities ODPs don't carry their own bandwidth - the
    absolute bandwidth and (pre-averaging) channel width live on the parent
    beam's continuum or zoom mode config instead.

    ponytail: if both continuum and zoom are enabled on the same beam we
    can't tell which mode a given calibrated_visibilities ODP belongs to
    (that link isn't kept once the config is parsed), so we prefer continuum
    over zoom. Revisit if that ever occurs in practice.
    """
    if beam.continuum is not None and getattr(beam.continuum, "enabled", False):
        bw_hz = beam.continuum.bandwidth * 1e6  # MHz -> Hz
        cw_hz = beam.continuum.channelWidth * 1e3  # kHz -> Hz
        return bw_hz, cw_hz
    if beam.zoom is not None and getattr(beam.zoom, "enabled", False):
        cw_hz = beam.zoom.channelWidth  # already Hz
        n_chan = beam.zoom.numberOfZoomChannels
        bw_hz = cw_hz * n_chan if (cw_hz and n_chan) else None
        return bw_hz, cw_hz
    return None, None


def _known_params_for_odp(odp_type: str, settings: dict, beam, layout: dict | None):
    """Return (ptype, known_params) derived from schema + ska_ost_array_config."""
    ptype = _ODP_TO_PTYPE.get(odp_type)
    if ptype is None:
        return None, {}

    known = dict(_LIBRARY_DEFAULTS.get(ptype, {}))

    if odp_type == "calibrated_visibilities":
        bw_hz, cw_hz = _parent_mode_bandwidth_hz(beam)
        if bw_hz is not None:
            known["bw"] = bw_hz
        n_freq_avg = settings.get("n_frequency_averaging")
        if cw_hz is not None and n_freq_avg is not None:
            known["cbw"] = cw_hz * n_freq_avg
        if layout is not None:
            known["n_station"] = layout["n_stations"]

    elif odp_type in ("image_cube", "mfs_image"):
        if settings.get("mode") == "zoom":
            n_chan, n_stokes = settings.get("n_channel_zoom"), settings.get("n_polarization_zoom")
        else:
            n_chan, n_stokes = settings.get("n_channel"), settings.get("n_polarization")
        if n_chan is not None:
            known["n_chan"] = n_chan
        if n_stokes is not None:
            known["n_stokes"] = n_stokes

    elif odp_type == "pst_dynamic_spectrum":
        if settings.get("bandwidth") is not None:
            known["bw"] = settings["bandwidth"] * 1e6  # MHz -> Hz
        if settings.get("frequencyResolution") is not None:
            known["cbw"] = settings["frequencyResolution"]  # already Hz
        if settings.get("timeResolution") is not None:
            known["t_int"] = settings["timeResolution"]  # already s
        if settings.get("n_polarization") is not None:
            known["n_stokes"] = settings["n_polarization"]

    elif odp_type == "pst_flowthrough":
        if settings.get("bandwidth") is not None:
            known["bw"] = settings["bandwidth"] * 1e6  # MHz -> Hz
        if settings.get("bitDepth") is not None:
            known["n_bit"] = settings["bitDepth"]
        if settings.get("n_polarization") is not None:
            known["n_pol"] = settings["n_polarization"]

    return ptype, known


def list_odps(obs_config: dict) -> list[dict]:
    """Validate an obs_config, then list every ODP it contains with what's derivable.

    Args:
        obs_config: Same dict shape accepted by
            rag.observing_setup_validator.validate_observing_setup.

    Returns:
        List of dicts, one per ODP: subarray_id, beam_id, odp_label, odp_type,
        ptype (mapped data-size-tool product_type, or None if unsupported),
        known_params (dict derived from schema + ska_ost_array_config),
        missing_params (list[str] of required data-size-tool keys not
        derivable - empty if ptype is None, since nothing was attempted).

    Raises:
        ValueError: obs_config fails validate_observing_setup.
    """
    result = validate_observing_setup(obs_config)
    if result.get("status") != "success":
        raise ValueError(f"Invalid observing setup: {result.get('alerts', result.get('message'))}")

    obs = load_frontend_config(config_dict=obs_config)
    entries = []
    for sa in obs.config.subarrays:
        try:
            layout = get_subarray_layout(sa.template)
        except ValueError:
            layout = None
        for beam in sa.beams:
            for odp in beam.odps:
                settings = odp.settings.param.values()
                ptype, known = _known_params_for_odp(odp.type, settings, beam, layout)
                missing = [k for k in _REQUIRED_PARAMS.get(ptype, ()) if k not in known] if ptype else []
                entries.append({
                    "subarray_id": sa.id,
                    "beam_id": beam.id,
                    "odp_label": odp.id,
                    "odp_type": odp.type,
                    "ptype": ptype,
                    "known_params": known,
                    "missing_params": missing,
                })
    return entries


def estimate_setup_data_volume(obs_config: dict, extra_params: dict[str, dict] | None = None) -> dict:
    """Estimate total data volume for every supported ODP in a validated obs_config.

    Args:
        obs_config: Same dict shape accepted by validate_observing_setup.
        extra_params: Optional dict keyed by "{subarray_id}/{beam_id}/{odp_label}",
            each value a dict of data-size-tool params that fill in / override
            what list_odps() couldn't derive (e.g. duration_s equivalent
            "duration": "3600.000 s", or for calibrated_visibilities a raw
            correlator dump time as "t_int": <seconds>).

    Returns:
        dict with:
            odps: list of {subarray_id, beam_id, odp_label, odp_type,
                data_volume_TB, data_rate_TBs} for every ODP successfully
                estimated.
            total_data_volume_TB: sum of the above.
            unresolved: list of {subarray_id, beam_id, odp_label, odp_type,
                missing_params} for supported ODPs still missing required
                params after merging extra_params (skipped, not raised).
            unsupported: sorted list of odp_types found that have no
                auto-mapping (e.g. "pst_timing").
    """
    extra_params = extra_params or {}
    entries = list_odps(obs_config)

    odps_out = []
    unresolved = []
    unsupported = set()
    total = 0.0

    for e in entries:
        if e["ptype"] is None:
            unsupported.add(e["odp_type"])
            continue

        key = f"{e['subarray_id']}/{e['beam_id']}/{e['odp_label']}"
        merged = {**e["known_params"], **extra_params.get(key, {})}
        missing = [k for k in _REQUIRED_PARAMS[e["ptype"]] if k not in merged]
        if missing:
            unresolved.append({
                "subarray_id": e["subarray_id"],
                "beam_id": e["beam_id"],
                "odp_label": e["odp_label"],
                "odp_type": e["odp_type"],
                "missing_params": missing,
            })
            continue

        result = estimate_data_product_size(e["ptype"], merged)
        odps_out.append({
            "subarray_id": e["subarray_id"],
            "beam_id": e["beam_id"],
            "odp_label": e["odp_label"],
            "odp_type": e["odp_type"],
            "data_volume_TB": result["data_volume_TB"],
            "data_rate_TBs": result["data_rate_TBs"],
        })
        total += result["data_volume_TB"]

    return {
        "odps": odps_out,
        "total_data_volume_TB": total,
        "unresolved": unresolved,
        "unsupported": sorted(unsupported),
    }
