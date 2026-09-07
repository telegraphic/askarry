# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/frontend_defaults.py).
"""frontend_defaults.py - Return default values for the frontend."""

import os

from .telescope.receivers import receiver_dict
from .unit_utils import to_quantity
from .validation_schema import get_minmax, load_yaml_or_json


def _get_max(schema: str, param_id: str, context: str = None, telescope: str = None, band: str = None) -> float:
    rx = receiver_dict[telescope][band]
    rp = {"$rx_low": rx["low"], "$rx_high": rx["high"], "$rx_bw": rx["bw"]}
    _min, mmax, _step = get_minmax(schema, param_id, context=context, telescope=telescope, runtime_params=rp)
    return to_quantity(mmax).value


def get_defaults(context: str) -> dict:
    """Return a dictionary containing default values that are needed by the frontend.

    Args:
        telescope (str): Name of telescope, e.g. 'SKA-Mid' or 'SKA-Low'.
        context (str): Name of context, e.g. 'SV-AA2' or 'Cycle 0'.

    Notes:
        "continuumBandwidth": 300,
        "maxContinuumBandwidth": 300,
        "pstBandwidth": 300,
        "maxPstBandwidth": 300,
        "pssBandwidth": 118,
        "maxPssBandwidth": 118,
        "maxNumberOfZoomChannels": 64000,
    """
    fp = os.path.dirname(os.path.realpath(__file__))

    # Load frontend defaults (YAML preferred, JSON fallback)
    defaults = None
    for ext in (".yaml", ".yml", ".json"):
        candidate = f"{fp}/schema/frontend_defaults{ext}"
        if os.path.exists(candidate):
            defaults = load_yaml_or_json(candidate)
            break
    if defaults is None:
        raise FileNotFoundError("Could not locate frontend_defaults in YAML/JSON form")

    telescope = "SKA-Mid"
    for band in defaults["skaMidDefaults"]:
        band_dfl = defaults["skaMidDefaults"][band]
        band_dfl["continuumBandwidth"] = _get_max(
            "continuum_settings", "bandwidth", context=context, telescope=telescope, band=band
        )
        band_dfl["maxContinuumBandwidth"] = band_dfl["continuumBandwidth"]
        band_dfl["pssBandwidth"] = _get_max(
            "pss_settings", "bandwidth", context=context, telescope=telescope, band=band
        )
        band_dfl["maxPssBandwidth"] = band_dfl["pssBandwidth"]
        band_dfl["pstBandwidth"] = _get_max(
            "odps/pst_dynamic_spectrum", "bandwidth", context=context, telescope=telescope, band=band
        )
        band_dfl["maxPstBandwidth"] = band_dfl["pstBandwidth"]
        band_dfl["maxPstBeams"] = _get_max(
            "odps/odp_config", "n_pst_odps", context=context, telescope=telescope, band=band
        )
        band_dfl["maxNumberOfZoomChannels"] = _get_max(
            "odps/image_cube", "n_channel_zoom", context=context, telescope=telescope, band=band
        )
        band_dfl["maxChannelCount"] = _get_max(
            "odps/image_cube", "n_channel", context=context, telescope=telescope, band=band
        )
        band_dfl["maxSpectralChannelCount"] = _get_max(
            "odps/image_cube", "n_channel_spectral", context=context, telescope=telescope, band=band
        )

    telescope, band = "SKA-Low", "SKALA"
    low_dfl = defaults["skaLowDefaults"]
    low_dfl["continuumBandwidth"] = _get_max(
        "continuum_settings", "bandwidth", context=context, telescope=telescope, band=band
    )
    low_dfl["maxContinuumBandwidth"] = low_dfl["continuumBandwidth"]
    low_dfl["pssBandwidth"] = _get_max("pss_settings", "bandwidth", context=context, telescope=telescope, band=band)
    low_dfl["maxPstBeams"] = _get_max("odps/odp_config", "n_pst_odps", context=context, telescope=telescope, band=band)
    low_dfl["maxPssBandwidth"] = low_dfl["pssBandwidth"]
    low_dfl["pstBandwidth"] = _get_max(
        "odps/pst_dynamic_spectrum", "bandwidth", context=context, telescope=telescope, band=band
    )
    low_dfl["maxPstBandwidth"] = low_dfl["pstBandwidth"]
    low_dfl["maxNumberOfZoomChannels"] = _get_max(
        "odps/image_cube", "n_channel_zoom", context=context, telescope=telescope, band=band
    )
    low_dfl["maxChannelCount"] = _get_max(
        "odps/image_cube", "n_channel", context=context, telescope=telescope, band=band
    )
    low_dfl["maxSpectralChannelCount"] = _get_max(
        "odps/image_cube", "n_channel_spectral", context=context, telescope=telescope, band=band
    )
    return defaults
