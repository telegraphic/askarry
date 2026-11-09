"""
Thin wrapper around the vendored SKA data-product size estimator
(rag/vendor/data_size/, from ska-sci-ops/odp-data-size-tool), exposing a
single dict-in/dict-out function suitable for an MCP tool.
"""

from __future__ import annotations

import math

from astropy.units import Quantity

from rag.config import DATA_PRODUCT_POLICY
from rag.vendor.data_size.data_products import CalibratedVisibilities, Image
from rag.vendor.data_size.model_io import _PTYPE_CLASS, row_to_model

# Modes sized here rather than in the vendored tool, for whole observations
# the zoom-window model gets wrong: an HI survey over a whole band (sizing it
# as one 4,000 x 13.44 kHz zoom window, 54 MHz wide, under-counts it by an
# order of magnitude) and single-dish intensity mapping (autocorrelations
# only: no cross-correlation visibilities to store).
WIDEBAND_CUBE = "Spectral Cube (wide-band)"
AUTOCORRELATION = "Single-dish Autocorrelation"
PRODUCT_TYPES = sorted(_PTYPE_CLASS) + [WIDEBAND_CUBE, AUTOCORRELATION]

# Native continuum-correlator channel widths, kHz (setup-validator frontend
# defaults; Low's is 781.25 kHz / 144).
_NATIVE_CHANNEL_KHZ = {"mid": 13.44, "low": 5.425}


def _channels(params: dict) -> tuple[float, float, int]:
    """(bandwidth Hz, channel width Hz, number of channels) from
    freq_min_mhz/freq_max_mhz (or bandwidth_mhz) and channel_width_khz
    (default: the telescope's native continuum channel)."""
    if "bandwidth_mhz" in params:
        bw = float(params["bandwidth_mhz"]) * 1e6
    else:
        bw = (float(params["freq_max_mhz"]) - float(params["freq_min_mhz"])) * 1e6
    if bw <= 0:
        raise ValueError("bandwidth must be positive (freq_max_mhz > freq_min_mhz)")
    cw = float(params.get("channel_width_khz", _NATIVE_CHANNEL_KHZ[params.get("telescope", "mid")])) * 1e3
    return bw, cw, math.ceil(round(bw / cw, 3))


def _tb(q: Quantity) -> float:
    return float(q.to("TB").value)


def _tiers(visibilities_tb: float | None, products: dict[str, float], delivered_tb: float) -> dict:
    """Visibilities, what the archive keeps under DATA_PRODUCT_POLICY, and
    what the science team receives."""
    archived = sum(v for k, v in products.items() if DATA_PRODUCT_POLICY["archived"].get(k, False))
    return {"visibilities_TB": visibilities_tb, "archived_TB": archived, "delivered_TB": delivered_tb}


def _wideband_cube(params: dict) -> dict:
    """A spectral-line cube over a whole band (an HI survey), with the
    visibilities behind it."""
    bw, cw, n_chan = _channels(params)
    duration = Quantity(float(params["duration_h"]) * 3600, "s")
    vis = CalibratedVisibilities(n_station=int(params["n_station"]), t_int=float(params["t_int_s"]), bw=bw, cbw=cw,
                                 n_stokes=4, duration=duration, storage_model=params.get("storage_model", "msv2"))
    cube = Image(resolution=Quantity(float(params["resolution_arcsec"]), "arcsec").to("deg"),
                 fov=Quantity(float(params["fov_deg"]), "deg"), n_chan=n_chan,
                 n_stokes=int(params.get("n_stokes", 1)), n_beam=int(params.get("n_beam", 1)),
                 n_bit=int(params.get("n_bit", 32)), n_psf_osamp=int(params.get("n_psf_osamp", 3)))
    cube_tb = _tb(cube.data_volume) * int(params.get("n_pointings", 1))
    return {"product_type": WIDEBAND_CUBE, "bandwidth_MHz": bw / 1e6, "channel_width_kHz": cw / 1e3,
            "n_chan": n_chan, "n_pix": cube.n_pix,
            "tiers_TB": _tiers(_tb(vis.data_volume), {WIDEBAND_CUBE: cube_tb}, cube_tb)}


def _autocorrelation(params: dict) -> dict:
    """Single-dish intensity mapping: one autocorrelation spectrum per dish,
    polarisation and dump, and optionally the gridded maps delivered."""
    bw, cw, n_chan = _channels(params)
    n_dumps = math.ceil(float(params["duration_h"]) * 3600 / float(params["t_int_s"]))
    bytes_per = int(params.get("n_bit", 32)) / 8
    spectra_tb = int(params["n_dish"]) * int(params.get("n_pol", 2)) * n_chan * n_dumps * bytes_per / 1e12
    delivered = spectra_tb
    if params.get("map_area_deg2") and params.get("pixel_arcmin"):
        n_pix = float(params["map_area_deg2"]) * 3600 / float(params["pixel_arcmin"]) ** 2
        delivered = n_pix * n_chan * int(params.get("n_stokes", 1)) * bytes_per / 1e12
    return {"product_type": AUTOCORRELATION, "bandwidth_MHz": bw / 1e6, "channel_width_kHz": cw / 1e3,
            "n_chan": n_chan, "n_dumps": n_dumps,
            # No cross-correlations are needed, so there are no visibilities to store.
            "tiers_TB": _tiers(None, {AUTOCORRELATION: spectra_tb}, delivered)}


def estimate_data_product_size(product_type: str, params: dict) -> dict:
    """Estimate the data volume (and rate, where applicable) of an SKA data product.

    Args:
        product_type: one of PRODUCT_TYPES: the vendored types ("Image",
            "Image Cutout", "Gridded Visibilities", "PSS", "PST Folded",
            "Dynamic Spectrum", "Flowthrough", "VLBI", "Transient Dump",
            "Calibrated Vis"), or WIDEBAND_CUBE / AUTOCORRELATION, sized
            here for a whole observation (see _wideband_cube and
            _autocorrelation for their params).
        params: dict of parameter values for that product type, matching
            row_to_model's expected row dict shape for that ptype. Angular
            values (resolution, fov) and durations are given as unit strings
            (e.g. "1.000 arcsec", "3600.000 s"); bandwidths (bw, cbw) accept
            either a plain Hz number or a unit string (e.g. "300.000 MHz");
            everything else is a plain number/bool/string. See row_to_model
            in rag/vendor/data_size/model_io.py for the exact required keys
            per product_type.

    Returns:
        dict with data_volume_TB (float), data_rate_TBs (float, or None if
        the product type doesn't define a data rate), whether the archive
        keeps it, and the policy source. The two whole-observation modes
        also return tiers_TB: visibilities_TB (None for autocorrelations),
        archived_TB and delivered_TB.

    Raises:
        KeyError: product_type is not a recognised key.
        Exception: params is missing a required key for that product_type,
            or a value can't be parsed (propagated from row_to_model/astropy;
            left for the caller to catch and format).
    """
    if product_type in (WIDEBAND_CUBE, AUTOCORRELATION):
        out = (_wideband_cube if product_type == WIDEBAND_CUBE else _autocorrelation)(params)
        out["data_volume_TB"] = out["tiers_TB"]["archived_TB"] or out["tiers_TB"]["delivered_TB"]
        out["data_rate_TBs"] = None
        out["policy_source"] = DATA_PRODUCT_POLICY["source"]
        return out
    model = row_to_model(product_type, params)
    result = {"data_volume_TB": float(model.data_volume.to("TB").value)}
    if hasattr(model, "data_rate"):
        result["data_rate_TBs"] = float(model.data_rate.to("TB/s").value)
    else:
        result["data_rate_TBs"] = None
    # One product: say whether the archive keeps it (an assumption, stated).
    result["archived"] = DATA_PRODUCT_POLICY["archived"].get(product_type, False)
    result["policy_source"] = DATA_PRODUCT_POLICY["source"]
    return result
