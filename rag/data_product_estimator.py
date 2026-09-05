"""
Thin wrapper around the vendored SKA data-product size estimator
(rag/vendor/data_size/, from ska-sci-ops/odp-data-size-tool), exposing a
single dict-in/dict-out function suitable for an MCP tool.
"""

from __future__ import annotations

from rag.vendor.data_size.model_io import row_to_model


def estimate_data_product_size(product_type: str, params: dict) -> dict:
    """Estimate the data volume (and rate, where applicable) of an SKA data product.

    Args:
        product_type: one of the keys in model_io's _PTYPE_CLASS, e.g.
            "Image", "Image Cutout", "Gridded Visibilities", "PSS",
            "PST Folded", "Dynamic Spectrum", "Flowthrough", "VLBI",
            "Transient Dump", "Calibrated Vis".
        params: dict of parameter values for that product type, matching
            row_to_model's expected row dict shape for that ptype. Angular
            values (resolution, fov) and durations are given as unit strings
            (e.g. "1.000 arcsec", "3600.000 s"); bandwidths (bw, cbw) accept
            either a plain Hz number or a unit string (e.g. "300.000 MHz");
            everything else is a plain number/bool/string. See row_to_model
            in rag/vendor/data_size/model_io.py for the exact required keys
            per product_type.

    Returns:
        dict with data_volume_TB (float) and data_rate_TBs (float, or None
        if the product type doesn't define a data rate).

    Raises:
        KeyError: product_type is not a recognised key.
        Exception: params is missing a required key for that product_type,
            or a value can't be parsed (propagated from row_to_model/astropy;
            left for the caller to catch and format).
    """
    model = row_to_model(product_type, params)
    result = {"data_volume_TB": float(model.data_volume.to("TB").value)}
    if hasattr(model, "data_rate"):
        result["data_rate_TBs"] = float(model.data_rate.to("TB/s").value)
    else:
        result["data_rate_TBs"] = None
    return result
