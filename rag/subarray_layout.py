"""
Thin wrapper around ska_ost_array_config (an SKA-internal antenna-layout
package — see requirements.txt for the private-index install command),
exposing per-subarray-template physical layout facts (max baseline, station
count) that the setup-validator schema itself doesn't carry.
"""

from __future__ import annotations


def get_subarray_layout(subarray_template: str) -> dict:
    """Look up physical layout facts for an SKA subarray template.

    Args:
        subarray_template: A subarray template name, e.g. "Low_full_AA2",
            "Mid_full_AA4" (case-insensitive — matches the `template` values
            from describe_schema("subarray_config", ...) or an obs_config's
            subarray "template" field).

    Returns:
        dict with max_baseline_m (float, metres), n_stations (int), and
        n_baselines (int).

    Raises:
        ValueError: subarray_template is not a recognised template name.
    """
    from ska_ost_array_config import get_subarray_template

    try:
        template = get_subarray_template(subarray_template)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "max_baseline_m": float(template.max_bl),
        "n_stations": len(template.array_config.names.values),
        "n_baselines": int(template.n_bl),
    }
