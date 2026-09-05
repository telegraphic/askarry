# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/continuum.py).
"""rules/continuum.py - Rules for CONT mode."""

from ..obs_config import ObsConfig
from ..unit_utils import to_quantity, to_si_prefix
from ..validation_schema import get_minmax


def ska_low_continuum_bandwidth_is_valid(obs_config: ObsConfig) -> bool:
    """Check SKA-Low beam bandwidth total across beams is ok.

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    if obs_config.telescope == "SKA-Mid":
        return True
    else:
        bw_total = 0

    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            if b.continuum.enabled:
                if b.continuum.bandwidth is None:
                    return False
                bw_total += b.continuum.bandwidth

    _bw_min, bw_max, _bw_step = get_minmax(
        "continuum_settings", "bandwidth", telescope="SKA-Low", context=obs_config.context
    )
    bw_max = to_quantity(bw_max)
    bw_total = to_quantity(bw_total, "MHz")
    print(bw_total, bw_max)
    if bw_total > bw_max:
        return (False, {"bw_total": to_si_prefix(bw_total), "bw_max": to_si_prefix(bw_max)})
    else:
        return True
