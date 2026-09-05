# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/pss.py).
"""rules/pst.py - Rules for PST mode."""

from ..obs_config import ObsConfig
from ..validation_schema import get_minmax


def check_sum_pss_beams_is_valid(obs_config: ObsConfig) -> bool:
    """Check SKA-Mid PSS beams are in range.

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    _n_min, n_max, _n_step = get_minmax(
        "pss_settings", "numberOfBeams", telescope=obs_config.telescope, context=obs_config.context
    )
    n_beams = 0

    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            if b.pss.enabled:
                if b.pss.numberOfBeams:
                    n_beams += b.pss.numberOfBeams

    if n_beams <= n_max:
        return True
    else:
        return False
