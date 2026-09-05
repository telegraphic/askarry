# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/zoom.py).
"""rules/continuum.py - Rules for CONT mode."""

from ..obs_config import ObsConfig
from ..validation_schema import get_minmax


def number_zoom_channels_valid(obs_config: ObsConfig) -> bool:
    """Check number of zoom channels is valid for given context.

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    n_chan_total = 0

    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            if b.zoom.enabled:
                n_chan_total += b.zoom.numberOfZoomChannels

    n_min, n_max, _step = get_minmax(
        "zoom_settings", "numberOfZoomChannels", telescope=obs_config.telescope, context=obs_config.context
    )

    if n_chan_total > n_max:
        return (False, {"n_max": n_max, "n_chan_total": n_chan_total})
    else:
        return True
