# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/beams.py).
"""rules/subarrays.py - check subarray templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..obs_config import ObsConfig


from ..logging import logger
from ..validation_schema import get_minmax


def number_of_beams_is_valid(obs_config: ObsConfig) -> (bool, dict):
    """Check total number of beams is valid for given context.

    Args:
        obs_config (ObsConfig): The observation configuration containing subarray details.

    Returns:
        is_pass (bool), fail_dict: Pass (True) or fail (False, {'n_beam': n_beam, 'n_max': n_max}).
    """
    _n_min, n_max, _n_step = get_minmax(
        "subarray_config", "n_beam", telescope=obs_config.telescope, context=obs_config.context
    )

    for sa in obs_config.config.subarrays:
        n_beam = len(sa.beams)

        if n_beam > n_max:
            logger.error(f"Number of beams {n_beam} exceeds maximum {n_max}.")
            return False, {"sa": sa.id, "n_beam": n_beam, "n_max": n_max}

    return True


def check_beam_has_enabled_mode(obs_config: ObsConfig) -> (bool, dict):
    """Check the beam has at least one mode enabled.

    Args:
        obs_config (ObsConfig): The observation configuration containing subarray details.

    Returns:
        is_pass (bool), fail_dict: Pass (True) or fail (False, {'errs': error_string}).
    """
    is_pass = True
    errs = ""
    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            is_enabled = False
            for mode in (b.continuum, b.pst, b.pss, b.zoom):
                if mode.enabled:
                    is_enabled = True
                    break

            if not is_enabled:
                err_msg = f"Beam {b.id} in subarray {sa.id} has no mode enabled."
                logger.error(err_msg)
                errs += err_msg + " \n"
                is_pass = False
    if is_pass:
        return True
    else:
        return False, {"errs": errs}
