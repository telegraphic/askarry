# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/subarrays.py).
"""rules/subarrays.py - check subarray templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..obs_config import ObsConfig

import numpy as np

from ..logging import logger

# ponytail: ska_ost_array_config is an SKA-internal package (antenna-layout
# templates, not on public PyPI) that upstream uses here just to look up
# antenna names per subarray template. It's a heavy, unrelated dependency to
# pull into a "framework-free" vendored copy, so we degrade gracefully (skip
# the check, logging a warning) when it isn't installed, rather than making
# every validate() call hard-require it. Install ska_ost_array_config
# alongside this vendored package to get full subarray-overlap/template
# checking.
try:
    from ska_ost_array_config import get_subarray_template
except ImportError:  # pragma: no cover
    get_subarray_template = None


def subarrays_do_not_overlap(obs_config: ObsConfig) -> bool:
    """Check subarray templates are legit.

    Args:
        obs_config (ObsConfig): The observation configuration containing subarray details.

    Returns:
        bool: Pass (True) or fail (False).
    """
    if get_subarray_template is None:
        logger.warning("ska_ost_array_config not installed - skipping subarray overlap check.")
        return True

    names = []
    count = 0
    for sa in obs_config.config.subarrays:
        if sa.template is None:
            return False
        sa_tpl = list(get_subarray_template(sa.template).array_config.names.values)
        names += sa_tpl
        count += len(sa_tpl)
    try:
        assert len(np.unique(names)) == count
        return True
    except AssertionError:
        return False


def subarray_templates_are_valid(obs_config: ObsConfig) -> bool:
    """Check subarray templates are in ska-ost-array-config.

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    if get_subarray_template is None:
        logger.warning("ska_ost_array_config not installed - skipping subarray template check.")
        return True

    for sa in obs_config.config.subarrays:
        if sa.template is None:
            return False
        try:
            get_subarray_template(sa.template)
        except AssertionError:
            logger.error(f"Subarray template {sa.template} is not valid.")
            return False
    return True
