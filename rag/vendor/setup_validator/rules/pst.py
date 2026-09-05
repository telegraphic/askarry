# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/pst.py).
"""rules/pst.py - Rules for PST mode."""

import numpy as np

from ..data_products import FlowthroughArchive
from ..obs_config import ObsConfig
from ..unit_utils import to_quantity
from ..validation_schema import get_minmax


def check_sum_pst_beams_is_valid(obs_config: ObsConfig) -> bool:
    """Check SKA PST beams are in range (across subarrays).

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    _n_min, n_max, _n_step = get_minmax(
        "odps/odp_config", "n_pst_odps", telescope=obs_config.telescope, context=obs_config.context
    )

    n_beam = 0
    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            if b.pst.enabled:
                for o in b.odps:
                    if o.type.startswith("pst"):
                        n_beam += 1

    if n_beam <= n_max:
        return True
    else:
        return False, {"n_beam": n_beam, "n_max": n_max}


def check_pst_odp_compatible_with_continuum(obs_config) -> bool:
    """Check continuum and pst mode configs are valid.

    ODPs must have bandwidths less than or equal to the max continuum mode bandwidth.

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    err_msg = ""
    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            _bw_min, bw_max, _n_step = get_minmax(
                "continuum_settings",
                "bandwidth",
                telescope=obs_config.telescope,
                context=obs_config.context,
                runtime_params=obs_config.config.subarrays[0].rx_runtime_params,
            )

            if b.continuum.enabled:
                max_freq_cont = b.continuum.centralFrequency + b.continuum.bandwidth / 2
                min_freq_cont = b.continuum.centralFrequency - b.continuum.bandwidth / 2
            else:
                break

            if b.pst.enabled:
                for o in b.odps:
                    if o.type.startswith("pst"):
                        max_freq_pst = o.settings.centralFrequency + o.settings.bandwidth / 2
                        min_freq_pst = o.settings.centralFrequency - o.settings.bandwidth / 2
                        req_bw = np.max((max_freq_cont, max_freq_pst)) - np.min((min_freq_cont, min_freq_pst))
                        if req_bw > to_quantity(bw_max).value:
                            err_msg += f"beam {b.id} ODP {o.id} \n"
    if err_msg:
        return False, {"errs": err_msg}
    else:
        return True


def check_pst_flowthrough_data_rate(obs_config: ObsConfig) -> bool:
    """Check the computed data rate for PST flowthrough ODPs is within schema bounds.

    Constructs a FlowthroughArchive data product from each flowthrough ODP's settings
    and compares the computed data_rate against the min/max defined in the schema.

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False, dict) with details.
    """
    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            if not b.pst.enabled:
                continue
            for o in b.odps:
                if o.type != "pst_flowthrough":
                    continue
                s = o.settings
                bw_hz = to_quantity(s.bandwidth, "MHz").to("Hz").value
                polarizations = getattr(s, "polarizations", None)
                if isinstance(polarizations, list):
                    n_pol = len(polarizations)
                else:
                    n_pol = int(getattr(s, "n_polarization", 1))
                n_bit = int(getattr(s, "bitDepth", getattr(s, "n_bit", 4)))

                product = FlowthroughArchive(bw=bw_hz, n_pol=n_pol, n_bit=n_bit, n_beam=1, osamp=1)
                try:
                    is_valid, computed_dr, max_dr = product.check_data_rate(obs_config.context, obs_config.telescope)
                except (FileNotFoundError, KeyError):
                    continue

                if not is_valid:
                    return False, {
                        "beam": b.id,
                        "odp": o.id,
                        "data_rate": f"{computed_dr:.4f}",
                        "dr_max": f"{max_dr:.4f}",
                    }
    return True
