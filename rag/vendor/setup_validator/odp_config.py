# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/odp_config.py).
"""obs_config.py - Define data model for observation/telescope configuration.

Uses the Parameterized dataclass from the param package.
Do not use classes directly - use create_odp() to instantiate a new ODP object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import param as p

if TYPE_CHECKING:  # pragma: no cover
    from .obs_config import ObsConfig, SubArrayConfig

from .validation_schema import (
    attach_params_from_schema,
    load_validation_schema,
)
from .validators import ConfigValidator


class CalibratedVisibilities(ConfigValidator):
    """Calibrated visibilities ODP settings."""

    def __init__(self, **params):
        super().__init__(**params)


class MfsImage(ConfigValidator):
    """MFS Stokes I settings."""

    def __init__(self, **params):
        super().__init__(**params)


class ImageCube(ConfigValidator):
    """Channel images settings."""

    def __init__(self, **params):
        super().__init__(**params)


class PstTiming(ConfigValidator):
    """PST timing settings."""

    def __init__(self, **params):
        super().__init__(**params)


class PstDynamicSpectrum(ConfigValidator):
    """PST dynamic spectrum settings."""

    def __init__(self, **params):
        super().__init__(**params)


class PstFlowthrough(ConfigValidator):
    """PST flowthrough settings."""

    def __init__(self, **params):
        super().__init__(**params)


ODP_CLASSES = (CalibratedVisibilities, MfsImage, ImageCube, PstTiming, PstDynamicSpectrum, PstFlowthrough)


class ODPValidator(ConfigValidator):
    """Base ODP ConfigValidator class."""

    id = p.String(doc="ODP name / identifier", allow_None=True)
    type = p.String(doc="ODP data type", allow_None=True)
    settings = p.ClassSelector(doc="Telescope config", class_=ODP_CLASSES, allow_None=True, precedence=-1)

    def validate(self, raise_error: bool = True) -> list[str]:
        """Apply validation to settings."""
        return self.settings.validate(raise_error=True)


def create_odp(
    odp_type: str, id: str, obs_config: ObsConfig, subarray_config: SubArrayConfig, odp_config: dict = None
) -> ConfigValidator:
    """Create a calibrated visibilities config from the odp_config.

    Args:
        odp_type (str): Type of ODP to create. One of "calibrated_visibilities", "image_cube", or "mfs_image".
        id (str): Name / identifier for the ODP.
        obs_config (ObsConfig): Observation configuration object.
        subarray_config (SubArrayConfig): Subarray configuration object.
        odp_config (dict, optional): ODP configuration dictionary for instantiation.

    Returns:
        odp (ConfigValidator): An instance of the specified ODP type.
    """
    # fmt: off
    odp_type_dict = {
        "calibrated_visibilities": CalibratedVisibilities,
        "image_cube": ImageCube,
        "mfs_image": MfsImage,
        "pst_timing": PstTiming,
        "pst_dynamic_spectrum": PstDynamicSpectrum,
        "pst_flowthrough": PstFlowthrough
    }
    odp_settings = odp_type_dict[odp_type]()

    odp_settings._schema = load_validation_schema(f"odps/{odp_type}", context=obs_config.context, telescope=obs_config.telescope)

    # Need to get receiver runtime params from the subarray config
    odp_settings.runtime_params = {}
    odp_settings.runtime_params.update(subarray_config.rx_runtime_params)

    odp_settings = attach_params_from_schema(odp_settings, odp_settings.runtime_params)

    if odp_config:
        for key, value in odp_config.items():
            setattr(odp_settings, key, value)

    odp = ODPValidator(id=id, type=odp_type, settings=odp_settings)
    return odp


def odp_config_to_dict(odp: ConfigValidator) -> dict:
    """Convert ODP ConfigValidator object to a a JSON-serializable dict.

    Args:
        odp (ConfigValidator): ODP to convert to JSON-serializable dict.

    Returns:
        jdict (dict): A JSON-serializable dict.

    Notes:
        The param package does not support JSON serialization of
        nested classes, so we handle this manually.
    """
    jdict = {}
    for k, v in odp.param.values().items():
        if isinstance(v, ConfigValidator):
            jdict[k] = v.param.values()
            jdict[k].pop("name")
        else:
            jdict[k] = v
    jdict.pop("name")
    return jdict


def jdict_to_odp_config(jdict: dict, obs_config: ObsConfig, subarray_config: SubArrayConfig) -> ConfigValidator:
    """Convert dict-formatted data to an ODP config.

    Args:
        jdict (dict): Dictionary containing ODP configuration.
        obs_config (ObsConfig): Observation configuration object.
        subarray_config (SubArrayConfig): Subarray configuration object.

    Returns:
        odp_config (ConfigValidator): An instance of the specified ODP type.
    """
    odp_config = create_odp(
        jdict["type"], jdict["id"], obs_config=obs_config, subarray_config=subarray_config, odp_config=jdict["settings"]
    )
    return odp_config
