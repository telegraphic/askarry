# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/frontend_utils.py).
"""frontend_utils.py - Utility functions for ska-sci-ops-setup-validator-frontend."""

from copy import deepcopy

import simplejson

from .data_products import DetectedFilterbank, FlowthroughArchive
from .logging import logger
from .obs_config import (
    BeamConfig,
    ContinuumConfig,
    ObsConfig,
    SubArrayConfig,
    ZoomConfig,
    create_beam_config,
    create_obs_config,
    create_subarray_config,
)
from .odp_config import create_odp
from .unit_utils import to_quantity
from .validation_schema import get_allowed_values
from .validators import ConfigValidator

###########################
## load config from JSON ##
###########################


def load_frontend_config(config_dict: dict = None, json_filepath: str = None, json_str: str = None) -> list[ObsConfig]:
    """Generate an ObsConfig by parsing JSON (or from pre-parsed config dict).

    Args:
        config_dict (dict): Create from a preloaded configuration dictionary.
        json_str (str): Create from a JSON string.
        json_filepath (str): Load JSON from a given filename.

    Returns:
        obs_config (ObsConfig): The ObsConfig object
    """
    if config_dict is not None:
        config = config_dict
    elif json_filepath:
        with open(json_filepath, "r") as fh:
            config = simplejson.load(fh)
    elif json_str:
        config = simplejson.loads(json_str)
    else:
        raise RuntimeError("Need to set config_dict, json_filepath, or json_str")

    obs_config = create_obs_config(context=config["context"], telescope=config["telescopeType"])
    subarrays = load_frontend_subarray_config(config["subarrays"], obs_config=obs_config)
    obs_config.config.subarrays = subarrays

    return obs_config


def load_frontend_subarray_config(sa_config: dict, obs_config: ObsConfig) -> list[SubArrayConfig]:
    """Load a dict of SubarrayConfig objects by parsing JSON.

    Args:
        sa_config (dict): Dictionary with configuration details.
        obs_config (ObsConfig): The parent ObsConfig (can be None if not set)

    Returns:
        subarrays (list): A list of SubArrayConfig objects
    """
    subarrays = []

    for sa_id, d in sa_config.items():
        # WAR: SKA-Low JSON does not include 'band', so we force it here to SKALA.
        if obs_config.telescope == "SKA-Low":
            d["band"] = "SKALA"
        sa = create_subarray_config(id=sa_id, template=d["template"], band=d["band"], obs_config=obs_config)
        beams = load_frontend_beam_config(d, obs_config=obs_config, subarray_config=sa)
        sa.beams = beams
        subarrays.append(sa)

    return subarrays


def load_frontend_beam_config(
    beam_config: dict, obs_config: ObsConfig, subarray_config: SubArrayConfig
) -> dict[BeamConfig]:
    """Load a dict of PrimaryBeam objects by parsing JSON.

    Args:
        beam_config (dict): Dictionary with beam configuration details.
        obs_config (ObsConfig): The parent ObsConfig (can be None if not set)
        subarray_config (SubArrayConfig): The parent SubArrayConfig (can be None if not set)

    Returns:
        beams (dict): A dictionary of SubArrayConfig objects
    """
    beams = []

    # Update modes with settings from JSON
    for beam_id in beam_config["modes"].keys():
        b = create_beam_config(beam_id, subarray_config, obs_config)
        modes = {"continuum": b.continuum, "zoom": b.zoom, "pss": b.pss, "pst": b.pst}
        odp_list = []

        for mode_id, mode_config in modes.items():
            # Copy over parsed value into param
            settings = beam_config[f"{mode_id}Settings"][beam_id]
            if settings["enabled"]:
                for k, v in settings.items():
                    # if "unit" in settings:
                    #    v = to_quantity(v, settings["unit"]).to_string()
                    if k == "odpList" or k == "beamsList":
                        if v:
                            try:
                                _odps, errs = load_frontend_odp_config(
                                    v, obs_config, subarray_config, mode_config=mode_config
                                )
                            except ValueError as e:
                                logger.debug(f"{e}")
                                b.load_errs.append(f"{mode_id}: {e}")
                            if len(_odps) > 0:
                                odp_list += _odps
                            if len(errs) > 0:
                                for e in errs:
                                    b.load_errs.append(f"{mode_id}: {e}")

                    try:
                        setattr(modes[mode_id], k, v)
                    except ValueError as e:
                        logger.warning(f"Failed to set {k} to {v} for {mode_id} in beam {beam_id}")
                        b.load_errs.append(f"{mode_id}: {e}")
        b.odps = odp_list
        beams.append(b)

    return beams


def load_frontend_odp_config(
    odp_list: list, obs_config: ObsConfig, subarray_config: SubArrayConfig, mode_config: ConfigValidator = None
) -> tuple[list, list]:
    """Load a dict of ODP items by parsing JSON.

    Args:
        odp_list (list): Dictionary with ODP configuration details.
        obs_config (ObsConfig): The parent ObsConfig
        subarray_config (SubArrayConfig): The parent SubArrayConfig
        mode_config (ConfigValidator): One of ContinuumConfig, ZoomConfig, PssConfig or PstConfig

    Returns:
        odps (list): A list of OdpItem objects
        errs (list): A list of errors encountered.
    """
    odps = []
    errs = []

    PST_ODP_TYPES = {"pulsar_timing", "detected_filterbank", "flowthrough_archive"}

    for odp_item in odp_list:
        logger.debug(f"Loading ODP item: {odp_item}")
        if odp_item["type"] in PST_ODP_TYPES:
            try:
                allowed = get_allowed_values(
                    "pst_settings", "availableOdpTypes", context=obs_config.context, telescope=obs_config.telescope
                )
                normalized_allowed = {v.lower().replace(" ", "_") for v in allowed}
                if odp_item["type"] not in normalized_allowed:
                    errs.append(
                        f"PST ODP type '{odp_item['type']}' is not one of the allowed types: {sorted(normalized_allowed)}"
                    )
                    continue
            except (KeyError, TypeError):
                pass  # availableOdpTypes not defined in schema for this context/telescope
        if odp_item["type"] == "calibrated_visibilities":
            odp_settings = deepcopy(odp_item["settings"])
            odp_settings["n_time_averaging"] = odp_settings.pop("timeAveraging")
            odp_settings["n_frequency_averaging"] = odp_settings.pop("frequencyAveraging")
            try:
                odp = create_odp(
                    odp_type="calibrated_visibilities",
                    id=odp_item["label"],
                    obs_config=obs_config,
                    subarray_config=subarray_config,
                    odp_config=odp_settings,
                )
                odps.append(odp)
            except ValueError as e:
                errs.append(f"{odp_item['label']}: {e}")
        elif odp_item["type"] == "pulsar_timing":
            odp_settings = deepcopy(odp_item)
            odp_settings.pop("type")
            try:
                odp = create_odp(
                    odp_type="pst_timing",
                    id="Timing",
                    obs_config=obs_config,
                    subarray_config=subarray_config,
                    odp_config=odp_item,
                )
                odps.append(odp)
            except ValueError as e:
                errs.append(f"PST timing beam: {e}")
        elif odp_item["type"] == "detected_filterbank":
            missing = [f for f in ("polarizations", "bitDepth") if not odp_item.get(f)]
            if missing:
                errs.append(f"PST detected filterbank beam: required field(s) not set: {', '.join(missing)}")
                continue

            odp_config = deepcopy(odp_item)
            odp_config.pop("type", None)
            # Drop None values coming from optional Pydantic fields that are not set
            odp_config = {k: v for k, v in odp_config.items() if v is not None}

            # Normalise nested {value, unit} dicts to plain floats in schema units
            if isinstance(odp_config.get("timeResolution"), dict):
                tr = odp_config["timeResolution"]
                odp_config["timeResolution"] = to_quantity(tr["value"], tr["unit"]).to("s").value
            if isinstance(odp_config.get("frequencyResolution"), dict):
                fr = odp_config["frequencyResolution"]
                odp_config["frequencyResolution"] = to_quantity(fr["value"], fr["unit"]).to("Hz").value

            # Compute data rate and inject for schema validation
            if "timeResolution" in odp_config and "frequencyResolution" in odp_config:
                try:
                    _dp = DetectedFilterbank.from_pst_beam(odp_item)
                    computed_rate = _dp.data_rate.to("Gbyte/s")
                    odp_config["data_rate"] = computed_rate.value
                    is_valid, computed, max_rate = _dp.check_data_rate(obs_config.context, obs_config.telescope)
                    if not is_valid:
                        errs.append(
                            f"PST detected filterbank beam: data rate "
                            f"({computed.value:.4g} GB/s) exceeds limit — "
                            f"must be at most {max_rate.value:.4g} GB/s"
                        )
                        continue
                except Exception as e:
                    logger.debug(f"Could not compute data_rate for detected_filterbank: {e}")

            try:
                odp = create_odp(
                    odp_type="pst_dynamic_spectrum",
                    id="Dynamic Spectrum",
                    obs_config=obs_config,
                    subarray_config=subarray_config,
                    odp_config=odp_config,
                )
                odps.append(odp)
            except ValueError as e:
                errs.append(f"PST detected filterbank beam: {e}")
        elif odp_item["type"] == "flowthrough_archive":
            missing = [f for f in ("polarizations", "bitDepth") if not odp_item.get(f)]
            if missing:
                errs.append(f"PST flowthrough beam: required field(s) not set: {', '.join(missing)}")
                continue

            try:
                _dp = FlowthroughArchive.from_pst_beam(odp_item)
                is_valid, computed, max_rate = _dp.check_data_rate(obs_config.context, obs_config.telescope)
                if not is_valid:
                    errs.append(
                        f"PST flowthrough beam: data rate "
                        f"({computed.value:.4g} GB/s) exceeds limit — "
                        f"must be at most {max_rate.value:.4g} GB/s"
                    )
                    continue
            except Exception as e:
                logger.debug(f"Could not compute data_rate for flowthrough_archive: {e}")

            odp_settings = deepcopy(odp_item)
            odp_settings.pop("type")
            try:
                odp = create_odp(
                    odp_type="pst_flowthrough",
                    id="Flowthrough",
                    obs_config=obs_config,
                    subarray_config=subarray_config,
                    odp_config=odp_item,
                )
                odps.append(odp)
            except ValueError as e:
                errs.append(f"PST flowthrough beam: {e}")
        elif odp_item["type"] in ("images", "continuum_image", "spectral_image"):
            if odp_item["settings"].get("makeMfsStokesI"):
                odp_settings = deepcopy(odp_item["settings"]["mfsStokesI"])
                if isinstance(mode_config, ZoomConfig):
                    odp_settings["mode"] = "zoom"
                else:
                    odp_settings["mode"] = "continuum"
                # Add additional image types
                for imgt in ("psf", "residual", "model"):
                    odp_settings[f"create_{imgt}_image"] = True if imgt in odp_settings["additionalImages"] else False
                try:
                    odp = create_odp(
                        odp_type="mfs_image",
                        id=odp_item["label"],
                        obs_config=obs_config,
                        subarray_config=subarray_config,
                        odp_config=odp_settings,
                    )
                    odps.append(odp)
                except ValueError as e:
                    errs.append(f"{odp_item['label']}: {e}")

            if odp_item["settings"].get("makeChannelImages"):
                odp_settings = deepcopy(odp_item["settings"]["channelImages"])
                if isinstance(mode_config, ZoomConfig):
                    odp_settings["n_channel_zoom"] = odp_settings.pop("channelCount")
                    odp_settings["n_polarization_zoom"] = len(odp_settings.pop("polarizations"))
                    odp_settings["mode"] = "zoom"

                elif isinstance(mode_config, ContinuumConfig) or mode_config is None:
                    if odp_item["type"] == "spectral_image":
                        odp_settings["n_channel_spectral"] = odp_settings.pop("channelCount")
                    else:
                        odp_settings["n_channel"] = odp_settings.pop("channelCount")

                    n_pol = len(odp_settings.pop("polarizations"))
                    if n_pol == 0:
                        errs.append("Select at least one polarization for imaging.")
                        odp_settings["n_polarization"] = 1
                    else:
                        odp_settings["n_polarization"] = n_pol
                    odp_settings["mode"] = "continuum"

                # Add additional image types
                for imgt in ("psf", "residual", "model"):
                    odp_settings[f"create_{imgt}_image"] = True if imgt in odp_settings["additionalImages"] else False

                odp_settings.pop("additionalImages")

                try:
                    odp = create_odp(
                        odp_type="image_cube",
                        id=odp_item["label"],
                        obs_config=obs_config,
                        subarray_config=subarray_config,
                        odp_config=odp_settings,
                    )
                    odps.append(odp)
                except ValueError as e:
                    errs.append(f"{odp_item['label']}: {e}")
        else:
            logger.warning(f"Unknown ODP type: {odp_item['type']}")
            errs.append(f"Unknown ODP type: {odp_item['type']} ({odp_item.get('label', '<unknown>')})")

    return odps, errs


def validate(obs_config_dict: dict) -> dict:
    """Function to validate the telescope configuration received by the API.

    Args:
        obs_config_dict (dict):  Dictionary containing the telescope configuration.

    Returns:
    dict: Dictionary containing the validation results.
    """
    response = {}
    try:
        obs = load_frontend_config(config_dict=obs_config_dict)
        alerts = obs.validate(raise_error=False)

    except ValueError as err:
        response["status"] = "error"
        response["message"] = "Telescope configuration validation failed."
        response["alerts"] = [str(err)]
        return response
    finally:
        logger.debug("sending", response)

    if alerts:
        response["status"] = "error"
        response["message"] = "Telescope configuration validation failed."
        response["alerts"] = alerts
        return response
    else:
        response["status"] = "success"
        response["message"] = "Telescope configuration validated successfully."
    return response
