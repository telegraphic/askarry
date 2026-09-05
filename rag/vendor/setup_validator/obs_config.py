# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/obs_config.py).
"""obs_config.py - Define data model for observation/telescope configuration.

Uses the Parameterized dataclass from the param package.
The primary method is load_obs_config().

Defines:
    ObsConfig - Top level config, has subarray config children
    ObsConfigContainer - Container for subarray configs
    SubArrayConfig - Subarray configuration, has BeamConfig children
    BeamConfig - Primary beam mode configuration.

    The hierarchical dict structure is generated from the JSON validation schema in ./schema

    ObsConfig (created with load_obs_config)
        - ObsConfigContainer (created with create_obs_config_container)
            - SubarrayConfig (load_subarray_config called by load_obs_config)
                - BeamConfig (load_beam_config called by load_subarray_config)
                ...
            - SubarrayConfig
                -BeamConfig
                ...
        ...
"""

from copy import deepcopy

import param as p
import simplejson

from .logging import logger
from .odp_config import jdict_to_odp_config, odp_config_to_dict
from .telescope.receivers import receiver_dict
from .validation_schema import (
    attach_params_from_schema,
    get_allowed_values,
    get_minmax,
    get_rules,
    load_validation_schema,
)
from .validators import ConfigValidator, _raise_or_return, apply_rules

OBS_SCHEMA = load_validation_schema("obs_config")
RULE_DICT = load_validation_schema("validation_rules")


###################
## param classes ##
###################


class SubArrayConfig(ConfigValidator):
    """SubArray configuration."""

    # fmt: off
    id         = p.String(doc='Subarray ID')
    template   = p.Selector(doc='Subarray template', allow_None=True)
    band       = p.Selector(doc='Observing band', allow_None=True)
    n_beam     = p.Integer(doc='Number of beams', allow_None=True, default=1)
    # fmt: on

    def __init__(self, **params):
        super().__init__(**params)

    def __repr__(self):
        return f"<SubArrayConfig: {self.template} ({self.band})>"

    @property
    def rx(self):
        """Return receiver information."""
        telescope = "SKA-Mid" if self.template.lower().startswith("mid") else "SKA-Low"
        return receiver_dict[telescope][self.band]

    @property
    def rx_runtime_params(self):
        """Return receiver info as runtime template replace dict."""
        rp = {"$rx_low": self.rx["low"], "$rx_high": self.rx["high"], "$rx_bw": self.rx["bw"]}
        return rp


class ObsConfigContainer(ConfigValidator):
    """Container to hold subarray configs."""

    # fmt: off
    n_subarray = p.Integer(doc="Number of subarrays", allow_None=True)
    subarrays  = p.List(doc="Instantiated subarrays", allow_None=True, item_type=SubArrayConfig, precedence=-1)
    # fmt: on


class ObsConfig(ConfigValidator):
    """Observation configuration."""

    # fmt: off
    context   = p.Selector(doc="Telescope context", objects=get_allowed_values('obs_config', 'context'))
    telescope = p.Selector(doc="Telscope name", objects=get_allowed_values('obs_config', 'telescope'))
    config    = p.ClassSelector(doc="Telescope config", class_=ObsConfigContainer, allow_None=True, precedence=-1)
    # fmt: on

    def __init__(self, **params):
        super().__init__(**params)
        self._schema = OBS_SCHEMA

    def __repr__(self):
        return f"<ObsConfig: {self.telescope} ({self.context}) >"

    def validate(self, raise_error: bool = True):
        """Run validation checks."""
        errmsg_list = []

        rules = get_rules("obs_config")
        errmsgs = apply_rules(rules, params={"obs_config": self}, raise_error=raise_error, raise_on_first=True)

        if errmsgs:
            errmsg_list += errmsgs

        if len(self.config.subarrays) == 0:
            errmsg_list.append(_raise_or_return("No Subarrays defined!", raise_error))

        # Run validate for all children subarrays + beams
        for sa in self.config.subarrays:
            errs = sa.validate(raise_error=False)
            if errs:
                errmsg_list += errs
            if len(sa.beams) == 0:
                msg = f"Subarray {sa.id} has no primary beam configuration."
                errmsg_list.append(_raise_or_return(msg, raise_error))
            for b in sa.beams:
                errs = b.validate(raise_error=raise_error)
                if errs:
                    for err in errs:
                        if isinstance(err, list):
                            for _err in err:
                                errmsg_list.append(
                                    _raise_or_return(f"subarray {sa.id}: beam {b.id}: {_err}", raise_error)
                                )
                        else:
                            errmsg_list.append(_raise_or_return(f"subarray {sa.id}: beam {b.id}: {err}", raise_error))
        return errmsg_list

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return obs_config_to_dict(self)

    def to_json(self) -> str:
        """Convert to JSON."""
        jdict = obs_config_to_dict(self)
        json_str = simplejson.dumps(jdict, indent=2)
        return json_str


class ContinuumConfig(ConfigValidator):
    """Base ContinuumConfig class."""

    def __init__(self, **params):
        super().__init__(**params)


class ZoomConfig(ConfigValidator):
    """Base ZoomConfig class."""

    def __init__(self, **params):
        super().__init__(**params)


class PssConfig(ConfigValidator):
    """Base PssConfig class."""

    def __init__(self, **params):
        super().__init__(**params)


class PstConfig(ConfigValidator):
    """Base PstConfig class."""

    def __init__(self, **params):
        super().__init__(**params)


class BeamConfig(ConfigValidator):
    """Primary beam configuration."""

    # fmt: off
    id        = p.String(doc="Beam identifier", constant=True)
    continuum = p.ClassSelector(doc="CONT mode setup dict (continuum)", class_=ContinuumConfig, allow_None=True)
    zoom      = p.ClassSelector(doc="ZOOM mode setup dict (zoom windows)", class_=ZoomConfig, allow_None=True)
    pss       = p.ClassSelector(doc="PSS mode setup dict (pulsar search)", class_=PssConfig, allow_None=True)
    pst       = p.ClassSelector(doc="PST mode setup dict (pulsar timing)", class_=PstConfig, allow_None=True)
    odps      = p.List(doc='Output data products', allow_None=True, item_type=ConfigValidator, precedence=-1)
    # fmt: on

    def __repr__(self):
        return f"<BeamConfig: {self.id}>"

    def validate(self, raise_error: bool = True):
        """Run validation checks."""
        ret = self.load_errs

        errmsg = self.continuum.validate(raise_error=raise_error)
        if errmsg:
            for _errmsg in errmsg:
                ret.append(f"continuum: {_errmsg}")

        errmsg = self.zoom.validate(raise_error=raise_error)
        if errmsg:
            for _errmsg in errmsg:
                ret.append(f"zoom: {_errmsg}")

        errmsg = self.pss.validate(raise_error=raise_error)
        if errmsg:
            for _errmsg in errmsg:
                ret.append(f"PSS: {_errmsg}")

        errmsg = self.pst.validate(raise_error=raise_error)
        if errmsg:
            for _errmsg in errmsg:
                ret.append(f"PST: {_errmsg}")

        if len(ret) > 0:
            return ret


#########################
## create_xx_config()  ##
#########################


def create_obs_config_container(obs_config: ObsConfig) -> ObsConfigContainer:
    """Create ObsConfigContainer object.

    This stores subarrays as a list (and potentially more metadata).
    Implements subarray checks (e.g. max subarrays for context).

    Args:
        n_subarray (int): Number of subarrays to generate
        obs_config (ObsConfig): Observation config object.

    Returns:
        obs_container (obsConfigContainer): A container for subarrays.
    """
    obs_container = ObsConfigContainer()
    obs_container._schema = load_validation_schema(
        "obs_config_container", telescope=obs_config.telescope, context=obs_config.context
    )
    obs_container.runtime_params = deepcopy(obs_config.runtime_params)
    mmin, mmax, _ = get_minmax(
        "obs_config_container", "n_subarray", context=obs_config.context, telescope=obs_config.telescope
    )

    obs_container.param.n_subarray.bounds = (mmin, mmax)
    obs_container.n_subarray = mmin
    return obs_container


def create_obs_config(telescope: str, context: str) -> ObsConfig:
    """Create ObsConfig object.

    Args:
        telescope (str): One of 'SKA-Low' or 'SKA-Mid'
        context (str): context to load, e.g. SV-AA2

    Returns:
        obs_config (ObsConfig): Observation configuration.
    """
    obs_config = ObsConfig(context=context, telescope=telescope)
    # Load validation schema
    obs_config._schema = load_validation_schema("obs_config")

    obs_config.runtime_params = {}
    obs_config.config = create_obs_config_container(obs_config)
    return obs_config


def create_subarray_config(id: str, template: str, band: str, obs_config: ObsConfig) -> SubArrayConfig:
    """Create SubArrayConfig object.

    Args:
        id (str): Identifier for subarray.
        template (str): Name of template to load.
        band (str): Receiver band (e.g. 'Band 1', 'Band 2', 'SKALA')
        obs_config (ObsConfig): observation config to use.

    Returns:
        sa (SubArrayConfig): Configuration object for subarray.
    """
    # fmt: off
    beams_param = p.List(doc='Primary beam mode setup', allow_None=True, item_type=BeamConfig, precedence=-1)
    band_avals  = get_allowed_values('subarray_config', 'band', obs_config.context, telescope=obs_config.telescope)
    band_param  = p.Selector(doc='Observing band', objects=band_avals, default=band_avals[0])
    tpl_avals   = get_allowed_values('subarray_config', 'template', context=obs_config.context, telescope=obs_config.telescope)
    nb_min, nb_max, nb_step = get_minmax('subarray_config', 'n_beam', context=obs_config.context, telescope=obs_config.telescope)
    # fmt: on

    sa = SubArrayConfig(id=id)
    sa._schema = load_validation_schema("subarray_config", context=obs_config.context, telescope=obs_config.telescope)
    sa.param.add_parameter("band", band_param)
    sa.param.add_parameter("beams", beams_param)
    sa.param.template.objects = tpl_avals
    sa.param.template.default = tpl_avals[0]

    sa.param.n_beam.bounds = (nb_min, nb_max)
    sa.param.n_beam = nb_min

    if template:
        sa.template = template
    else:
        sa.template = tpl_avals[0]

    if band:
        sa.band = band
    else:
        sa.band = band_avals[0]

    sa.runtime_params = deepcopy(obs_config.runtime_params)

    return sa


def create_beam_config(id: str, subarray_config: SubArrayConfig, obs_config: ObsConfig) -> BeamConfig:
    """Create BeamConfig object.

    Args:
        id (str): Identifier for beam.
        subarray_config (SubArrayConfig): subarray configuration to use.
        obs_config (ObsConfig): observation config to use.

    Returns:
        beam (BeamConfig): Beam configuration object.
    """
    beam_schema = load_validation_schema("beam_config", context=obs_config.context, telescope=obs_config.telescope)
    runtime_params = {}

    # Set runtime params
    runtime_params.update(subarray_config.runtime_params)
    runtime_params.update(subarray_config.rx_runtime_params)
    logger.debug(f"{runtime_params}")

    b = BeamConfig(id=id)
    b._schema = beam_schema
    b.runtime_params = runtime_params
    b.load_errs = []

    # Create ModeConfig classes to attach to beam object
    _mode_configs = []
    _classes = {"continuum": ContinuumConfig, "zoom": ZoomConfig, "pss": PssConfig, "pst": PstConfig}

    # Loop through modes, load their schema, and create ModeConfig classes
    for mode in _classes.keys():
        # Create mode_config with params loaded from schema
        mode_schema = load_validation_schema(
            f"{mode}_settings", context=obs_config.context, telescope=obs_config.telescope
        )
        mode_config = _classes[mode]()
        mode_config._schema = mode_schema
        mode_config = attach_params_from_schema(mode_config, b.runtime_params)

        b.param.add_parameter(mode, p.ClassSelector(doc=f"{mode} mode setup", class_=_classes[mode]))
        setattr(b, mode, mode_config)

    return b


######################################
## Config to JSON-serializable dict ##
#######################################


def beam_config_to_dict(beam: BeamConfig) -> dict:
    """Convert a BeamConfig to a a JSON-serializable dict.

    Args:
        beam (BeamConfig): BeamConfig to convert to JSON-serializable dict.

    Returns:
        jdict (dict): A JSON-serializable dict.

    Notes:
        The param package does not support JSON serialization of
        nested classes, so we handle this manually.
    """
    jdict = {}

    for k, v in beam.param.values().items():
        if isinstance(v, ConfigValidator):
            jdict[k] = v.param.values()
            jdict[k].pop("name")
        elif k == "odps":
            jdict[k] = []
            for vv in v:
                jdict[k].append(odp_config_to_dict(vv))
        else:
            jdict[k] = v
    jdict.pop("name")
    return jdict


def subarray_config_to_dict(sa: SubArrayConfig) -> dict:
    """Convert a SubarrayConfig to a a JSON-serializable dict.

    Args:
        sa (SubArrayConfig): SubArrayConfig to convert to JSON-serializable dict.

    Returns:
        jdict (dict): A JSON-serializable dict.
    """
    jdict = {}
    for k, v in sa.param.values().items():
        if k == "beams":
            jdict[k] = []
            for vv in v:
                jdict[k].append(beam_config_to_dict(vv))
        else:
            jdict[k] = v
    jdict.pop("name")
    return jdict


def obs_config_to_dict(obs: ObsConfig) -> dict:
    """Convert an ObsConfig to a JSON-serializable dict.

    Args:
        obs (ObsConfig): ObsConfig to convert to JSON-serializable dict.

    Returns:
        jdict (dict): A JSON-serializable dict.
    """
    jdict = {}
    for k, v in obs.param.values().items():
        # 'config' key is an ObsConfigContainer
        if k == "config":
            occ = {}
            # subarrays stored in 'subarrays'
            for kk, vv in v.param.values().items():
                if kk == "subarrays":
                    occ[kk] = []
                    for vvv in vv:
                        occ[kk].append(subarray_config_to_dict(vvv))
                else:
                    occ[kk] = vv
            occ.pop("name")
            jdict[k] = occ

        else:
            jdict[k] = v
    jdict.pop("name")
    return jdict


#################################
## Convert to ConfigValidators ##
#################################


def jdict_to_obs_config(jdict: dict) -> ObsConfig:
    """Convert dict-formatted data back to ObsConfig."""
    obs_config = create_obs_config(telescope=jdict["telescope"], context=jdict["context"])

    subarrays = []
    for sa_dict in jdict["config"]["subarrays"]:
        subarrays.append(jdict_to_subarray_config(sa_dict, obs_config))
    obs_config.config.subarrays = subarrays
    return obs_config


def jdict_to_subarray_config(jdict: dict, obs_config: ObsConfig) -> SubArrayConfig:
    """Convert dict-formatted data to SubArrayConfig.

    Args:
        jdict (dict): JSON-formatted data for subarray.
        obs_config (ObsConfig): Observation config to use.

    Returns:
        sa_config (SubArrayConfig): Subarray configuration.
    """
    sa_config = create_subarray_config(
        id=jdict["id"], template=jdict["template"], band=jdict["band"], obs_config=obs_config
    )

    beams = []
    for beam_dict in jdict["beams"]:
        b = jdict_to_beam_config(beam_dict, subarray_config=sa_config, obs_config=obs_config)
        beams.append(b)
    sa_config.beams = beams
    return sa_config


def jdict_to_beam_config(jdict: dict, subarray_config: SubArrayConfig, obs_config: ObsConfig) -> BeamConfig:
    """Convert dict-formatted data to BeamConfig.

    Args:
        jdict (dict): JSON-formatted data for beam.
        subarray_config (SubArrayConfig): Subarray configuration to use.
        obs_config (ObsConfig): Observation config to use.

    Returns:
        b (BeamConfig): Beam configuration.
    """
    b = create_beam_config(id=jdict["id"], subarray_config=subarray_config, obs_config=obs_config)

    # loop through keys
    for k, v in jdict.items():
        attr = getattr(b, k)
        # check if it is a mode setup dict (e.g. zoom)
        if k == "odps":
            odps = []
            for vv in v:
                odp = jdict_to_odp_config(vv, obs_config=obs_config, subarray_config=subarray_config)
                odps.append(odp)
            b.odps = odps
        if isinstance(v, dict):
            # loop through mode settings
            for kk, vv in v.items():
                setattr(attr, kk, vv)
        else:
            attr = v
    return b


def load_json_config(filename: str) -> ObsConfig:
    """Load a JSON-formatted config file.

    Args:
        filename (str): Path to JSON file.

    Returns:
        obs_config (ObsConfig): Observation configuration.
    """
    with open(filename, "r") as fh:
        jdict = simplejson.load(fh)
    obs_config = jdict_to_obs_config(jdict)
    return obs_config
