# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/validation_schema.py).
"""validation_schema.py - load validation schema from JSON."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import param as p
import simplejson
import yaml

from .unit_utils import to_quantity

if TYPE_CHECKING:  # pragma: no cover
    from .validators import ConfigValidator


def merge_dicts(d1: dict, d2: dict) -> dict:
    """Update first dict with second recursively.

    Args:
        d1 (dict): First dictionary to update.
        d2 (dict): Second dictionary to merge into first.

    Returns:
        d1 (dict): Updated first dictionary.
    """
    for k, v in d1.items():
        if k in d2:
            # If the value is a dictionary, merge it recursively
            if isinstance(v, dict):
                d2[k] = merge_dicts(v, d2[k])
    d1.update(d2)
    return d1


def load_yaml_or_json(filepath: str) -> dict:
    """Load a YAML/JSON file, trying common extensions if none found.

    If ``filepath`` exists, it is opened directly. Otherwise, this will try
    appending ".yaml", ".yml", then ".json" in that order.
    """
    candidates = []
    if os.path.exists(filepath):
        candidates.append(filepath)
    else:
        for ext in (".yaml", ".yml", ".json"):
            candidates.append(f"{filepath}{ext}")

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        with open(candidate, "r") as fh:
            try:
                return yaml.safe_load(fh)
            except Exception:
                fh.seek(0)
                return simplejson.load(fh)
    raise FileNotFoundError(f"Could not locate file for base path: {filepath}")


def _resolve_schema_filepath(
    base_dir: str, id: str, context: str | None = None, telescope: str | None = None
) -> tuple[str | None, str | None]:
    """Resolve base paths (without extension) for base and context-specific files.

    Returns a tuple of (base_schema_basepath, context_schema_basepath). Either may be None.
    """
    if context:
        base_schema_path = f"{base_dir}/schema/base/{telescope}/{id}"
        context_schema_path = f"{base_dir}/schema/{context}/{telescope}/{id}"
    else:
        base_schema_path = f"{base_dir}/schema/{id}"
        context_schema_path = None
    return base_schema_path, context_schema_path


def load_validation_schema(id: str = "obs_config", context: str = None, telescope: str = None) -> dict:
    """Load validation schema from stored YAML/JSON configs.

    Args:
        id (str): Name of validation schema to load.
        context (str): Specific context for validation schema to load.
                  One of 'SV-AA2', 'AA*', 'obs_config'
        telescope (str): Name of telescope, SKA-Mid or SKA-Low

    Returns:
        tpl (dict): Schema template to load.
    """
    fp = os.path.dirname(os.path.realpath(__file__))
    id = id.lower().replace("-", "_")

    if context:
        context = context.lower().replace("-", "_").replace(" ", "_").replace("*", "star")
        telescope = telescope.lower().replace("-", "_").replace(" ", "_").replace("*", "star")
        base_basepath, ctx_basepath = _resolve_schema_filepath(fp, id, context=context, telescope=telescope)
        tpl = load_yaml_or_json(base_basepath)
        try:
            ctx_tpl = load_yaml_or_json(ctx_basepath)
            tpl = merge_dicts(tpl, ctx_tpl)
        except FileNotFoundError:
            pass

    else:
        base_basepath, _ = _resolve_schema_filepath(fp, id)
        tpl = load_yaml_or_json(base_basepath)
    return tpl


def get_allowed_values(schema: str, param_id: str, context: str = None, telescope: str = None) -> list[str]:
    """Return list of allowed values.

    Args:
        schema (str): Name of schema JSON to search, e.g. 'obs_config'
        param_id (str): Name of parameter.
        context (str or None): Name of specific context to load schema from, e.g. SV-AA2
        telescope (str or None): Name of telescope (SKA-Mid or SKA-Low)
    """
    schema = load_validation_schema(schema, context=context, telescope=telescope)
    return schema["params"][param_id][1]["allowed_values"]


def get_minmax(
    schema: str, param_id: str, context: str = None, telescope: str = None, runtime_params=None
) -> list[float]:
    """Return min, max and step for a minmax parameter.

    Args:
        schema (str): Name of schema JSON to search, e.g. 'obs_config'
        param_id (str): Name of parameter.
        context (str or None): Name of specific context to load schema from, e.g. SV-AA2
        telescope (str or None): Name of telescope (SKA-Mid or SKA-Low)
        runtime_params (dict or None): Runtime parameters to substitute into the schema.
    """
    schema = load_validation_schema(schema, context=context, telescope=telescope)
    val_type, mm = schema["params"][param_id]
    if val_type == "equals":
        return mm["ref_value"], mm["ref_value"], None
    else:
        mm = schema["params"][param_id][1]

    if runtime_params:
        runtime_replace(mm, runtime_params)
    if val_type == "minmax2":
        max2 = to_quantity(mm.get("max2", mm["max"]))
        min2 = to_quantity(mm.get("min2", mm["min"]))
        mm["min"] = np.max([to_quantity(mm["min"]).value, min2.value])
        mm["max"] = np.min([to_quantity(mm["max"]).value, max2.value])
    return mm["min"], mm["max"], mm.get("step", None)


def get_rules(schema: str, context: str = None, telescope: str = None) -> dict:
    """Return a list of rules within the schema.

    Args:
        schema (str): Name of schema JSON to search, e.g. 'obs_config'
        param_id (str): Name of parameter.
        context (str or None): Name of specific context to load schema from, e.g. SV-AA2
        telescope (str or None): Name of telescope (SKA-Mid or SKA-Low)

    Returns:
        rules (dict): A dictionary of all rules within the schema.
    """
    RULE_DICT = load_validation_schema("validation_rules")
    schema = load_validation_schema(schema, context=context, telescope=telescope)
    rules = {}
    for rule in schema.get("rules", {}):
        rules[rule] = RULE_DICT[rule]
    return rules


def runtime_replace(vkws: dict, runtime_params: dict) -> dict:
    """Replace parameters in a dictionary with a lookup table.

    Use to replace template variables stored in a dictionary
    that are unknown until runtime.

    Example:
        ```
        a = {'a': '$TODO', 'b': 7, 'c': '$CAT_NAME'}
        runtime_params = {'$TODO': 99, '$CAT_NAME': 'Tom'}
        a = _replace(a, runtime_params)
        # a = {'a': 99, 'b', 7, 'c': 'Tom'}
        ```

    Notes:
        This does not run recursively.

    Args:
        vkws (dict): Parameters to update.
        runtime_params (dict): Lookup table of replacement values

    Returns:
        vkws (dict): Updated dictionary

    """
    _vkws = deepcopy(vkws)
    for k, v in _vkws.items():
        if isinstance(v, list):
            for idx in range(len(v)):
                v[idx] = runtime_params.get(v[idx], v[idx])
            vkws[k] = v
        else:
            vkws[k] = runtime_params.get(v, v)
    return vkws


##################
## schema utils ##
##################


def param_from_schema(val_type: str, val_kws: dict, doc: str = None) -> p.Parameter:
    """Create a param from its schema dict entry.

    Creates a param based on the validation function used.

    Args:
        val_type (str): Validator method string (e.g. 'oneof', 'minmax')
        val_kws (dict): Dictionary of keywords for validator method.
        doc (str): Docstring for param.

    Returns:
        param (p.Parameter): Parameter with best subclass to match val_type
    """
    if val_type == "oneof":
        _param = p.Selector(objects=val_kws["allowed_values"], doc=doc)
    elif val_type in ("minmax", "minmax2"):
        unit = val_kws.get("unit", None)
        min = to_quantity(val_kws["min"], unit=unit).value
        max = to_quantity(val_kws["max"], unit=unit).value
        step = val_kws.get("step", None)
        if step:
            step = to_quantity(val_kws["max"], unit=unit).value
        if val_type == "minmax2":
            min2 = to_quantity(val_kws.get("min2", min), unit=unit).value
            max2 = to_quantity(val_kws.get("max2", max), unit=unit).value
            min = np.max([min, min2])
            max = np.min([max, max2])
        default = (max - min) / 2 + min if val_kws.get("step", None) is None else min
        _param = p.Number(bounds=(min, max), default=default, step=step, doc=doc)
    elif val_type == "equals":
        if "unit" in val_kws:
            _param = p.Selector(objects=[to_quantity(val_kws["ref_value"], val_kws["unit"]).value], doc=doc)
        else:
            _param = p.Selector(objects=[val_kws["ref_value"]], constant=True, doc=doc)
    return _param


def attach_params_from_schema(mode_config: ConfigValidator, runtime_params: dict) -> ConfigValidator:
    """Read the schema for a config and create params.

    Args:
        mode_config (ConfigValidator): config to attach params to by reading its _schema.
        runtime_params (dict): Runtime parameters to sub in during reading of schema.

    Returns:
        mode_config (ConfigValidator): Configuration with params attached.
    """
    mode_schema = mode_config._schema

    # Go through the mode schema and create params accordingly
    for param_id, (val_type, val_kws) in mode_schema["params"].items():
        # Get runtime parameters, and update any variables
        rp = deepcopy(runtime_params)
        val_kws = runtime_replace(val_kws, rp)

        # Update variables in schema
        mode_schema["params"][param_id] = (val_type, val_kws)

        # Get docstring if set
        doc = None
        if "docs" in mode_schema:
            doc = mode_schema["docs"].get(param_id, "")

        # create a param based on its schema entry
        if param_id.lower() == "enabled":
            # TODO: Get readonly/constant working with code that applies setattr(attr, kk, vv)
            # enbl = True if True in val_kws["allowed_values"] else False
            # _param  = p.Boolean(doc="Enable mode", constant=not(enbl))
            _param = p.Boolean(doc="Enable mode")
        else:
            _param = param_from_schema(val_type, val_kws, doc=doc)
        mode_config.param.add_parameter(param_id, _param)
        try:
            mode_config.runtime_params.update(rp)
        except (TypeError, AttributeError):
            mode_config.runtime_params = rp
    return mode_config
