# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/validators.py).
"""validators.py - Main validation functions to check against rules."""

from copy import deepcopy
from importlib import import_module
from string import Template

import numexpr as ne
import numpy as np
import param as p

from .logging import logger
from .unit_utils import to_base_value, to_quantity, u
from .validation_schema import load_validation_schema

RULE_DICT = load_validation_schema("validation_rules")


class ValidationError(Exception):
    """Raise a ValidationError when valiation fails."""

    pass


##########################
## PARAMETER VALIDATION ##
##########################


def _replace_awkward_param_error_msgs(msg: str) -> str:
    """Reword param error messages that are awkward with more readable ones."""
    msg = msg.replace("enabled True must be one of [False]", "mode is not available in current context.")
    msg = msg.replace("Number parameter", "")
    return msg


def _raise_or_return(errmsg: str, raise_error: bool) -> str:
    """Raise a ValidationError, or return an error message string.

    Args:
        errmsg (str): Error message to return
        raise_error (bool): If True, raises a ValidationError, else returns a string.

    Returns:
        errmsg (str or None): Error message (returned only if raise_error=False)
    """
    errmsg = _replace_awkward_param_error_msgs(errmsg)
    if raise_error:
        raise ValidationError(errmsg)
    else:
        return errmsg


def validate_minmax(key, value, min=None, max=None, step=None, unit=None, raise_error: bool = True) -> str | None:
    """Validate a value lies between a min and max, with optional step.

    Notes:
        All values are converted into u.Quantity then compared.
        Raises ValidationError when validation fails.

    Args:
        key: key name.
        value: value to check is valid.
        min: minimum value (greater or equal to)
        max: maximum value (less or equal to)
        step: if quantized, step size between allowed values.
        unit (str or astropy Unit): Unit for comparisons.
        raise_error (bool): Raise a ValidationError on failure.

    Returns:
        errmsg (str): Error message string, or None if passed
    """
    value = to_quantity(value, unit)
    if step:
        try:
            n_step = (value / to_quantity(step, unit)).to("").value
            assert np.isclose(n_step - np.round(n_step), 0)
            logger.debug(f"PASS (step): {key} {value} is a multiple of {step}")
        except AssertionError:
            errmsg = f"{key}: {value} is invalid (must be multiple of {step})"
            logger.error(f"FAIL (step): {errmsg}")
            return _raise_or_return(errmsg, raise_error)

    try:
        assert value <= to_quantity(max, unit)
        assert value >= to_quantity(min, unit)
        logger.debug(f"PASS (minmax): {key} {value} is valid")
    except AssertionError:
        logger.debug(f"FAIL (minmax): {key} {value} is invalid")
        errmsg = f"{key} must be between {min} <= {value} <= {max}"
        return _raise_or_return(errmsg, raise_error)


def validate_minmax2(
    key, value, min=None, max=None, step=None, max2=None, min2=None, unit=None, raise_error: bool = True
) -> str | None:
    """Validate a value lies between a min and max, AND min2 and max2, with optional step.

    Notes:
        All values are converted into u.Quantity then compared.
        Raises ValidationError when validation fails.

    Args:
        key: key name.
        value: value to check is valid.
        min: minimum value (greater or equal to)
        min2: second minimum value to compare against
        max: maximum value (less or equal to)
        max2: second maximum value to compare against
        step: if quantized, step size between allowed values.
        unit (str or astropy Unit): Unit for comparisons.
        raise_error (bool): Raise a ValidationError on failure.
    """
    value = to_quantity(value, unit)
    try:
        validate_minmax(key, value, min, max, step)
        if max2:
            assert value <= to_quantity(max2, unit)
        if min2:
            assert value >= to_quantity(min2, unit)
        logger.debug(f"PASS (minmax2): {key} {value} is valid")
    except AssertionError:
        logger.error(f"FAIL (minmax2): {key} {value} is invalid")
        if raise_error:
            return _raise_or_return(
                f"{key} must satisfy {min} <= {value} <= {max} and {min2} <= {value} <= {max2}", raise_error
            )


def validate_equals(key, value, ref_value=None, unit=None, raise_error: bool = True) -> str | None:
    """Validate a value is equal to reference value.

    Notes:
        All values are converted into u.Quantity then compared.
        Raises ValidationError when validation fails.

    Args:
        key: key name.
        value: value to check is valid.
        ref_value: reference value to check against
        unit (str or astropy Unit): Unit for comparisons.
        raise_error (bool): Raise a ValidationError on failure.
    """
    try:
        assert to_quantity(value, unit) == to_quantity(ref_value, unit)
        logger.debug(f"PASS (equals): {key} {value} is valid")
    except AssertionError:
        logger.error(f"FAIL (equals): {key} {value} is invalid")
        return _raise_or_return(f"{key} must equal to reference value {value} != {ref_value}", raise_error)


def validate_oneof(
    key: str, value, allowed_values: list = None, raise_error: bool = True, unit: str = None
) -> str | None:
    """Validate a value is within a list of allowed values.

    Notes:
        Raises ValidationError when validation fails.

    Args:
        key: key name.
        value: value to check is valid.
        allowed_values: list of allowed reference values to check against
        raise_error (bool): Raise a ValidationError on failure.
        unit (str or astropy Unit): Unit for comparisons (not currently used).
    """
    try:
        assert value in allowed_values
        logger.debug(f"PASS (oneof): {key} {value} is valid")
    except AssertionError:
        logger.error(f"FAIL (oneof): {key} {value} is invalid")
        return _raise_or_return(f"{key} {value} must be one of {allowed_values}", raise_error)


def validate_param(key, value, vfunc: str, vkws: dict, raise_error: bool = True) -> str | None:
    """Validate value using given function name.

    Notes:
        Passes dictionary of keyword arguments to validation functions.
        Raises ValidationError when validation fails.

    Args:
        key: key name.
        value: value to check is valid.
        vfunc (str): validation function to use. One of
                     'mimax', 'equals', 'minmax2', 'oneof'.
        vkws (dict): Dictionary of keyword arguments to pass to validator.
        raise_error (bool): Will raise ValidationError unless set to False

    Returns:
        errmsg (str): Returned error message (if raise_error=False)
    """
    vfuncs = {
        "minmax": validate_minmax,
        "equals": validate_equals,
        "minmax2": validate_minmax2,
        "oneof": validate_oneof,
    }

    validator = vfuncs.get(vfunc)
    return validator(key, value, raise_error=raise_error, **vkws)


#####################
## RULE VALIDATION ##
#####################


def apply_rule_expr(rule: dict, params: dict, raise_error: bool = True) -> str | None:
    """Apply a rule expression to check params dict.

    Args:
        rule (dict): Rule definition (a dict with name/description and expr/func keys).
        params (dict): Dictionary of parameters to validate.
        raise_error (bool): Raise a ValidationError on failure.

    Example:
        ```
        rule = {
            'name': 'check_bw',
            'description': 'Check requested observation band does not exceed the receiver upper limit',
            'expr': 'centralFrequency + bandwidth/2 <= rx_high'
        }

        params = {
            'rx_high': 1300
            'bandwidth': 300
            'centralFrequency': 1000
        }

        apply_rule(rules, params)
        ```
    """
    input_params = deepcopy(params)
    for k in params:
        params[k] = to_base_value(params[k])

    rule_passes = ne.evaluate(rule["expr"], local_dict=params)
    if rule_passes:
        logger.debug(f"PASS ({rule['name']}): {rule['expr']}")
        return True
    else:
        msg = Template(rule["errmsg"]).substitute(**input_params)
        logger.error(msg)
        return _raise_or_return(msg, raise_error)


def apply_rule_func(rule: dict, params: dict, raise_error: bool = True):
    """Apply a rule by calling a function to check params dict.

    Example:
        ```
        rule = {
            'name': 'is_close',
            'description': 'Check a and b are close-ish',
            'func': 'numpy.isclose'
        }

        params = {
            'a': 1300
            'b': 1300.00000000001
        }

        apply_rule(rules, params)

    Notes:
        This works with most functions, but will not work with functions
        that use the PEP 3102 keyword-only argument e.g.
        compare(a, b, *, key=None)

    Args:
        rule (dict): Dictionary with keys 'name', 'description' and 'func'.
                     'func' is the name of a function within a package.
        params (dict): Dictionary of parameters to apply rules to.
        raise_error (bool): Raise a ValidationError if true (default).
    """
    for k in params:
        if isinstance(params[k], str):
            if params[k][0].isdigit():
                params[k] = to_base_value(params[k])
        elif isinstance(params[k], (u.Quantity, float, int)):
            params[k] = to_base_value(params[k])
        else:
            pass
    if isinstance(rule["func"], str):
        pkg_str, func_str = rule["func"].rsplit(".", 1)
        pkg = import_module(pkg_str)
        func = getattr(pkg, func_str)
    elif callable(rule["func"]):
        func = rule["func"]
    else:
        raise ValidationError(f"Function {rule['func']} not callable")

    result = func(**params)

    if isinstance(result, (bool, np.bool_)):
        if result:
            logger.debug(f"PASS ({rule['name']}) - {rule['func']}")
            return result
        else:
            logger.error(rule["errmsg"])
            return _raise_or_return(rule["errmsg"], raise_error)
    else:
        msg = rule["errmsg"]
        # If result is a tuple or list, we treat the error message as a template.
        # result must be (False, dict_of_vars_for_err_template_substitution)
        err_dict = result[1]
        msg = Template(msg).substitute(**err_dict)

        logger.error(msg)
        return _raise_or_return(msg, raise_error)


def apply_rule(rule: dict, params: dict, raise_error: bool = True):
    """Apply a rule to check params dict.

    Args:
        rule (dict): Rule definition (a dict with name/description and expr/func keys).
        params (dict): Dictionary of parameters to validate.
        raise_error (bool): Raise a ValidationError on failure.

    Returns:
        errmsg (str): Error message if validation fails, None if validation passes.
    """
    if "expr" in rule.keys():
        return apply_rule_expr(rule, params, raise_error=raise_error)
    elif "func" in rule.keys():
        return apply_rule_func(rule, params, raise_error=raise_error)


def apply_rules(rules: dict, params: dict, raise_error: bool = True, raise_on_first: bool = False) -> list:
    """Apply a set of rules to validate params dict.

    Notes:
        Each rule is verified against apply_rule_func() or
        apply_rule_expr() methods.

    Args:
        rules (dict): Dictionary of rules. Each rule should have
                      keys 'name', 'description' and 'func' OR 'expr'.
        params (dict): Dictionary of parameters to apply rules to.
        raise_error (bool): Raise a ValidationError if true (default).
        raise_on_first (bool): Raise a ValidationError at first failure (default False),
                               otherwise test for failure after running all checks.

    Returns:
        errmsg_list (list): List of error messages if raise_error=False.
    """
    results = []
    rraise = np.logical_and(raise_error, raise_on_first)
    for _rule_id, rule in rules.items():
        rule_result = apply_rule(rule, params, raise_error=rraise)
        logger.debug(f"RULE RESULT: {rule_result}")

        if isinstance(rule_result, str):
            if raise_on_first:
                return [rule_result]
            else:
                results.append(rule_result)

    if raise_error:
        if len(results) > 0:
            logger.debug(f"Results: {results}")
            raise ValidationError("Not all rules passed validation")
    else:
        logger.debug(f"Returned results: {results}")
        return results


##########################################
## CONFIG VALIDATION - PARAM BASE CLASS ##
##########################################


class ConfigValidator(p.Parameterized):
    """Base class for configuration validation.

    Uses p.Parameterized class, adds _schema param and a validate() method.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._schema = p.Dict(doc="Validation schema", allow_None=True, precedence=-1)
        self.runtime_params = p.Dict(doc="Runtime parameters (for rules / validation)", default={}, precedence=-1)
        self.load_errs = p.List(doc="List of errors raised when loading data", default=[], precedence=-1)

    def validate(self, raise_error: bool = True) -> list[str]:
        """Apply validation.

        Applies validation of parameters and rules, based on schema.

        Args:
            raise_error (bool): Raise ValidationError if it fails. Default True.

        Returns:
            errmsg_list (list): List of error messages (only if raise_error=False)
        """
        params = self._schema.get("params", None)

        errmsg_list = []

        if params:
            val_dict = self.param.values()
            # Skip validation if enabled flag is set to False.
            if "enabled" in params:
                if not val_dict["enabled"]:
                    return []

            # Extract out values in schema
            val_dict = {x: val_dict.get(x) for x in params}

            for param_id, (vfunc, vkws) in params.items():
                val_dict[param_id] = to_quantity(val_dict[param_id], unit=vkws.get("unit", None))
                errmsg = validate_param(param_id, val_dict[param_id], vfunc, vkws, raise_error=raise_error)
                if errmsg:
                    errmsg_list.append(errmsg)

            rules = self._schema.get("rules", None)
            if rules:
                # Add in runtime params
                val_dict.update({x.strip("$"): self.runtime_params[x] for x in self.runtime_params})
                for rule in rules:
                    errmsg = apply_rules({rule: RULE_DICT[rule]}, val_dict, raise_error=raise_error)
                    if errmsg:
                        for _errmsg in errmsg:
                            errmsg_list.append(_errmsg)

        return errmsg_list
