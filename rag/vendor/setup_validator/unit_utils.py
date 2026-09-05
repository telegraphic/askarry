# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/unit_utils.py).
"""unit_utils.py - utilities for working with astropy units."""

import numpy as np
from astropy import units as u

SI_prefix_lut = {
    30: "Q",
    27: "R",
    24: "Y",
    21: "Z",
    18: "E",
    15: "P",
    12: "T",
    9: "G",
    6: "M",
    3: "k",
    -3: "m",
    -6: "u",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
    -21: "z",
    -24: "y",
    -27: "r",
    -30: "q",
}


def get_base_unit(x: u.Quantity) -> u.Unit:
    """Return base unit (e.g. Hz from MHz).

    Args:
        x (u.Quantity): Astropy quantity to get base unit of.

    Returns:
        x_unit (u.Unit): Corresponding base unit
    """
    base_unit = x.unit
    if isinstance(x.unit, u.PrefixUnit):
        base_unit = u.Unit(str(x.unit)[1:])
    return base_unit


def to_si_prefix(x: u.Quantity) -> u.Quantity:
    """Convert to closest SI prefix.

    E.g. convert 1e6 Hz -> 1 MHz

    Args:
        x (u.Quantity): Astropy quantity to convert to human-readable SI unit.

    Returns:
        x (u.Quantity): Input quantity with closest SI prefix.
    """
    base_unit = get_base_unit(x)

    x = x.to(base_unit)
    x_power = int(np.log10(x.value) // 3) * 3
    prefix = SI_prefix_lut[x_power]
    x = x.to(f"{prefix}{x.unit}")
    return x


def to_display_str(value: float, display_unit: str) -> str:
    """Convert value to a string for display.

    Args:
        value (float): Value to convert, in base units (e.g. Hz, without SI prefix).
        display_unit (str): Unit to display display_str as (with SI prefix).

    Returns:
        display_str (str): Value displayed in desired display_unit.

    Examples:
        to_display_str(1e9, 'MHz') -> '1000 MHz'
        to_display_str(100, 'MHz') -> '0.0001 MHz'
    """
    base_unit = get_base_unit(1 * u.Unit(display_unit))
    quantity = value * base_unit
    return str(np.round(quantity.to(display_unit), 10))


def from_display_str(disp_str: str) -> float:
    """Convert to unitless value in base units (no SI prefixes).

    Args:
        disp_str (str): Value displayed (in display_units).

    Returns:
        value (float): Value converted to base units.

    Examples:
        from_display_str('1000 MHz') -> 1e9
    """
    quantity = u.Quantity(disp_str)
    base_unit = get_base_unit(1 * u.Unit(quantity))
    return quantity.to(base_unit).value


def to_quantity(x: float | u.Quantity, unit: str = None) -> u.Quantity:
    """Convert a number into an astropy quantity.

    Args:
        x (float or u.Quantity): Number to convert.
        unit (str): Unit string e.g. 'MHz'
    """
    if unit is None:
        if isinstance(x, str):
            if x[0].isdigit():
                return u.Quantity(x)
            elif x[0] == "$":
                raise ValueError(f"Runtime variable must be converted into value first: {x}")
            else:
                return x
        else:
            return u.Quantity(x)
    else:
        if isinstance(x, u.Quantity):
            x = x.to(unit)
        elif isinstance(x, str):
            if x[0] == "$":
                raise ValueError(f"Runtime variable must be converted into value first: {x}")
            elif x[0].isdigit():
                try:
                    x = u.Quantity(x).to(unit)
                except:  # noqa: E722
                    x = u.Quantity(f"{x} {unit}")
            else:
                return x
        else:
            x = u.Quantity(f"{x} {unit}")
        return x


def to_base_value(x: float | u.Quantity, unit: str = None) -> float:
    """Convert a number to its base value.

    Converts to value, in base (no SI prefix), e.g. MHz -> Hz

    Args:
        x (float or u.Quantity): Number of convert
        unit (str): Unit string e.g. 'MHz'

    Returns:
        x (float): Value of x in base units, as a float.
    """
    x = to_quantity(x, unit)
    base_unit = get_base_unit(x)
    return x.to(base_unit).value
