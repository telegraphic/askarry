# Vendored from ska-sci-ops/odp-data-size-tool (str_utils.py); unchanged.
"""str_utils.py — String formatting and parsing utilities.

Contains pretty-printing helpers and unit-string parsers used across the app.
No Panel dependency, so these are straightforward to unit-test.
"""

from astropy import units as u
from astropy.units import Quantity


# ---------------------------------------------------------------------------
# String parsing
# ---------------------------------------------------------------------------

def parse_qty(text: str, default_unit: str | None = None) -> Quantity:
    """Parse a text like '2 deg' or '3600 s' into an astropy Quantity.
    If no unit provided, uses default_unit when given; else raises.
    """
    text = (text or '').strip()
    if not text:
        raise ValueError("Empty quantity string")
    if any(ch.isalpha() for ch in text):
        return u.Quantity(text)
    if default_unit is None:
        raise ValueError("No unit provided and no default unit specified")
    return u.Quantity(float(text), default_unit)


def md_escape(text: str) -> str:
    """Escape Markdown special characters (currently just '*') in *text*."""
    return text.replace('*', r'\*')


def rate_alert_html(rate_gb_s: float, n_beam: int, vol_str: str, prefix: str = "") -> str:
    """HTML for a data-rate/volume result, highlighting the rate red if it
    exceeds 0.5 GB/s per beam. *prefix* is inserted before the rate label,
    e.g. an "n_chan: ..." fragment.
    """
    rate_str = f"{rate_gb_s:.3f} GB/s"
    if rate_gb_s > 0.5 * n_beam:
        rate_display = f'<span style="color:red;font-weight:bold">{rate_str}</span>'
    else:
        rate_display = rate_str
    return f"<p>{prefix}<b>Data rate:</b> {rate_display}</p><h3>Data volume: {vol_str}</h3>"


def parse_volume_to_tb(vol_str: str) -> float:
    """Parse a data-volume string such as ``'4.000 TB'`` or ``'512.000 GB'`` into TB.

    Returns 0.0 if the string cannot be parsed.
    """
    try:
        q = Quantity(vol_str)
        return float(q.to('TB').value)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Bandwidth unit helpers
# ---------------------------------------------------------------------------

_BW_UNITS = ['Hz', 'kHz', 'MHz', 'GHz']
_BW_FACTORS = {'Hz': 1.0, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}


def to_hz(value: float, unit: str) -> float:
    """Convert a bandwidth value expressed in *unit* to Hz."""
    return value * _BW_FACTORS.get(unit, 1.0)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def sig_fmt(value: float, decimals: int = 3) -> str:
    """Format *value* to up to *decimals* decimal places, trimming trailing zeros.

    Unlike a fixed sig-fig count, this never drops real precision (e.g. 781.25
    stays 781.25) while still trimming padding zeros (849.000 -> 849).
    """
    s = f"{value:.{decimals}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s or "0"


def pretty(q: Quantity) -> str:
    try:
        return f"{sig_fmt(float(q.value))} {q.unit}"
    except Exception:
        return str(q)


def auto_volume(q: Quantity) -> Quantity:
    """Return *q* converted to the most readable data-volume unit.

    Rules (applied in order):
    - >= 1 000 TB  → PB
    -   0.1 – 1 000 TB → TB  (explicit override: always use TB in this range)
    -   1 GB – 0.1 TB  → GB
    -  < 1 GB           → MB
    """
    tb = float(q.to("TB").value)
    if tb >= 1000:
        return q.to("PB")
    if tb >= 0.1:
        return q.to("TB")
    gb = tb * 1000
    if gb >= 1:
        return q.to("GB")
    return q.to("MB")


def pretty_freq(hz: float) -> str:
    """Format a frequency value (in Hz) using the most readable unit."""
    abs_hz = abs(hz)
    if abs_hz >= 1e9:
        return f"{sig_fmt(hz / 1e9)} GHz"
    elif abs_hz >= 1e6:
        return f"{sig_fmt(hz / 1e6)} MHz"
    elif abs_hz >= 1e3:
        return f"{sig_fmt(hz / 1e3)} kHz"
    else:
        return f"{sig_fmt(hz)} Hz"


def pretty_time(seconds: float) -> str:
    """Format a time value (in seconds) using the most readable unit (s, ms, us)."""
    abs_s = abs(seconds)
    if abs_s >= 1.0:
        return f"{sig_fmt(seconds)} s"
    elif abs_s >= 1e-3:
        return f"{sig_fmt(seconds * 1e3)} ms"
    else:
        return f"{sig_fmt(seconds * 1e6)} us"


def pretty_rate(q: Quantity) -> str:
    """Format a data-rate Astropy Quantity using the most readable TB/s sub-unit."""
    try:
        tb_s = float(q.to('TB/s').value)
    except Exception:
        return str(q)
    abs_val = abs(tb_s)
    if abs_val >= 1.0:
        return f"{sig_fmt(tb_s)} TB/s"
    elif abs_val >= 1e-3:
        return f"{sig_fmt(tb_s * 1e3)} GB/s"
    elif abs_val >= 1e-6:
        return f"{sig_fmt(tb_s * 1e6)} MB/s"
    else:
        return f"{sig_fmt(tb_s * 1e9)} KB/s"


def parse_pretty_rate_to_tb_s(rate_str: str) -> float:
    """Parse a :func:`pretty_rate` string (e.g. ``'20.408 MB/s'``) to TB/s.

    Inverse of :func:`pretty_rate`.

    Raises :class:`ValueError` if the string cannot be parsed.
    """
    s = str(rate_str).strip()
    for suffix, factor in (('TB/s', 1.0), ('GB/s', 1e-3), ('MB/s', 1e-6), ('KB/s', 1e-9)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)].strip()) * factor
    raise ValueError(f"Cannot parse rate string: {rate_str!r}")


def pretty_angle(q: Quantity) -> str:
    """Format an angular quantity with intelligently chosen units.

    Small angles show in arcsec, medium in arcmin, larger in degrees.
    """
    try:
        val_deg = float(q.to(u.deg).value)
        if val_deg < 0.01:  # < 36 arcsec, show in arcsec
            return f"{sig_fmt(float(q.to(u.arcsec).value))} arcsec"
        elif val_deg < 1.0:  # < 60 arcmin, show in arcmin
            return f"{sig_fmt(float(q.to(u.arcmin).value))} arcmin"
        else:  # >= 1 deg, show in deg
            return f"{sig_fmt(val_deg)} deg"
    except Exception:
        return str(q)


def md_two_lines(label1: str, val1: str, label2: str, val2: str) -> str:
    """Return a two-line markdown snippet with *label2/val2* as a heading."""
    # val2 (data volume) is displayed as a prominent heading
    return f"""**{label1}:** {val1}

### {label2}: {val2}"""
