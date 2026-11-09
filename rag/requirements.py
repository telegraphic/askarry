"""
Structured observing requirements, and field-by-field checks against the
quote they came from.

Atlas v2 extracted requirements as free text ("~1.1 microJy/beam (5a)"),
so a "match" with the sensitivity calculator could rest on a weighting or a
centre frequency the paper never stated (C-19's robust -2 came from the
agent, not the quote). A requirement record keeps each field's value, the
words it came from, and whether it was inferred:

    {"sensitivity": {"value": 1.1, "unit": "uJy/beam", "source_text": "1.1 μJy/beam", "inferred": false},
     "weighting":   {"value": "robust -2", "source_text": null, "inferred": true}, ...}

extract_draft() fills the fields a deterministic parser can find in a quote;
an agent completes the rest, marking them inferred. check_fields() then says,
for every field, whether its value really appears in the quote.
"""

from __future__ import annotations

import re
import unicodedata

# Field -> (kind, allowed units). Units are normalised (see _UNIT_ALIASES).
REQUIREMENT_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "telescope": ("text", ()),
    "mode": ("text", ()),
    "band": ("text", ()),
    "array_stage": ("text", ()),
    "frequency": ("number", ("Hz", "kHz", "MHz", "GHz")),
    "bandwidth": ("number", ("Hz", "kHz", "MHz", "GHz")),
    "channel_width": ("number", ("Hz", "kHz", "MHz", "km/s")),
    "sensitivity": ("number", ("nJy", "uJy", "mJy", "Jy", "nJy/beam", "uJy/beam", "mJy/beam", "Jy/beam", "uK", "mK", "K")),
    "sigma_level": ("number", ("sigma",)),
    "weighting": ("text", ()),
    "hours": ("number", ("h",)),
    "area": ("number", ("deg2",)),
    "fov": ("number", ("deg2", "deg", "arcmin")),
    "angular_resolution": ("number", ("mas", "arcsec", "arcmin", "deg")),
    "time_resolution": ("number", ("ns", "us", "ms", "s")),
}
MODES = ("continuum", "zoom", "pss", "pst", "vlbi", "transient_buffer", "solar")

_UNIT_ALIASES = {
    "μjy": "uJy", "µjy": "uJy", "ujy": "uJy", "microjy": "uJy", "njy": "nJy", "mjy": "mJy", "jy": "Jy",
    "hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz", "k": "K", "mk": "mK", "uk": "uK", "μk": "uK", "µk": "uK",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "deg2": "deg2", "deg²": "deg2", "sqdeg": "deg2", "squaredegrees": "deg2", "squaredegree": "deg2",
    "arcsec": "arcsec", "″": "arcsec", "′′": "arcsec", "''": "arcsec", "\"": "arcsec", "arcmin": "arcmin", "′": "arcmin", "mas": "mas",
    "deg": "deg", "°": "deg", "s": "s", "ms": "ms", "us": "us", "μs": "us", "µs": "us", "ns": "ns",
    "sigma": "sigma", "σ": "sigma", "km/s": "km/s", "kms-1": "km/s",
}
# Scale to a base unit per dimension, so "1.1 uJy" in a record matches "1100 nJy" in a quote.
_SCALE = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "nJy": 1e-9, "uJy": 1e-6, "mJy": 1e-3, "Jy": 1,
          "uK": 1e-6, "mK": 1e-3, "K": 1, "h": 1, "deg2": 1, "mas": 1e-3, "arcsec": 1, "arcmin": 60, "deg": 3600,
          "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "sigma": 1, "km/s": 1}

_NUM = r"(\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?\s*-?\d+)?)"
# value + unit patterns found in quotes, each tagged with the dimension it measures.
_QUANTITY = re.compile(
    _NUM + r"\s*(?P<unit>"
    r"[nuμµm]?\s?Jy(?:\s*/\s*beam|\s+beam\s*-?1)?|[μµum]?K\b|[kMG]?Hz\b|km\s*/?\s*s(?:\s*-1)?|"
    r"hours?\b|hrs?\b|h\b|deg(?:rees?)?\s*(?:2|²|squared)|sq\.?\s*deg\w*|square\s+degrees?|"
    r"arcsec\b|arcmin\b|mas\b|″|′′|''|′|°|"
    r"[μµu]s\b|ms\b|ns\b|s\b|σ|-?sigma\b)", re.IGNORECASE)
_WEIGHTING = re.compile(r"\b(?:briggs\s+)?robust(?:ness)?(?:\s+(?:imaging|weighting|parameter|value|of)){0,3}"
                        r"\s*(?:=|of)?\s*(-?\s*\d+(?:\.\d+)?)|"
                        r"\b(natural|uniform)\s+weight\w*|\bweight\w*\s*[:=]?\s*(natural|uniform)", re.IGNORECASE)
_STAGE = re.compile(r"\bAA\s*(\*|4|2|1|0\.5)(?!\d)|\bAA\s*star\b", re.IGNORECASE)
_BAND = re.compile(r"\bBand\s*(1|2|3|4|5a|5b|5)\b", re.IGNORECASE)


def normalise_text(text: str) -> str:
    """Quote text as PDF extraction leaves it, made comparable: NFKC (so the
    mathematical 𝜇 becomes μ), "1 . 3" -> "1.3", tildes and "about" marks
    removed, whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", t)
    t = re.sub(r"[∼~≈≃]", " ", t)
    t = t.replace("μ", "u").replace("µ", "u").replace("−", "-")
    return re.sub(r"\s+", " ", t).strip()


def _unit(raw: str) -> str | None:
    u = re.sub(r"\s+", "", raw.lower())
    u = re.sub(r"(jy)(/beam|beam-?1)$", r"\1", u)
    u = re.sub(r"^deg(rees?)?(2|²|squared)$|^sq\.?deg\w*$|^squaredegrees?$", "deg2", u)
    u = u.lstrip("-")
    return _UNIT_ALIASES.get(u)


def _number(raw: str) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)\s*(?:[x×]\s*10\^?\s*(-?\d+))?", raw)
    return float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1)


def quantities(text: str) -> list[dict]:
    """Every value-with-unit in *text*: [{"value", "unit", "base", "text", "start"}]."""
    t = normalise_text(text)
    out = []
    for m in _QUANTITY.finditer(t):
        unit = _unit(m.group("unit"))
        if unit is None:
            continue
        value = _number(m.group(1))
        out.append({"value": value, "unit": unit, "base": value * _SCALE.get(unit, 1),
                    "text": m.group(0), "start": m.start()})
    return out


def _dimension(unit: str) -> str:
    for dim, units in (("freq", ("Hz", "kHz", "MHz", "GHz")), ("flux", ("nJy", "uJy", "mJy", "Jy")),
                       ("temp", ("uK", "mK", "K")), ("time_h", ("h",)), ("area", ("deg2",)),
                       ("angle", ("mas", "arcsec", "arcmin", "deg")), ("time_s", ("ns", "us", "ms", "s")),
                       ("sigma", ("sigma",)), ("velocity", ("km/s",))):
        if unit in units:
            return dim
    return unit


def _context_before(t: str, start: int, width: int = 40) -> str:
    return t[max(0, start - width):start].lower()


def extract_draft(quote: str) -> dict:
    """A requirement record with every field a deterministic parser finds in
    *quote* (value, unit, source_text, inferred=false); other fields null.

    Frequencies are told apart by the words just before them: "bandwidth"
    makes a bandwidth, "channel"/"resolution"/"smoothed" a channel width,
    otherwise a centre frequency. The first match of each field wins; an
    agent should review the draft, not trust it.
    """
    t = normalise_text(quote)
    rec: dict = {f: None for f in REQUIREMENT_FIELDS}

    def put(field, value, unit, source):
        if rec[field] is None:
            rec[field] = {"value": value, "unit": unit, "source_text": source, "inferred": False}

    for q in quantities(quote):
        dim, before = _dimension(q["unit"]), _context_before(t, q["start"])
        if dim == "freq":
            # Hz and kHz figures are channel widths (no SKA centre frequency is that low).
            field = ("bandwidth" if re.search(r"bandwidth|band width|wide\b", before) else
                     "channel_width" if re.search(r"channel|resolution|smooth|spectral", before)
                     or q["unit"] in ("Hz", "kHz") else "frequency")
            put(field, q["value"], q["unit"], q["text"])
        elif dim == "velocity":
            put("channel_width", q["value"], q["unit"], q["text"])
        elif dim in ("flux", "temp"):
            beam = "beam" in t[q["start"]:q["start"] + len(q["text"]) + 8].lower()
            put("sensitivity", q["value"], q["unit"] + ("/beam" if beam and dim == "flux" else ""), q["text"])
        elif dim == "time_h":
            put("hours", q["value"], "h", q["text"])
        elif dim == "area":
            put("fov" if re.search(r"field of view|fov|primary beam", before) else "area", q["value"], "deg2", q["text"])
        elif dim == "angle":
            put("angular_resolution", q["value"], q["unit"], q["text"])
        elif dim == "time_s":
            put("time_resolution", q["value"], q["unit"], q["text"])
        elif dim == "sigma":
            put("sigma_level", q["value"], "sigma", q["text"])
    m = _WEIGHTING.search(t)
    if m:
        value = f"robust {m.group(1).replace(' ', '')}" if m.group(1) else (m.group(2) or m.group(3)).lower()
        put("weighting", value, None, m.group(0))
    m = _STAGE.search(t)
    if m:
        put("array_stage", "AA*" if m.group(1) in ("*", None) else f"AA{m.group(1)}", None, m.group(0))
    m = _BAND.search(t)
    if m:
        put("band", f"Band {m.group(1).lower()}", None, m.group(0))
    if re.search(r"SKA[- ]?Low", t, re.I):
        put("telescope", "low", None, re.search(r"SKA[- ]?Low", t, re.I).group(0))
    elif re.search(r"SKA[- ]?Mid", t, re.I):
        put("telescope", "mid", None, re.search(r"SKA[- ]?Mid", t, re.I).group(0))
    return rec


def normalise_record(record: dict) -> dict:
    """Accept a bare value for a field ("telescope": "mid") as {"value": ...}."""
    return {k: v if v is None or isinstance(v, dict) else {"value": v} for k, v in (record or {}).items()}


def validate_record(record: dict) -> list[str]:
    """Problems with a requirement record's structure (empty if none):
    unknown fields, fields that aren't {value, ...} objects, units a field
    doesn't take, unknown modes."""
    problems = []
    for field, entry in normalise_record(record).items():
        if field not in REQUIREMENT_FIELDS:
            problems.append(f"unknown field {field!r}")
            continue
        if entry is None:
            continue
        if not isinstance(entry, dict) or "value" not in entry:
            problems.append(f"{field}: expected {{'value', 'unit', 'source_text', 'inferred'}}")
            continue
        kind, units = REQUIREMENT_FIELDS[field]
        if kind == "number" and entry["value"] is not None and not isinstance(entry["value"], (int, float)):
            problems.append(f"{field}: value must be a number")
        if units and entry.get("unit") is not None and entry["unit"] not in units:
            problems.append(f"{field}: unit {entry['unit']!r} not one of {list(units)}")
        if field == "mode" and entry["value"] and str(entry["value"]).lower() not in MODES:
            problems.append(f"mode: {entry['value']!r} not one of {list(MODES)}")
    return problems


def value(record: dict, field: str, unit: str | None = None):
    """A field's value, converted to *unit* when both are numeric units."""
    entry = (record or {}).get(field)
    if not entry or entry.get("value") is None:
        return None
    v = entry["value"]
    if unit and entry.get("unit") and isinstance(v, (int, float)):
        src = entry["unit"].replace("/beam", "")
        return v * _SCALE[src] / _SCALE[unit.replace("/beam", "")]
    return v


def _text_in_quote(value, field: str, quote_t: str, found: list[dict]) -> str | None:
    """The quote's own words for *value*, or None if the quote doesn't state it."""
    if field == "weighting":
        m = _WEIGHTING.search(quote_t)
        if not m:
            return None
        stated = f"robust {m.group(1).replace(' ', '')}" if m.group(1) else (m.group(2) or m.group(3)).lower()
        want = re.sub(r"^briggs\s+", "", str(value).lower()).replace("robustness", "robust").replace("=", " ")
        want = re.sub(r"\s+", " ", want).strip()
        return m.group(0) if stated == want else None
    if field == "array_stage":
        want = str(value).upper().replace("STAR", "*").replace(" ", "")
        m = next((m for m in _STAGE.finditer(quote_t)
                  if ("AA*" if m.group(1) in ("*", None) else f"AA{m.group(1)}") == want), None)
        return m.group(0) if m else None
    if field == "band":
        m = next((m for m in _BAND.finditer(quote_t) if f"band {m.group(1).lower()}" == str(value).lower()), None)
        return m.group(0) if m else None
    if isinstance(value, (int, float)):
        for q in found:
            if math_close(q["value"], value):
                return q["text"]
        # A number sharing a later unit: "1.1 and 1.3 μJy/beam", "4-8 h".
        for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", quote_t):
            if math_close(float(m.group(1)), value) and any(
                    0 <= q["start"] - m.end() <= 30 and float(m.group(1)) != q["value"] for q in found):
                return m.group(0)
        return None
    return str(value) if str(value).lower() in quote_t.lower() else None


def math_close(a: float, b: float, rel: float = 0.01) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)


def check_fields(quote: str, record: dict) -> dict:
    """For every non-null field: "stated" (its value appears in the quote,
    in any unit spelling) or "inferred" (it doesn't). A field marked
    inferred=false whose value isn't in the quote is reported as inferred
    and flagged, because the record claims more than the quote says."""
    t = normalise_text(quote)
    found = quantities(quote)
    out = {}
    for field, entry in normalise_record(record).items():
        if not entry or entry.get("value") is None:
            continue
        v = entry["value"]
        unit = entry.get("unit")
        same_dim = [q for q in found if not unit or _dimension(q["unit"]) == _dimension(unit.replace("/beam", ""))]
        # Compare numbers in the record's own unit: "1.1 uJy" matches "1100 nJy".
        if isinstance(v, (int, float)) and unit and unit.replace("/beam", "") in _SCALE:
            scale = _SCALE[unit.replace("/beam", "")]
            same_dim = [{**q, "value": q["base"] / scale} for q in same_dim]
        words = _text_in_quote(v, field, t, same_dim)
        source = entry.get("source_text")
        if words is None and source and normalise_text(source).lower() in t.lower():
            # The source text is in the quote; accept it if it carries this value.
            src_q = [q for q in quantities(source)]
            if not isinstance(v, (int, float)) or any(math_close(q["value"], v) for q in src_q):
                words = source
        row = {"value": v, "unit": unit, "verdict": "stated" if words else "inferred", "quote_text": words}
        if not words and entry.get("inferred") is False:
            row["flag"] = "marked as stated, but the quote does not contain it"
        out[field] = row
    return out
