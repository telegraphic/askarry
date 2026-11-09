"""
Read-only astronomical object lookup via astroquery: SIMBAD for names,
coordinates, object types and aliases; NED as a fallback and for
extragalactic redshifts.

Deliberately a handful of fixed queries rather than a generic "call any
astroquery method" bridge, so the MCP surface can't download, upload, log in
or otherwise do more than look things up.
"""

from __future__ import annotations

import re

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

TIMEOUT_S = 30
MAX_ALIASES = 15
# SIMBAD returns cone results unsorted, so fetch the whole cone (up to this
# cap) before sorting by distance; dense fields report "truncated".
CONE_ROW_CAP = 20000


def _clean(value):
    """Masked/NumPy table values → plain JSON-safe Python (None if masked)."""
    if value is np.ma.masked or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _coords(ra_deg: float, dec_deg: float) -> dict:
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    return {
        "ra_deg": round(ra_deg, 6),
        "dec_deg": round(dec_deg, 6),
        "ra_hms": c.ra.to_string(u.hourangle, sep=":", precision=2, pad=True),
        "dec_dms": c.dec.to_string(sep=":", precision=1, alwayssign=True, pad=True),
        "gal_l_deg": round(c.galactic.l.deg, 4),
        "gal_b_deg": round(c.galactic.b.deg, 4),
    }


def _simbad():
    from astroquery.simbad import Simbad

    s = Simbad()
    s.TIMEOUT = TIMEOUT_S
    s.add_votable_fields("otype", "rvz_redshift")
    return s


def ned_lookup(name: str) -> dict:
    """NED's entry for an object: position, type, redshift and velocity."""
    from astroquery.exceptions import RemoteServiceError
    from astroquery.ipac.ned import Ned

    Ned.TIMEOUT = TIMEOUT_S
    try:
        t = Ned.query_object(name)
    except RemoteServiceError:
        return {"found": False, "name": name, "service": "NED"}
    row = t[0]
    return {
        "found": True,
        "service": "NED",
        "name": _clean(row["Object Name"]),
        "object_type": _clean(row["Type"]),
        "redshift": _clean(row["Redshift"]),
        "velocity_km_s": _clean(row["Velocity"]),
        **_coords(float(row["RA"]), float(row["DEC"])),
    }


# Catalogues readers know, most familiar first, for picking a display name
# when SIMBAD's main identifier is obscure (Cygnus A's is "MCG+07-41-003").
_FAMILIAR = [r"M\s+\d+$", r"NGC\s+\d+\w?$", r"IC\s+\d+$", r"3C\s+\d+(\.\d+)?\w?$", r"PSR\s+[JB]\S+$",
             r"4C\s+\S+$", r"PKS\s+\S+$", r"SN\s+\S+$", r"GRB\s+\S+$", r"FRB\s+\S+$"]


def _norm_id(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def display_name(main_id: str, aliases: list[str], query: str | None = None) -> str:
    """A familiar name for an object: the query itself if it is one of
    SIMBAD's common names; else a Messier number ("M 31"); else SIMBAD's
    common name ("NAME Cygnus A" -> "Cygnus A"); else the first alias from a
    well-known catalogue (NGC, IC, 3C, PSR, ...); else the main identifier.
    Runs of spaces are collapsed ("M  31" -> "M 31")."""
    clean = lambda s: re.sub(r"\s+", " ", str(s)).strip()  # noqa: E731
    names = [clean(a)[5:] for a in aliases if clean(a).startswith("NAME ")]
    q = _norm_id(query) if query else None
    hit = next((n for n in names if _norm_id(n) == q), None)
    if hit:
        return hit
    messier = next((clean(a) for a in [main_id, *aliases] if re.match(_FAMILIAR[0], clean(a))), None)
    if messier or names:
        return messier or names[0]
    for pattern in _FAMILIAR[1:]:
        hit = next((clean(a) for a in [main_id, *aliases] if re.match(pattern, clean(a))), None)
        if hit:
            return hit
    return clean(main_id)


def resolve_object(name: str) -> dict:
    """SIMBAD's entry for an object name (falling back to NED): position in
    several frames, object type, redshift, common aliases, a familiar
    display name (SIMBAD's main ID can be obscure: Cygnus A is
    "MCG+07-41-003"), and a confidence: 1.0 when the name is one of SIMBAD's
    identifiers for the object, 0.8 when SIMBAD resolved it some other way,
    0.6 from the NED fallback."""
    import warnings

    from astroquery.simbad import Simbad

    with warnings.catch_warnings():  # NoResultsWarning for unknown names
        warnings.simplefilter("ignore")
        t = _simbad().query_object(name)
    if len(t) == 0:
        out = ned_lookup(name)
        if out.get("found"):
            out.update(main_id=out["name"], display_name=out["name"], confidence=0.6)
        return out
    row = t[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = Simbad.query_objectids(name)
    every = [str(i) for i in ids["id"]] if ids is not None else []
    main_id = _clean(row["main_id"])
    known = {_norm_id(i) for i in [main_id, *every]}
    exact = _norm_id(name) in known or _norm_id(f"NAME {name}") in known
    return {
        "found": True,
        "service": "SIMBAD",
        "name": main_id,  # kept for existing callers; same as main_id
        "main_id": main_id,
        "display_name": display_name(main_id, every, name),
        "confidence": 1.0 if exact else 0.8,
        "object_type": _clean(row["otype"]),
        "redshift": _clean(row["rvz_redshift"]),
        **_coords(float(row["ra"]), float(row["dec"])),
        "aliases": every[:MAX_ALIASES],
    }


def to_skycoord(target: str) -> SkyCoord:
    """Parse "ra dec" in degrees, sexagesimal ("13:37:00 -29:51:56",
    "13h37m00s -29d51m56s"), or else resolve *target* as an object name."""
    parts = target.replace(",", " ").split()
    if len(parts) == 2:
        try:
            return SkyCoord(float(parts[0]) * u.deg, float(parts[1]) * u.deg)
        except ValueError:
            pass
    if re.search(r"\d[:hH]\s*\d", target):
        return SkyCoord(target, unit=(u.hourangle, u.deg))
    found = resolve_object(target)
    if not found["found"]:
        raise ValueError(f"Could not resolve {target!r} in SIMBAD or NED")
    return SkyCoord(found["ra_deg"] * u.deg, found["dec_deg"] * u.deg)


def cone_search(target: str, radius_arcmin: float = 5.0, max_results: int = 50,
                object_type: str | None = None) -> dict:
    """SIMBAD objects within *radius_arcmin* of *target* (a name or
    coordinates), nearest first, optionally only those of one SIMBAD object
    type (e.g. "Psr", "QSO", "G", "HII")."""
    import warnings

    center = to_skycoord(target)
    s = _simbad()
    s.ROW_LIMIT = CONE_ROW_CAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = s.query_region(center, radius=radius_arcmin * u.arcmin)
    rows = []
    for row in t:
        if object_type and _clean(row["otype"]) != object_type:
            continue
        pos = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        rows.append({
            "name": _clean(row["main_id"]),
            "object_type": _clean(row["otype"]),
            "redshift": _clean(row["rvz_redshift"]),
            "ra_deg": round(float(row["ra"]), 6),
            "dec_deg": round(float(row["dec"]), 6),
            "separation_arcmin": round(float(center.separation(pos).arcmin), 3),
        })
    rows.sort(key=lambda r: r["separation_arcmin"])
    return {
        "center": _coords(center.ra.deg, center.dec.deg),
        "radius_arcmin": radius_arcmin,
        "n_found": len(rows),
        "truncated": len(t) >= CONE_ROW_CAP,
        "objects": rows[:max_results],
    }
