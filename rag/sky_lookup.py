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


def resolve_object(name: str) -> dict:
    """SIMBAD's entry for an object name (falling back to NED): position in
    several frames, object type, redshift and common aliases."""
    import warnings

    from astroquery.simbad import Simbad

    with warnings.catch_warnings():  # NoResultsWarning for unknown names
        warnings.simplefilter("ignore")
        t = _simbad().query_object(name)
    if len(t) == 0:
        return ned_lookup(name)
    row = t[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = Simbad.query_objectids(name)
    return {
        "found": True,
        "service": "SIMBAD",
        "name": _clean(row["main_id"]),
        "object_type": _clean(row["otype"]),
        "redshift": _clean(row["rvz_redshift"]),
        **_coords(float(row["ra"]), float(row["dec"])),
        "aliases": [str(i) for i in ids["id"][:MAX_ALIASES]] if ids is not None else [],
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
