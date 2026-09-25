"""
Target visibility and LST pressure for SKA-Low / SKA-Mid.

visibility_windows  when one target is usable over a year: LST window above
                    the elevation limit, usable hours per month and LST bin
                    under a Sun constraint (night / avoid twilight / any)
lst_pressure        requested vs available hours per LST bin (and month ×
                    LST) for a set of observing requests

Relative analysis only: it compares requests against clock time, not
against a real allocation. All operational settings are the unverified
defaults in config.SCHEDULING and are echoed in every result.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time

from rag.config import SCHEDULING

SUN_MODES = ("night", "avoid_twilight", "any")


def _site(telescope: str):
    t = telescope.lower().replace("ska-", "").replace("ska_", "")
    if t == "low":
        from ska_ost_array_config.low_subarray import LOW_ARRAY_REF
        return LOW_ARRAY_REF
    if t == "mid":
        from ska_ost_array_config.mid_subarray import MID_ARRAY_REF
        return MID_ARRAY_REF
    raise ValueError(f"Unknown telescope {telescope!r}; expected 'low' or 'mid'")


def _default_year_start() -> str:
    return f"{date.today().year + 1}-01-01"


@lru_cache(maxsize=8)
def _grid(telescope: str, year_start: str) -> dict:
    """Year-long time grid at one site: LST (h), month index, Sun altitude and
    Sun RA/Dec (deg). Cached: the Sun transform is the only slow step."""
    import warnings

    from astropy.utils.exceptions import AstropyWarning

    site = _site(telescope)
    step = SCHEDULING["time_step_min"]
    times = Time(year_start) + np.arange(0, 365 * 24 * 60, step) * u.min
    from astropy.utils import iers

    # IERS Earth-orientation data ends before future years (and may be absent
    # offline): accept degraded, arcsec-level accuracy, irrelevant here.
    with warnings.catch_warnings(), iers.conf.set_temp("iers_degraded_accuracy", "ignore"):
        warnings.simplefilter("ignore", AstropyWarning)
        sun = get_sun(times)
        return {
            "lat": site.lat.deg,
            "lst_h": times.sidereal_time("apparent", longitude=site.lon).hour,
            "month": np.array([int(m) - 1 for m in times.strftime("%m")]),
            "sun_alt": sun.transform_to(AltAz(obstime=times, location=site)).alt.deg,
            "sun_ra": sun.ra.deg,
            "sun_dec": sun.dec.deg,
            "step_h": step / 60,
        }


def _usable(g: dict, ra: float, dec: float, min_el: float, sun: str) -> np.ndarray:
    """Boolean mask over the time grid: target above *min_el* and the Sun
    constraint met. Hour-angle formula on J2000 coordinates (precession and
    refraction ignored: well under a degree, fine for scheduling)."""
    if sun not in SUN_MODES:
        raise ValueError(f"Unknown sun mode {sun!r}; expected one of {SUN_MODES}")
    phi, d = np.radians(g["lat"]), np.radians(dec)
    ha = np.radians((g["lst_h"] * 15 - ra) % 360)
    alt = np.degrees(np.arcsin(np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(ha)))
    mask = alt >= min_el
    if sun == "night":
        mask &= g["sun_alt"] < SCHEDULING["night_sun_alt_deg"]
    elif sun == "avoid_twilight":
        lo, hi = SCHEDULING["twilight_sun_alt_band_deg"]
        mask &= (g["sun_alt"] < lo) | (g["sun_alt"] > hi)
    if SCHEDULING["min_sun_separation_deg"] > 0:
        sra, sdec = np.radians(g["sun_ra"]), np.radians(g["sun_dec"])
        cos_sep = np.sin(d) * np.sin(sdec) + np.cos(d) * np.cos(sdec) * np.cos(np.radians(ra) - sra)
        mask &= np.degrees(np.arccos(np.clip(cos_sep, -1, 1))) >= SCHEDULING["min_sun_separation_deg"]
    return mask


def _cells(g: dict, mask: np.ndarray) -> np.ndarray:
    """Hours of *mask* per (month, LST bin): shape (12, 24 / lst_bin_h)."""
    n_bins = int(24 / SCHEDULING["lst_bin_h"])
    lst_bin = (g["lst_h"] // SCHEDULING["lst_bin_h"]).astype(int) % n_bins
    out = np.zeros((12, n_bins))
    np.add.at(out, (g["month"][mask], lst_bin[mask]), g["step_h"])
    return out


def assumptions(min_el: float, sun: str, year_start: str) -> dict:
    return {**SCHEDULING, "min_elevation_deg": min_el, "sun": sun, "year_start": year_start,
            "note": "Unverified defaults (config.SCHEDULING); need an SKAO owner."}


def _fmt_lst(h: float) -> str:
    h %= 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def visibility_windows(ra: float, dec: float, telescope: str, min_elevation: float | None = None,
                       sun: str = "any", year_start: str | None = None) -> dict:
    """When a target at (*ra*, *dec*) degrees is usable from a telescope site."""
    min_el = SCHEDULING["default_min_elevation_deg"] if min_elevation is None else min_elevation
    year_start = year_start or _default_year_start()
    g = _grid(telescope, year_start)
    lat = g["lat"]
    max_el = 90 - abs(dec - lat)

    # Analytic LST window above min_el (independent of the Sun).
    phi, d = np.radians(lat), np.radians(dec)
    cos_h = (np.sin(np.radians(min_el)) - np.sin(phi) * np.sin(d)) / (np.cos(phi) * np.cos(d))
    if cos_h > 1:
        window = None
    else:
        half_h = 12.0 if cos_h < -1 else np.degrees(np.arccos(cos_h)) / 15
        window = {
            "lst_start": _fmt_lst(ra / 15 - half_h),
            "lst_end": _fmt_lst(ra / 15 + half_h),
            "hours_per_sidereal_day": round(float(2 * half_h), 2),
        }

    cells = _cells(g, _usable(g, ra, dec, min_el, sun))
    return {
        "ra_deg": ra, "dec_deg": dec, "telescope": telescope,
        "site_latitude_deg": round(lat, 4),
        "max_elevation_deg": round(max_el, 2),
        "never_reaches_min_elevation": window is None,
        "lst_window_above_min_elevation": window,
        "usable_hours_per_year": round(float(cells.sum()), 1),
        "usable_hours_per_month": [round(float(x), 1) for x in cells.sum(axis=1)],
        "usable_hours_per_lst_bin": [round(float(x), 1) for x in cells.sum(axis=0)],
        "assumptions": assumptions(min_el, sun, year_start),
    }


def _request_positions(req: dict) -> list[tuple[float, float]]:
    """Point target, or a grid over an area request: ra_range/dec_range, or
    gal_l_range/gal_b_range (e.g. the Galactic plane), in degrees."""
    if "gal_b_range" in req:
        from astropy.coordinates import SkyCoord

        (l0, l1), (b0, b1) = req.get("gal_l_range", [0, 360]), req["gal_b_range"]
        span = (l1 - l0) % 360 or 360
        ls = np.repeat([(l0 + span * (i + 0.5) / 36) % 360 for i in range(36)], 5)
        bs = np.tile(np.linspace(b0, b1, 5), 36)
        icrs = SkyCoord(l=ls * u.deg, b=bs * u.deg, frame="galactic").icrs
        return list(zip(icrs.ra.deg.tolist(), icrs.dec.deg.tolist()))
    if "ra_range" in req:
        (ra0, ra1), (dec0, dec1) = req["ra_range"], req["dec_range"]
        span = (ra1 - ra0) % 360 or 360
        ras = [(ra0 + span * (i + 0.5) / 24) % 360 for i in range(24)]
        decs = np.linspace(dec0, dec1, 5)
        return [(r, float(dd)) for r in ras for dd in decs]
    return [(float(req["ra"]), float(req["dec"]))]


def lst_pressure(requests: list[dict], telescope: str, year_start: str | None = None) -> dict:
    """Spread each request's hours over the (month, LST-bin) cells where it
    is usable, in proportion to availability, and compare with clock time.

    A request is {"name", "hours", and either "ra"/"dec" or
    "ra_range"/"dec_range" (deg), optional "sun", "min_elevation",
    "commensal_group", "telescope"}. Requests for another telescope are
    skipped. Within a commensal group only the per-cell maximum counts
    (commensal observations share the same time). Requests with no position
    are not spread over the sky: their hours are reported as "unplaced".
    """
    year_start = year_start or _default_year_start()
    g = _grid(telescope, year_start)
    t_norm = telescope.lower().replace("ska-", "").replace("ska_", "")
    clock = _cells(g, np.ones_like(g["lst_h"], dtype=bool))
    supply = clock * (1 - SCHEDULING["maintenance_fraction"])

    demand = np.zeros_like(clock)
    groups: dict[str, np.ndarray] = {}
    never, used, unplaced = [], [], []
    for req in requests:
        if req.get("telescope") and req["telescope"].lower().replace("ska-", "") != t_norm:
            continue
        if not {"ra", "ra_range", "gal_b_range"} & req.keys():
            unplaced.append({"name": req.get("name", "?"), "hours": float(req["hours"]),
                             "position_note": req.get("position_note")})
            continue
        min_el = req.get("min_elevation", SCHEDULING["default_min_elevation_deg"])
        avail = np.mean([_cells(g, _usable(g, ra, dec, min_el, req.get("sun", "any")))
                         for ra, dec in _request_positions(req)], axis=0)
        if avail.sum() == 0:
            never.append(req.get("name", "?"))
            continue
        share = float(req["hours"]) * avail / avail.sum()
        used.append(req.get("name", "?"))
        if req.get("commensal_group"):
            key = req["commensal_group"]
            groups[key] = np.maximum(groups.get(key, 0), share)
        else:
            demand += share
    for share in groups.values():
        demand += share

    per_bin_demand, per_bin_supply = demand.sum(axis=0), supply.sum(axis=0)
    pressure = np.divide(per_bin_demand, per_bin_supply, out=np.zeros_like(per_bin_demand),
                         where=per_bin_supply > 0)
    bins = [
        {"lst_bin": f"{_fmt_lst(i * SCHEDULING['lst_bin_h'])}-{_fmt_lst((i + 1) * SCHEDULING['lst_bin_h'])}",
         "demand_h": round(float(dm), 1), "supply_h": round(float(sp), 1), "pressure": round(float(p), 3)}
        for i, (dm, sp, p) in enumerate(zip(per_bin_demand, per_bin_supply, pressure))
    ]
    cell_pressure = np.divide(demand, supply, out=np.zeros_like(demand), where=supply > 0)
    return {
        "telescope": telescope,
        "n_requests_used": len(used),
        "total_demand_h": round(float(demand.sum()), 1),
        "total_supply_h": round(float(supply.sum()), 1),
        "lst_bins": bins,
        "most_pressured_bins": sorted(bins, key=lambda b: -b["pressure"])[:5],
        "month_by_lst_pressure": [[round(float(x), 3) for x in row] for row in cell_pressure],
        "never_visible": never,
        "unplaced_hours": round(sum(u["hours"] for u in unplaced), 1),
        "unplaced": unplaced,
        "assumptions": assumptions(SCHEDULING["default_min_elevation_deg"], "per request", year_start),
    }
