"""
Target visibility and LST pressure for SKA-Low / SKA-Mid.

visibility_windows  when one target is usable over a year: LST window above
                    the elevation limit, usable hours per month and LST bin
                    under a Sun constraint (night / avoid twilight / any)
lst_pressure        requested vs available hours per LST bin (and month ×
                    LST) for a set of observing requests, and the overall
                    demand-to-supply ratio

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
        # Below the dish limit nothing can be done; between it and min_elevation
        # the target is low but observable.
        "mechanical_min_elevation_deg": SCHEDULING["mechanical_min_elevation_deg"][telescope.lower().replace("ska-", "").replace("ska_", "")],
        "above_mechanical_limit": bool(max_el >= SCHEDULING["mechanical_min_elevation_deg"][
            telescope.lower().replace("ska-", "").replace("ska_", "")]),
        "lst_window_above_min_elevation": window,
        "usable_hours_per_year": round(float(cells.sum()), 1),
        "usable_hours_per_month": [round(float(x), 1) for x in cells.sum(axis=1)],
        "usable_hours_per_lst_bin": [round(float(x), 1) for x in cells.sum(axis=0)],
        "assumptions": assumptions(min_el, sun, year_start),
    }


def _assumed_dec_band(area_deg2: float, lat: float) -> tuple[float, float]:
    """A declination band of *area_deg2*, centred on the site's zenith
    declination and clipped to the sky: where a survey of that size with no
    stated position would most plausibly go."""
    target = np.radians(1) ** 2 * min(area_deg2, 41252.96) / (2 * np.pi)  # = sin(d_hi) - sin(d_lo)

    def band(h):
        lo, hi = max(-90.0, lat - h), min(90.0, lat + h)
        return lo, hi, np.sin(np.radians(hi)) - np.sin(np.radians(lo))

    a, b = 0.0, 180.0
    for _ in range(60):  # bisection on the half-width
        mid = (a + b) / 2
        a, b = (mid, b) if band(mid)[2] < target else (a, mid)
    lo, hi, _ = band(b)
    return float(lo), float(hi)


def _request_positions(req: dict, lat: float | None = None) -> list[tuple[float, float]]:
    """Point target, or a grid over an area request: ra_range/dec_range, a
    declination band (dec_range alone: all RA), gal_l_range/gal_b_range
    (e.g. the Galactic plane), or an area alone (area_deg2, placed on the
    band from _assumed_dec_band), in degrees."""
    if "gal_b_range" in req:
        from astropy.coordinates import SkyCoord

        (l0, l1), (b0, b1) = req.get("gal_l_range", [0, 360]), req["gal_b_range"]
        span = (l1 - l0) % 360 or 360
        ls = np.repeat([(l0 + span * (i + 0.5) / 36) % 360 for i in range(36)], 5)
        bs = np.tile(np.linspace(b0, b1, 5), 36)
        icrs = SkyCoord(l=ls * u.deg, b=bs * u.deg, frame="galactic").icrs
        return list(zip(icrs.ra.deg.tolist(), icrs.dec.deg.tolist()))
    if "dec_range" in req or "area_deg2" in req:
        (ra0, ra1) = req.get("ra_range", [0, 360])
        dec0, dec1 = req["dec_range"] if "dec_range" in req else _assumed_dec_band(req["area_deg2"], lat)
        span = (ra1 - ra0) % 360 or 360
        ras = [(ra0 + span * (i + 0.5) / 24) % 360 for i in range(24)]
        decs = np.linspace(dec0, dec1, 5)
        return [(r, float(dd)) for r in ras for dd in decs]
    return [(float(req["ra"]), float(req["dec"]))]


def _placed(req: dict) -> bool:
    return bool({"ra", "ra_range", "dec_range", "gal_b_range", "area_deg2"} & req.keys())


def lst_pressure(requests: list[dict], telescope: str, year_start: str | None = None,
                 years: float | None = None, min_elevation_deg: float | list[float] | None = None,
                 apply_sun_constraints: bool | None = None) -> dict:
    """Spread each request's hours over the (month, LST-bin) cells where it
    is usable, in proportion to availability, and compare with clock time.

    A request is {"name", "hours", and a position: "ra"/"dec", "ra_range"/
    "dec_range", "dec_range" alone (a declination band, all RA),
    "gal_l_range"/"gal_b_range", or "area_deg2" alone (placed on an assumed
    band, flagged); optional "sun", "min_elevation", "commensal_group",
    "telescope"}. Requests for another telescope are skipped. Within a
    commensal group only the per-cell maximum counts (commensal observations
    share time). Requests with no position are "unplaced": their hours count
    in the total demand-to-supply ratio but are not spread over LST.

    *min_elevation_deg* replaces the default elevation limit for requests
    without their own; a list gives one full result per value under
    "by_min_elevation". A target that never reaches its limit but does
    clear the telescope's mechanical limit is placed at the mechanical limit
    and listed under "elevation_relaxed"; only targets below the mechanical
    limit are "never_visible". Sun constraints ("night", "avoid_twilight")
    are ignored unless *apply_sun_constraints* (off by default for radio).
    Supply is *years* (default config planning_years) of clock time less the
    maintenance share, since requested hours are programme totals.
    """
    if isinstance(min_elevation_deg, (list, tuple)):
        runs = {f"{el:g}": lst_pressure(requests, telescope, year_start, years, el, apply_sun_constraints)
                for el in min_elevation_deg}
        first = next(iter(runs.values()))
        return {"telescope": telescope, "planning_years": first["planning_years"],
                "total_supply_h": first["total_supply_h"],
                "summary": {el: {k: r[k] for k in ("demand_supply_ratio", "total_demand_supply_ratio", "placed_h",
                                                     "total_demand_h_all")}
                            | {"n_elevation_relaxed": len(r["elevation_relaxed"]), "n_never_visible": len(r["never_visible"])}
                            for el, r in runs.items()},
                "by_min_elevation": runs}

    year_start = year_start or _default_year_start()
    g = _grid(telescope, year_start)
    t_norm = telescope.lower().replace("ska-", "").replace("ska_", "")
    clock = _cells(g, np.ones_like(g["lst_h"], dtype=bool))
    years = SCHEDULING["planning_years"] if years is None else years
    supply = clock * (1 - SCHEDULING["maintenance_fraction"]) * years
    sun_on = SCHEDULING["apply_sun_constraints"] if apply_sun_constraints is None else apply_sun_constraints
    default_el = SCHEDULING["default_min_elevation_deg"] if min_elevation_deg is None else float(min_elevation_deg)
    mech_el = SCHEDULING["mechanical_min_elevation_deg"][t_norm]

    demand = np.zeros_like(clock)
    groups: dict[str, np.ndarray] = {}
    unplaced_groups: dict[str, float] = {}
    never, used, unplaced, relaxed, assumed = [], [], [], [], []
    for req in requests:
        if req.get("telescope") and req["telescope"].lower().replace("ska-", "").replace("ska_", "") != t_norm:
            continue
        hours = float(req["hours"])
        if not _placed(req):
            unplaced.append({"name": req.get("name", "?"), "hours": hours, "position_note": req.get("position_note")})
            key = req.get("commensal_group") or f"__{len(unplaced)}"
            unplaced_groups[key] = max(unplaced_groups.get(key, 0.0), hours)
            continue
        positions = _request_positions(req, g["lat"])
        sun = req.get("sun", "any") if sun_on else "any"
        min_el = req.get("min_elevation", default_el)

        def availability(el):
            return np.mean([_cells(g, _usable(g, ra, dec, el, sun)) for ra, dec in positions], axis=0)

        avail = availability(min_el)
        if avail.sum() == 0 and min_el > mech_el:
            avail = availability(mech_el)
            if avail.sum() > 0:
                max_el = max(90 - abs(dec - g["lat"]) for _, dec in positions)
                relaxed.append({"name": req.get("name", "?"), "hours": hours, "min_elevation": min_el,
                                "max_elevation_deg": round(float(max_el), 2)})
        if avail.sum() == 0:
            never.append({"name": req.get("name", "?"), "hours": hours})
            continue
        if "area_deg2" in req and "dec_range" not in req:
            lo, hi = _assumed_dec_band(req["area_deg2"], g["lat"])
            assumed.append({"name": req.get("name", "?"), "hours": hours, "area_deg2": req["area_deg2"],
                            "assumed_dec_range": [round(lo, 1), round(hi, 1)]})
        share = hours * avail / avail.sum()
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
    placed_h = float(demand.sum())
    total_supply = float(supply.sum())
    # Everything requested (commensal groups counted once), placed or not:
    # the telescope's overall oversubscription.
    total_all = placed_h + sum(unplaced_groups.values()) + sum(n["hours"] for n in never)
    return {
        "telescope": telescope,
        "planning_years": years,
        "n_requests_used": len(used),
        "total_demand_h": round(placed_h, 1),  # placed demand only (as before)
        "placed_h": round(placed_h, 1),
        "total_demand_h_all": round(total_all, 1),
        "total_supply_h": round(total_supply, 1),
        "demand_supply_ratio": round(placed_h / total_supply, 3),
        "total_demand_supply_ratio": round(total_all / total_supply, 3),
        "lst_bins": bins,
        "most_pressured_bins": sorted(bins, key=lambda b: -b["pressure"])[:5],
        "month_by_lst_pressure": [[round(float(x), 3) for x in row] for row in cell_pressure],
        "never_visible": [n["name"] for n in never],
        "never_visible_hours": round(sum(n["hours"] for n in never), 1),
        "elevation_relaxed": relaxed,
        "footprint_assumed": assumed,
        "unplaced_hours": round(sum(x["hours"] for x in unplaced), 1),
        "unplaced": unplaced,
        "assumptions": {**assumptions(default_el, "per request" if sun_on else "any (sun constraints off)", year_start),
                        "mechanical_min_elevation_deg": mech_el, "apply_sun_constraints": sun_on},
    }
