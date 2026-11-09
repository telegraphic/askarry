import math

import pytest
from astropy.utils import iers

from rag import scheduling

iers.conf.auto_download = False  # offline: built-in tables are ample for this precision
YEAR = "2027-01-01"


@pytest.mark.parametrize("dec, hours", [(-26.82, 6.77), (0.0, 5.01)])
def test_hours_above_45_deg_at_ska_low(dec, hours):
    out = scheduling.visibility_windows(0.0, dec, "low", year_start=YEAR)
    assert out["lst_window_above_min_elevation"]["hours_per_sidereal_day"] == pytest.approx(hours, abs=0.02)
    assert out["usable_hours_per_year"] == pytest.approx(hours * 366.25, rel=0.01)


def test_northern_target_never_reaches_45_deg():
    out = scheduling.visibility_windows(0.0, 30.0, "low", year_start=YEAR)
    assert out["never_reaches_min_elevation"] and out["usable_hours_per_year"] == 0


def test_night_constraint_is_seasonal():
    months = scheduling.visibility_windows(201.365, -43.019, "low", sun="night", year_start=YEAR)["usable_hours_per_month"]
    assert max(months[2:5]) > 150 and months[9] == 0  # Cen A: autumn nights yes, October no


def test_lst_pressure_conserves_hours_and_commensal_max():
    reqs = [
        {"name": "a", "ra": 201.4, "dec": -43.0, "hours": 500, "sun": "night"},
        {"name": "survey", "ra_range": [0, 360], "dec_range": [-60, 0], "hours": 2000},
        {"name": "north", "ra": 180, "dec": 70, "hours": 100},  # never above Low's horizon
        {"name": "low north", "ra": 180, "dec": 50, "hours": 40},  # peaks at 13 deg: placed, relaxed
        {"name": "c1", "ra": 90, "dec": -30, "hours": 200, "commensal_group": "g"},
        {"name": "c2", "ra": 90, "dec": -30, "hours": 150, "commensal_group": "g"},
        {"name": "mid only", "ra": 0, "dec": -30, "hours": 999, "telescope": "mid"},
    ]
    out = scheduling.lst_pressure(reqs, "low", year_start=YEAR, years=1)
    assert out["never_visible"] == ["north"]
    assert [r["name"] for r in out["elevation_relaxed"]] == ["low north"]
    assert out["total_demand_h"] == pytest.approx(500 + 2000 + 200 + 40)
    assert out["total_supply_h"] == pytest.approx(8760 * (1 - scheduling.SCHEDULING["maintenance_fraction"]), rel=1e-3)


def test_unknown_telescope_and_sun_mode():
    with pytest.raises(ValueError):
        scheduling.visibility_windows(0, 0, "atca", year_start=YEAR)
    with pytest.raises(ValueError):
        scheduling.visibility_windows(0, 0, "low", sun="dusk", year_start=YEAR)


# --- Demand/supply ratio, elevation limits, bands, conflicts -----------------
from rag import source_db as sd
from rag.scheduling import lst_pressure, visibility_windows


def test_total_demand_supply_ratio_counts_unplaced_hours():
    reqs = [{"name": "placed", "hours": 100, "ra": 0, "dec": -30},
            {"name": "unplaced", "hours": 900, "position_note": "no position stated"}]
    p = lst_pressure(reqs, "mid", year_start=YEAR, years=1)
    assert p["placed_h"] == pytest.approx(100, rel=1e-6)
    assert p["total_demand_h_all"] == pytest.approx(1000, rel=1e-6)
    assert p["total_demand_supply_ratio"] == pytest.approx(1000 / p["total_supply_h"], abs=1e-3)
    assert p["demand_supply_ratio"] == pytest.approx(100 / p["total_supply_h"], abs=1e-3)


def test_dec_band_request_placed_over_all_ra():
    p = lst_pressure([{"name": "southern HI", "hours": 240, "dec_range": [-90, 0]}], "mid", year_start=YEAR, years=1)
    assert p["n_requests_used"] == 1 and not p["unplaced"]
    demand = [b["demand_h"] for b in p["lst_bins"]]
    assert max(demand) < 1.2 * min(demand)  # all RA: spread evenly over LST


def test_area_only_request_placed_with_assumed_footprint():
    p = lst_pressure([{"name": "all-sky HI", "hours": 10000, "area_deg2": 20000}], "mid", year_start=YEAR, years=1)
    assert p["placed_h"] == pytest.approx(10000, rel=1e-6)
    (fa,) = p["footprint_assumed"]
    lo, hi = fa["assumed_dec_range"]
    # A band of 20,000 deg2 centred on Mid's zenith declination (-30.7 deg).
    area = 2 * math.pi * (math.sin(math.radians(hi)) - math.sin(math.radians(lo))) * (180 / math.pi) ** 2
    assert area == pytest.approx(20000, rel=0.01) and (lo + hi) / 2 == pytest.approx(-30.7, abs=0.5)


def test_cygnus_a_and_m31_visible_above_mid_mechanical_limit():
    for name, ra, dec, max_el in (("Cygnus A", 299.868, 40.734, 18.55), ("M31", 10.685, 41.269, 18.02)):
        w = visibility_windows(ra, dec, "mid", min_elevation=15, year_start=YEAR)
        assert w["max_elevation_deg"] == pytest.approx(max_el, abs=0.05)
        assert w["above_mechanical_limit"] and w["usable_hours_per_year"] > 0
        # 20 deg is above their maximum elevation from the SKA-Mid site.
        assert visibility_windows(ra, dec, "mid", min_elevation=20, year_start=YEAR)["usable_hours_per_year"] == 0


def test_never_visible_uses_mechanical_limit():
    reqs = [{"name": "M31", "hours": 50, "ra": 10.685, "dec": 41.269},     # max 18 deg: relaxed, placed
            {"name": "Polaris", "hours": 5, "ra": 37.95, "dec": 89.26}]    # below the horizon
    p = lst_pressure(reqs, "mid", year_start=YEAR, years=1)
    assert p["never_visible"] == ["Polaris"]
    assert [r["name"] for r in p["elevation_relaxed"]] == ["M31"] and p["placed_h"] == pytest.approx(50, rel=1e-6)


def test_min_elevation_list_returns_each():
    p = lst_pressure([{"name": "M31", "hours": 50, "ra": 10.685, "dec": 41.269}], "mid", year_start=YEAR,
                     years=1, min_elevation_deg=[15, 45])
    assert set(p["by_min_elevation"]) == {"15", "45"}
    assert p["summary"]["15"]["n_elevation_relaxed"] == 0 and p["summary"]["45"]["n_elevation_relaxed"] == 1


def test_sun_constraint_off_by_default():
    req = [{"name": "night only", "hours": 100, "ra": 0, "dec": -30, "sun": "night"}]
    off = lst_pressure(req, "mid", year_start=YEAR, years=1)
    on = lst_pressure(req, "mid", year_start=YEAR, years=1, apply_sun_constraints=True)
    assert off["assumptions"]["apply_sun_constraints"] is False
    assert off["month_by_lst_pressure"] != on["month_by_lst_pressure"]


def _db(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "find_quote", lambda quote, paper, book=None: {"source": f"/x/{paper}.pdf", "page_no": 1})
    monkeypatch.setattr(sd, "_live_lookup", lambda name: {"found": False})
    return sd.connect(tmp_path / "s.db")


def test_duplicate_hours_conflict_reported(tmp_path, monkeypatch):
    conn = _db(tmp_path, monkeypatch)
    a = sd.add_request(conn, "IM survey", 10000, "mid", "Wolz01", "q", position_note="none", extracted_by="r1")["id"]
    b = sd.add_request(conn, "IM survey", 5000, "mid", "Elahi01", "q", position_note="none", extracted_by="r1")["id"]
    sd.review_request(conn, a, "accepted", "ok")
    sd.review_request(conn, b, "duplicate", "same tier", duplicate_of=a)
    (c,) = sd.hour_conflicts(conn, "mid", "r1")
    assert c["hours"] == 10000 and c["restated"][0]["paper"] == "Elahi01" and c["restated"][0]["hours"] == 5000


def test_extraction_agreement_across_runs(tmp_path, monkeypatch):
    conn = _db(tmp_path, monkeypatch)
    for run, hours in (("r1", 100), ("r2", 105), ("r3", 300)):
        sd.add_request(conn, "Deep field", hours, "mid", "Hale01", "q", position_note="none", extracted_by=run)
    sd.add_request(conn, "Only once", 10, "mid", "Hale01", "q", position_note="none", extracted_by="r1")
    agree = {e["request"]: e for e in sd.extraction_agreement(conn, ["r1", "r2"], "mid")}
    assert agree["Deep field"]["agree"] and not agree["Only once"]["agree"]
    three = {e["request"]: e for e in sd.extraction_agreement(conn, ["r1", "r2", "r3"], "mid")}
    assert three["Deep field"]["hours_by_run"] == {"r1": 100, "r2": 105, "r3": 300} and not three["Deep field"]["agree"]


def test_survey_field_with_dec_band_or_area_places_requests(tmp_path, monkeypatch):
    conn = _db(tmp_path, monkeypatch)
    band = sd.add_survey_field(conn, "JingWang01", "q", description="southern sky", dec_range=[-90, 0])
    area = sd.add_survey_field(conn, "JingWang01", "q", description="20,000 deg2", area_deg2=20000)
    assert band["geometry"]["dec_range"] == [-90.0, 0.0] and area["geometry"]["area_deg2"] == 20000
    r = sd.add_request(conn, "All-sky HI", 10000, "mid", "JingWang01", "q", position_note="area only")["id"]
    sd.review_request(conn, r, "accepted", "area stated", field=area["field"])
    (req,) = sd.list_requests(conn, "mid")
    assert req["area_deg2"] == 20000 and "position_note" not in req


def test_area_in_position_note_is_placed(tmp_path, monkeypatch):
    conn = _db(tmp_path, monkeypatch)
    sd.add_request(conn, "All-sky HI survey", 10000, "mid", "JingWang01", "q",
                   position_note="unnamed area, sky position not stated, only area (20000 deg2)")
    sd.add_request(conn, "100 galaxies", 50, "mid", "JingWang01", "q", position_note="not individually named")
    reqs = sd.list_requests(conn, "mid")
    assert reqs[0]["area_deg2"] == 20000 and reqs[0]["area_from"] == "position_note"
    p = lst_pressure(reqs, "mid", year_start=YEAR, years=1)
    assert [f["name"] for f in p["footprint_assumed"]] == ["All-sky HI survey"]
    assert [u["name"] for u in p["unplaced"]] == ["100 galaxies"]
