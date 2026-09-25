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
        {"name": "north", "ra": 180, "dec": 50, "hours": 100},
        {"name": "c1", "ra": 90, "dec": -30, "hours": 200, "commensal_group": "g"},
        {"name": "c2", "ra": 90, "dec": -30, "hours": 150, "commensal_group": "g"},
        {"name": "mid only", "ra": 0, "dec": -30, "hours": 999, "telescope": "mid"},
    ]
    out = scheduling.lst_pressure(reqs, "low", year_start=YEAR, years=1)
    assert out["never_visible"] == ["north"]
    assert out["total_demand_h"] == pytest.approx(500 + 2000 + 200)
    assert out["total_supply_h"] == pytest.approx(8760 * (1 - scheduling.SCHEDULING["maintenance_fraction"]), rel=1e-3)


def test_unknown_telescope_and_sun_mode():
    with pytest.raises(ValueError):
        scheduling.visibility_windows(0, 0, "atca", year_start=YEAR)
    with pytest.raises(ValueError):
        scheduling.visibility_windows(0, 0, "low", sun="dusk", year_start=YEAR)
