import numpy as np
import pytest

from rag.sky_lookup import _clean, to_skycoord


def test_clean_masked_and_numpy():
    assert _clean(np.ma.masked) is None
    assert _clean(float("nan")) is None
    assert _clean(np.float64(1.5)) == 1.5 and type(_clean(np.float64(1.5))) is float


def test_to_skycoord_parses_coordinates_without_network():
    assert to_skycoord("201.365 -43.019").ra.deg == pytest.approx(201.365)
    c = to_skycoord("13:25:27.6 -43:01:09")
    assert c.ra.deg == pytest.approx(201.365) and c.dec.deg == pytest.approx(-43.019167, abs=1e-5)


# --- Display name and confidence (SIMBAD stubbed: no network) --------------
import astroquery.simbad

import rag.sky_lookup as sl

CYG_A = ["MCG+07-41-003", "NAME Cygnus A", "3C 405", "NAME Cyg A", "2MASX J19592807+4044022"]


def test_display_name_prefers_common_name_over_main_id():
    assert sl.display_name("MCG+07-41-003", CYG_A) == "Cygnus A"
    assert sl.display_name("MCG+07-41-003", CYG_A, query="Cyg A") == "Cyg A"
    assert sl.display_name("M  31", ["NAME Andromeda Galaxy", "NGC   224"]) == "M 31"
    assert sl.display_name("2MASX J03194823+4130420", ["NGC 1275", "3C 84"]) == "NGC 1275"
    assert sl.display_name("Gaia DR3 123", ["Gaia DR3 123"]) == "Gaia DR3 123"


class _FakeSimbad:
    def query_object(self, name):
        return [{"main_id": "MCG+07-41-003", "otype": "Sy2", "rvz_redshift": 0.056,
                 "ra": 299.868, "dec": 40.734}] if "cyg" in name.lower() or "3c" in name.lower() else []


def test_confidence_by_resolution_path(monkeypatch):
    monkeypatch.setattr(sl, "_simbad", lambda: _FakeSimbad())
    monkeypatch.setattr(astroquery.simbad.Simbad, "query_objectids", staticmethod(lambda name: {"id": CYG_A}))
    exact = sl.resolve_object("Cygnus A")
    assert (exact["main_id"], exact["display_name"], exact["confidence"]) == ("MCG+07-41-003", "Cygnus A", 1.0)
    assert exact["name"] == "MCG+07-41-003"  # unchanged for existing callers
    assert sl.resolve_object("Cygnus-A radio galaxy")["confidence"] == 0.8
    monkeypatch.setattr(sl, "ned_lookup", lambda name: {"found": True, "service": "NED", "name": "VV 1"})
    assert sl.resolve_object("VV 1")["confidence"] == 0.6
