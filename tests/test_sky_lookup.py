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
