import pytest
from astropy import units as u

from rag.data_product_estimator import estimate_data_product_size
from rag.vendor.data_size.data_products import CalibratedVisibilities, ImageCutout, TransientDump


# ---------------------------------------------------------------------------
# Core formula regressions, ported from odp-data-size-tool's test_data_products.py,
# run directly against the vendored classes.
# ---------------------------------------------------------------------------

def test_image_cutout_volume_scales_with_n_cutout():
    kwargs = dict(
        resolution=(1 * u.arcsec).to(u.deg), fov=2 * u.deg,
        n_chan=1, n_stokes=1, n_products=1, n_beam=1, n_bit=32,
        n_timestep=1, n_psf_osamp=1, n_pix=100,
    )
    one = ImageCutout(n_cutout=1, **kwargs)
    five = ImageCutout(n_cutout=5, **kwargs)
    assert five.data_volume.to("TB").value == pytest.approx(5 * one.data_volume.to("TB").value)


def test_transient_dump_data_rate_treats_n_bit_as_bits():
    model = TransientDump(bw=1e8, n_station=1, n_pol=2, n_bit=2, n_dump=1)
    expected_tb_s = (2 * 1 * 1e8 * 2 * 2 * u.bit / u.s).to("TB/s").value
    assert model.data_rate.to("TB/s").value == pytest.approx(expected_tb_s)


def test_calibrated_vis_n_timestep_rounds_up_for_partial_integration():
    model = CalibratedVisibilities(
        n_station=1, t_int=1200, bw=1e8, cbw=1e5, n_stokes=1,
        duration=3599 * u.s, storage_model="msv2",
    )
    assert model.n_timestep == 3


# ---------------------------------------------------------------------------
# End-to-end round trip cases through estimate_data_product_size, one per
# product type, adapted from odp-data-size-tool's test_model_io.py.
# ---------------------------------------------------------------------------

def test_estimate_image():
    result = estimate_data_product_size("Image", {
        "resolution": "1.000 arcsec", "fov": "2.000 deg",
        "n_chan": 16, "n_stokes": 2, "n_products": 4,
        "n_beam": 1, "n_bit": 32, "n_timestep": 1, "n_psf_osamp": 3,
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] is None


def test_estimate_image_cutout():
    result = estimate_data_product_size("Image Cutout", {
        "resolution": "1.000 arcsec", "fov": "2.000 deg",
        "n_pix": 4096, "n_cutout": 1,
        "n_chan": 16, "n_stokes": 2, "n_products": 4,
        "n_beam": 1, "n_bit": 32, "n_timestep": 1, "n_psf_osamp": 1,
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] is None


def test_estimate_gridded_visibilities():
    result = estimate_data_product_size("Gridded Visibilities", {
        "resolution": "1.000 arcsec", "fov": "2.000 deg",
        "n_psf_osamp": 3, "n_chan": 16, "n_stokes": 4, "n_beam": 36,
        "n_stack": 1, "kappa": 2.1, "B_sparseness": 0.90,
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] is None


def test_estimate_pss():
    result = estimate_data_product_size("PSS", {
        "mode": "single", "duration": "3600.000 s",
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] == pytest.approx(0.0209 / 1024.0)


def test_estimate_pst_folded():
    result = estimate_data_product_size("PST Folded", {
        "n_chan": 4096, "n_stokes": 4, "n_beam": 1,
        "n_bit": 16, "n_phase_bin": 1024, "n_subint": 16,
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] is None


def test_estimate_dynamic_spectrum():
    result = estimate_data_product_size("Dynamic Spectrum", {
        "bw": "300.000 MHz", "cbw": "1.000 MHz", "t_int": 1.0,
        "n_beam": 1, "n_stokes": 1, "n_bit": 8, "duration": "3600.000 s",
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] > 0


def test_estimate_flowthrough():
    result = estimate_data_product_size("Flowthrough", {
        "bw": 300e6, "n_beam": 1, "osamp": 4 / 3,
        "n_pol": 2, "n_bit": 4, "duration": "3600.000 s",
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] > 0


def test_estimate_vlbi():
    result = estimate_data_product_size("VLBI", {
        "bw": 300e6, "n_beam": 1, "osamp": 4 / 3,
        "n_pol": 2, "n_bit": 4, "duration": "3600.000 s",
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] > 0


def test_estimate_transient_dump():
    result = estimate_data_product_size("Transient Dump", {
        "bw": 300e6, "n_station": 64, "n_pol": 2, "n_bit": 2, "n_dump": 1,
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] > 0


def test_estimate_calibrated_vis():
    result = estimate_data_product_size("Calibrated Vis", {
        "n_station": 64, "t_int": 1.42, "bw": 300e6, "cbw": 13400.0,
        "n_stokes": 1, "duration": "3600.000 s", "storage_model": "msv2",
        "include_uvw": True, "include_weights": True,
        "include_uncalibrated": False, "include_model": False,
    })
    assert result["data_volume_TB"] > 0
    assert result["data_rate_TBs"] is None


def test_unknown_product_type_raises_key_error():
    with pytest.raises(KeyError, match="Unknown product type"):
        estimate_data_product_size("NonExistentProduct", {})


def test_missing_required_field_raises():
    with pytest.raises(Exception):
        estimate_data_product_size("PSS", {"mode": "single"})  # no 'duration' key
