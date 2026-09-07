"""Tests for rag.observing_setup_data_volume (obs_config -> data-volume bridge)."""

from pathlib import Path

import pytest
import simplejson

from rag.observing_setup_data_volume import estimate_setup_data_volume, list_odps

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "setup_validator"


def _load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / name) as fh:
        return simplejson.load(fh)


def test_list_odps_reports_known_and_missing_params():
    """frontend_pass.json has calibrated_visibilities, mfs_image, image_cube, pst_timing ODPs."""
    obs_config = _load_fixture("frontend_pass.json")

    entries = list_odps(obs_config)

    odp_types = {e["odp_type"] for e in entries}
    assert odp_types == {"calibrated_visibilities", "mfs_image", "image_cube", "pst_timing"}

    by_type = {e["odp_type"]: e for e in entries}

    # pst_timing is explicitly unsupported - no auto-mapping attempted.
    pst_timing = by_type["pst_timing"]
    assert pst_timing["ptype"] is None
    assert pst_timing["missing_params"] == []

    # calibrated_visibilities: bw/cbw/n_station derivable from the parent
    # continuum mode + subarray template; t_int and duration are not.
    cal_vis = by_type["calibrated_visibilities"]
    assert cal_vis["ptype"] == "Calibrated Vis"
    assert cal_vis["known_params"]["bw"] == pytest.approx(700e6)
    assert cal_vis["known_params"]["cbw"] == pytest.approx(13.4e3 * 5)  # channelWidth * n_frequency_averaging
    assert cal_vis["known_params"]["n_station"] == 143
    assert set(cal_vis["missing_params"]) == {"t_int", "duration"}

    # mfs_image: only n_stokes derivable (no n_channel field in this ODP's schema at all).
    mfs_image = by_type["mfs_image"]
    assert mfs_image["ptype"] == "Image"
    assert mfs_image["known_params"]["n_chan"] == 1
    assert mfs_image["known_params"]["n_stokes"] == 1
    assert set(mfs_image["missing_params"]) == {"resolution", "fov"}

    # image_cube: n_channel/n_polarization both present and derivable.
    image_cube = by_type["image_cube"]
    assert image_cube["ptype"] == "Image"
    assert image_cube["known_params"]["n_chan"] == 7
    assert set(image_cube["missing_params"]) == {"resolution", "fov"}


def test_list_odps_raises_on_invalid_config():
    obs_config = _load_fixture("frontend_pass.json")
    obs_config["context"] = ""

    with pytest.raises(ValueError):
        list_odps(obs_config)


def test_estimate_setup_data_volume_end_to_end():
    """Supplying the caller-only fields yields a positive total across all supported ODPs."""
    obs_config = _load_fixture("frontend_pass.json")

    extra_params = {
        "1/1/Calibrated visibilities 1": {
            "t_int": 0.849,  # raw correlator dump time (s) - not in the validator schema
            "duration": "3600.000 s",
        },
        "1/1/Image 1": {
            # Both mfs_image and image_cube ODPs share the label "Image 1" here.
            "resolution": "1.000 arcsec",
            "fov": "1.000 deg",
        },
    }

    result = estimate_setup_data_volume(obs_config, extra_params=extra_params)

    assert result["unsupported"] == ["pst_timing"]
    # Both "Image 1" ODPs (mfs_image + image_cube) and calibrated_visibilities resolve.
    assert len(result["odps"]) == 3
    assert result["unresolved"] == []
    assert result["total_data_volume_TB"] > 0
    for odp in result["odps"]:
        assert odp["data_volume_TB"] > 0


def test_estimate_setup_data_volume_reports_unresolved_when_extra_params_missing():
    obs_config = _load_fixture("frontend_pass.json")

    result = estimate_setup_data_volume(obs_config)  # no extra_params supplied

    assert result["odps"] == []
    assert result["total_data_volume_TB"] == 0.0
    unresolved_types = {u["odp_type"] for u in result["unresolved"]}
    assert unresolved_types == {"calibrated_visibilities", "mfs_image", "image_cube"}
    assert result["unsupported"] == ["pst_timing"]
