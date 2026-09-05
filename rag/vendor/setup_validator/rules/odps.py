# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/rules/odps.py).
"""rules/odps.py - check ODPs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..obs_config import ObsConfig


from ..validation_schema import get_minmax


def check_number_images_is_valid(obs_config: ObsConfig) -> bool:
    """Check the overall number of images is valid (across subarrays).

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    n_min, n_max, _n_step = get_minmax(
        "odps/odp_config", "n_image_odps_per_subarray", telescope=obs_config.telescope, context=obs_config.context
    )

    n_min, n_max_zoom, _n_step = get_minmax(
        "odps/odp_config", "n_zoom_image_odps_per_subarray", telescope=obs_config.telescope, context=obs_config.context
    )

    for sa in obs_config.config.subarrays:
        n_image_cube_cont = 0
        n_image_cube_zoom = 0
        n_image_mfs_cont = 0
        n_image_mfs_zoom = 0
        for b in sa.beams:
            for o in b.odps:
                if o.type == "image_cube":
                    if o.settings.mode == "continuum":
                        n_image_cube_cont += 1
                    if o.settings.mode == "zoom":
                        n_image_cube_zoom += 1
                if o.type == "mfs_image":
                    if o.settings.mode == "continuum":
                        n_image_mfs_cont += 1
                    if o.settings.mode == "zoom":
                        n_image_mfs_zoom += 1

        if n_image_cube_cont > n_max:
            return (
                False,
                {"image_type": "Continuum image cube", "n_max": n_max, "n_image": n_image_cube_cont, "sa": sa.id},
            )
        if n_image_mfs_cont > n_max:
            return (
                False,
                {"image_type": "Continuum MFS image", "n_max": n_max, "n_image": n_image_mfs_cont, "sa": sa.id},
            )
        if n_image_cube_zoom > n_max_zoom:
            return (
                False,
                {"image_type": "Zoom image cube", "n_max": n_max_zoom, "n_image": n_image_cube_zoom, "sa": sa.id},
            )
        if n_image_mfs_zoom > n_max_zoom:
            return (
                False,
                {"image_type": "Zoom MFS image", "n_max": n_max_zoom, "n_image": n_image_mfs_zoom, "sa": sa.id},
            )

    return True


def check_odp_image_cube_channels_valid(obs_config: ObsConfig) -> bool:
    """Check the zoom channels are valid (compare against number of channels in mode).

    Args:
        obs_config (ObsConfig): ObsConfig object.

    Returns:
        bool: Pass (True) or fail (False).
    """
    for sa in obs_config.config.subarrays:
        for b in sa.beams:
            for o in b.odps:
                if o.type == "image_cube":
                    if o.settings.mode == "zoom":
                        if o.settings.n_channel_zoom < 1 or o.settings.n_channel_zoom > b.zoom.numberOfZoomChannels:
                            return (
                                False,
                                {
                                    "mode": "zoom",
                                    "n_chan_mode": b.zoom.numberOfZoomChannels,
                                    "n_chan": o.settings.n_channel_zoom,
                                    "sa": sa.id,
                                },
                            )
                    if o.settings.mode == "continuum":
                        n_chan_cont = int(b.continuum.bandwidth * 1e3 / b.continuum.channelWidth)
                        print("A", n_chan_cont)
                        if o.settings.n_channel < 1 or o.settings.n_channel > n_chan_cont:
                            return (
                                False,
                                {
                                    "mode": "continuum",
                                    "n_chan_mode": b.zoom.numberOfZoomChannels,
                                    "n_chan": n_chan_cont,
                                    "sa": sa.id,
                                },
                            )
    return True
