# Vendored from ska-sci-ops/odp-data-size-tool (model_io.py); recalc_computed_values kept for parity, imports rewired to rag.vendor.data_size.
"""model_io.py — Round-trip helpers: model instance ↔ DataFrame row dict.

No Panel dependency — functions are straightforward to unit-test.

Complements project_io.py (JSON serialisation of the full app state) by
providing the layer that converts individual data product *instances* to
and from the flat row dicts stored inside each DataFrame.
"""

from astropy import units as u
from astropy.units import Quantity

from rag.vendor.data_size.data_products import (
    Image, ImageCutout, ZoomImage, ZoomImageCutout,
    GriddedVisibilities,
    PulsarSearchProduct, PssModes, PulsarTimingProduct,
    DynamicSpectrum, FlowthroughProduct, VlbiProduct,
    TransientDump, CalibratedVisibilities,
)
from rag.vendor.data_size.str_utils import (
    parse_qty,
    pretty, pretty_freq, pretty_time, pretty_rate, pretty_angle,
    auto_volume, md_escape,
    parse_pretty_rate_to_tb_s,    parse_volume_to_tb,)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_angle(v) -> Quantity:
    """Parse an angular string (e.g. ``'1.000 arcsec'``) to degrees."""
    return parse_qty(str(v)).to(u.deg)


def _parse_hz(v) -> float:
    """Return a bandwidth cell value in Hz.

    Accepts either a raw numeric Hz value or a ``pretty_freq`` string such as
    ``'300.000 MHz'``.
    """
    if isinstance(v, (int, float)):
        return float(v)
    return float(parse_qty(str(v)).to('Hz').value)


def _parse_duration(v) -> Quantity:
    """Parse a duration string (e.g. ``'3600.000 s'``) to an astropy Quantity."""
    return parse_qty(str(v), 's')


def _parse_t_int(v) -> float:
    """Return t_int in seconds, accepting either a raw float or a duration string."""
    if isinstance(v, (int, float)):
        return float(v)
    return float(parse_qty(str(v), 's').to('s').value)


def _parse_pss_mode(v) -> PssModes:
    """Accept a raw PssModes enum *or* its string value."""
    if isinstance(v, PssModes):
        return v
    return PssModes(str(v))


# ---------------------------------------------------------------------------
# model → row
# ---------------------------------------------------------------------------

# Columns in a project-dataframe row that hold row metadata rather than
# product-parameter data. Shared so the dashboard table and the printed
# report agree on what counts as "internal".
NON_DATA_COLS = ('project', 'summary', 'telescope')


def model_to_row(model, telescope: str, project: str, summary: str) -> dict:
    """Convert a data product model instance into a flat dict for a DataFrame row.

    Parameters
    ----------
    model : DataProduct
        Any concrete data product instance.
    telescope : str
        Telescope name (e.g. ``'SKA-Low'``).
    project : str
        Project name.
    summary : str
        Optional free-text project summary.

    Returns
    -------
    dict
        Flat dict whose values are primitives or formatted strings, suitable
        for direct construction of a one-row ``pd.DataFrame``.
    """
    row: dict = {'project': project, 'telescope': telescope}
    if summary:
        row['summary'] = summary
    row['name'] = getattr(model, 'name', '')

    for attr in model.param:
        if attr in ('name', 'description'):
            continue
        val = getattr(model, attr, None)
        if val is None:
            continue
        if isinstance(val, Quantity):
            try:
                val.to(u.deg)          # angular quantity?
                row[attr] = pretty_angle(val)
            except (u.UnitConversionError, TypeError):
                row[attr] = pretty(val)
        elif attr in ('bw', 'cbw') and isinstance(val, (int, float)):
            row[attr] = pretty_freq(val)
        elif attr == 't_int' and isinstance(val, (int, float)):
            row[attr] = pretty_time(float(val))
        else:
            row[attr] = val

    # Also include select computed properties (not Param parameters).
    for attr in ('n_pix', 'n_chan', 'n_bls', 'n_timestep'):
        if attr in row:
            continue
        try:
            val = getattr(model, attr)
            if attr == 'n_bls' and isinstance(val, (int, float)):
                row[attr] = int(val) if float(val).is_integer() else val
            else:
                row[attr] = val
        except Exception:
            continue

    # For Image-like and GriddedVisibilities, insert n_pix immediately after fov
    # so the column order in the displayed table is logical.
    if isinstance(model, (Image, GriddedVisibilities)) and 'fov' in row and 'n_pix' in row:
        reordered: dict = {}
        for key, val in row.items():
            reordered[key] = val
            if key == 'fov':
                reordered['n_pix'] = row['n_pix']
        row = reordered

    # data_volume — always stored as a TB string for consistent table totals.
    try:
        tb_val = float(model.data_volume.to('TB').value)
        dp = 3 if tb_val >= 0.001 else 6
        row['data_volume'] = f"{tb_val:.{dp}f} TB"
    except Exception:
        row['data_volume'] = pretty(model.data_volume)

    # data_rate — only present on products that define it.
    if hasattr(model, 'data_rate'):
        try:
            row['data_rate'] = pretty_rate(model.data_rate)
        except Exception:
            pass

    return row


# ---------------------------------------------------------------------------
# row → model
# ---------------------------------------------------------------------------

# Maps product-type key (tab name in the UI) to the default class.
_PTYPE_CLASS: dict = {
    'Image':                Image,
    'Image Cutout':         ImageCutout,
    'Gridded Visibilities': GriddedVisibilities,
    'PSS':                  PulsarSearchProduct,
    'PST Folded':           PulsarTimingProduct,
    'Dynamic Spectrum':     DynamicSpectrum,
    'Flowthrough':          FlowthroughProduct,
    'VLBI':                 VlbiProduct,
    'Transient Dump':       TransientDump,
    'Calibrated Vis':       CalibratedVisibilities,
}

# Within a ptype the stored ``name`` field can discriminate subclasses
# (e.g. ``'zoom_image'`` inside the ``'Image'`` tab).
_NAME_CLASS_OVERRIDE: dict = {
    'zoom_image':          ZoomImage,
    'zoom_cutout':         ZoomImageCutout,
    'flowthrough_product': FlowthroughProduct,
    'vlbi_product':        VlbiProduct,
}


def row_to_model(ptype: str, row: dict):
    """Reconstruct a data product model instance from a stored row dict.

    Parameters
    ----------
    ptype : str
        Product-type key (tab name in the UI), e.g. ``'Image'`` or ``'PSS'``.
        Must be one of the keys in :data:`_PTYPE_CLASS`.
    row : dict
        Row dict as produced by :func:`model_to_row` (or loaded from JSON and
        possibly normalised by ``_prettify_loaded_df``).

    Returns
    -------
    DataProduct
        A freshly instantiated model with parameters populated from *row*.

    Raises
    ------
    KeyError
        If *ptype* is not a recognised product-type key.
    """
    if ptype not in _PTYPE_CLASS:
        raise KeyError(f"Unknown product type: {ptype!r}")

    cls = _PTYPE_CLASS[ptype]

    # The stored 'name' field lets us pick the right subclass within a ptype
    # (e.g. ZoomImage vs Image, VlbiProduct vs FlowthroughProduct).
    name_val = str(row.get('name', ''))
    if name_val in _NAME_CLASS_OVERRIDE:
        cls = _NAME_CLASS_OVERRIDE[name_val]

    # ---- Image family -------------------------------------------------------
    if issubclass(cls, (ImageCutout, ZoomImageCutout)):
        return cls(
            resolution=_parse_angle(row['resolution']),
            fov=_parse_angle(row['fov']),
            n_pix=int(row['n_pix']),
            n_cutout=int(row['n_cutout']),
            n_chan=int(row['n_chan']),
            n_stokes=int(row['n_stokes']),
            n_products=int(row['n_products']),
            n_beam=int(row['n_beam']),
            n_bit=int(row['n_bit']),
            n_timestep=int(row['n_timestep']),
            n_psf_osamp=int(row['n_psf_osamp']),
        )

    if issubclass(cls, Image):   # Image or ZoomImage (not cutout)
        return cls(
            resolution=_parse_angle(row['resolution']),
            fov=_parse_angle(row['fov']),
            n_chan=int(row['n_chan']),
            n_stokes=int(row['n_stokes']),
            n_products=int(row['n_products']),
            n_beam=int(row['n_beam']),
            n_bit=int(row['n_bit']),
            n_timestep=int(row['n_timestep']),
            n_psf_osamp=int(row['n_psf_osamp']),
        )

    # ---- Gridded Visibilities -----------------------------------------------
    if cls is GriddedVisibilities:
        return cls(
            resolution=_parse_angle(row['resolution']),
            fov=_parse_angle(row['fov']),
            n_psf_osamp=int(row['n_psf_osamp']),
            n_chan=int(row['n_chan']),
            n_stokes=int(row['n_stokes']),
            n_beam=int(row['n_beam']),
            n_stack=int(row['n_stack']),
            kappa=float(row['kappa']),
            B_sparseness=float(row['B_sparseness']),
        )

    # ---- Pulsar Search (PSS) ------------------------------------------------
    if cls is PulsarSearchProduct:
        return cls(
            mode=_parse_pss_mode(row['mode']),
            duration=_parse_duration(row['duration']),
        )

    # ---- Pulsar Timing (PST) ------------------------------------------------
    if cls is PulsarTimingProduct:
        # n_chan is param.Number (not Integer) in PST — keep as float.
        return cls(
            n_chan=float(row['n_chan']),
            n_stokes=int(row['n_stokes']),
            n_beam=int(row['n_beam']),
            n_bit=int(row['n_bit']),
            n_phase_bin=int(row['n_phase_bin']),
            n_subint=int(row['n_subint']),
        )

    # ---- Dynamic Spectrum ---------------------------------------------------
    if cls is DynamicSpectrum:
        return cls(
            bw=_parse_hz(row['bw']),
            cbw=_parse_hz(row['cbw']),
            t_int=_parse_t_int(row['t_int']),
            n_beam=int(row['n_beam']),
            n_stokes=int(row['n_stokes']),
            n_bit=int(row['n_bit']),
            duration=_parse_duration(row['duration']),
        )

    # ---- Flowthrough / VLBI (same constructor) ------------------------------
    if issubclass(cls, FlowthroughProduct):
        return cls(
            bw=_parse_hz(row['bw']),
            n_beam=int(row['n_beam']),
            osamp=float(row['osamp']),
            n_pol=int(row['n_pol']),
            n_bit=int(row['n_bit']),
            duration=_parse_duration(row['duration']),
        )

    # ---- Transient Dump -----------------------------------------------------
    if cls is TransientDump:
        return cls(
            bw=_parse_hz(row['bw']),
            n_station=int(row['n_station']),
            n_pol=int(row['n_pol']),
            n_bit=int(row['n_bit']),
            n_dump=int(row['n_dump']),
        )

    # ---- Calibrated Visibilities --------------------------------------------
    if cls is CalibratedVisibilities:
        return cls(
            n_station=int(row['n_station']),
            t_int=_parse_t_int(row['t_int']),
            bw=_parse_hz(row['bw']),
            cbw=_parse_hz(row['cbw']),
            n_stokes=int(row['n_stokes']),
            duration=_parse_duration(row['duration']),
            storage_model=str(row['storage_model']),
            include_uvw=bool(row['include_uvw']),
            include_weights=bool(row['include_weights']),
            include_uncalibrated=bool(row['include_uncalibrated']),
            include_model=bool(row['include_model']),
        )

    raise ValueError(f"No reconstruction handler for class {cls.__name__!r}")


# ---------------------------------------------------------------------------
# Computed-value recalculation on load
# ---------------------------------------------------------------------------

def recalc_computed_values(frames: dict) -> tuple:
    """Reconstruct models from loaded rows and verify computed values.

    For every row in *frames*, attempts to reconstruct a model via
    :func:`row_to_model`.  The reconstructed model's ``data_volume`` and
    ``data_rate`` (where present) are compared against the stored
    pretty-printed values; if either differs by more than 0.01 % the
    corrected value replaces the stored one in-place.

    Parameters
    ----------
    frames : dict
        Mapping of ``(telescope, project, product_type)`` to
        ``pd.DataFrame`` — the same structure as ``project_dataframes``
        in ``panel_app.py``.  DataFrames are mutated in-place.

    Returns
    -------
    warnings : list[str]
        Human-readable strings for rows where either stored value
        differed from the recalculated value (now corrected).
    load_errors : list[str]
        Human-readable strings for rows where model reconstruction
        failed entirely (stored values are left unchanged).
    """
    warnings: list = []
    load_errors: list = []
    for (tel, proj, ptype), df in frames.items():
        has_rate_col = 'data_rate' in df.columns
        has_vol_col = 'data_volume' in df.columns
        new_rates = list(df['data_rate']) if has_rate_col else None
        new_vols = list(df['data_volume']) if has_vol_col else None
        for i, (idx, row) in enumerate(df.iterrows()):
            safe_proj = md_escape(proj)
            row_label = f"**{safe_proj}** / {ptype} row {i + 1}"
            try:
                model = row_to_model(ptype, row.to_dict())
            except Exception as exc:
                load_errors.append(f"{row_label}: {exc}")
                continue

            # --- data_volume ---
            if has_vol_col:
                try:
                    new_vol_tb = float(model.data_volume.to('TB').value)
                    dp = 3 if new_vol_tb >= 0.001 else 6
                    new_vol_str = f"{new_vol_tb:.{dp}f} TB"
                    stored_vol = str(row['data_volume'])
                    if new_vol_str != stored_vol:
                        stored_vol_tb = parse_volume_to_tb(stored_vol)
                        if stored_vol_tb == 0.0 or abs(stored_vol_tb - new_vol_tb) / max(abs(new_vol_tb), 1e-30) > 1e-4:
                            warnings.append(
                                f"{row_label} data_volume: "
                                f"stored {stored_vol!r} → recalculated **{new_vol_str}**"
                            )
                            new_vols[i] = new_vol_str
                except Exception:
                    pass

            # --- data_rate ---
            if has_rate_col:
                try:
                    rate_qty = model.data_rate
                except AttributeError:
                    continue
                new_rate_str = pretty_rate(rate_qty)
                stored_rate = str(row['data_rate'])
                if new_rate_str == stored_rate:
                    continue
                try:
                    stored_tb = parse_pretty_rate_to_tb_s(stored_rate)
                    new_tb = float(rate_qty.to('TB/s').value)
                    rel_diff = abs(stored_tb - new_tb) / max(abs(new_tb), 1e-30)
                    if rel_diff > 1e-4:
                        warnings.append(
                            f"{row_label} data_rate: "
                            f"stored {stored_rate!r} → recalculated **{new_rate_str}**"
                        )
                        new_rates[i] = new_rate_str
                except Exception:
                    new_rates[i] = new_rate_str

        if has_rate_col:
            df['data_rate'] = new_rates
        if has_vol_col:
            df['data_volume'] = new_vols
    return warnings, load_errors
