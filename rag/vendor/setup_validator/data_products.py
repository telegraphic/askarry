# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/data_products.py).
"""Data product models for SKA science operations setup validation."""

from enum import Enum

import numpy as np
import param
from astropy.units import Quantity, Unit


class PssModes(str, Enum):
    """Enumeration of PSS (Pulsar Search) operating modes."""

    SINGLE = "single"
    FOLDED = "folded"


class DataProduct(param.Parameterized):
    """Base class for all data product models."""

    name = param.String(default="")
    description = param.String(default="")

    @property
    def data_volume(self):
        return Quantity("0 B")

    def report(self) -> None:
        print(f"Mode: {self.name}")
        dur = getattr(self, "duration", None)
        if dur is not None:
            print(f" Duration (hr): {dur}")
        dr = getattr(self, "data_rate", None)
        if dr is not None:
            try:
                print(f" Data Rate: {dr:.5f}")
            except Exception:
                print(f" Data Rate: {dr}")
        print(f" Data volume: {self.data_volume:.2f}")


class Image(DataProduct):
    """SDP to SRC image data product."""

    name = param.String(default="image")
    description = param.String(default="SDP to SRC image size")

    resolution = param.ClassSelector(class_=Quantity)
    fov = param.ClassSelector(class_=Quantity)
    n_chan = param.Integer(default=1, bounds=(1, None))
    n_stokes = param.Integer(default=1, bounds=(1, 4))
    n_products = param.Integer(default=1, bounds=(1, None))
    n_beam = param.Integer(default=1, bounds=(1, None))
    n_bit = param.Integer(default=32)
    n_timestep = param.Integer(default=1)
    n_psf_osamp = param.Integer(default=3)

    @property
    def n_pix(self) -> int:
        return int((self.fov / self.resolution)) * self.n_psf_osamp

    @property
    def data_volume(self):
        image_size_B = (self.n_pix**2) * (self.n_bit / 8)
        image_cube_B = self.n_beam * self.n_stokes * self.n_chan * image_size_B * self.n_products * self.n_timestep
        total_B = image_cube_B
        vol = Quantity(total_B, Unit("B")).to("TB")
        if vol < 0.1 * Unit("TB"):
            vol = vol.to("GB")
        return vol

    def report(self) -> None:
        print(f"Data product: {self.name}")
        print(f"N_pix: {self.n_pix} x {self.n_pix}")
        print(f"Data volume: {self.data_volume:.2f}")


class ImageCutout(Image):
    """SDP to SRC image cutout data product."""

    name = param.String(default="image_cutout")
    description = param.String(default="SDP to SRC image size")

    n_pix = param.Integer(default=1, bounds=(1, None))
    n_cutout = param.Integer(default=1)


class ZoomImage(Image):
    """Image data product from zoom spectral mode."""

    name = param.String(default="zoom_image")
    description = param.String(default="Image from zoom mode")


class ZoomImageCutout(ImageCutout):
    """Cutout from a zoom image data product."""

    name = param.String(default="zoom_cutout")
    description = param.String(default="Cutout from zoom image")


class PulsarSearchProduct(DataProduct):
    """PSS to SDP pulsar search data product."""

    name = param.String(default="pss_product")
    description = param.String(default="PSS to SDP data rate")

    mode = param.ObjectSelector(default=PssModes.SINGLE, objects=[PssModes.SINGLE, PssModes.FOLDED])
    duration = param.ClassSelector(class_=Quantity)

    @property
    def data_rate(self):
        if self.mode == PssModes.SINGLE:
            data_rate = 0.0209 / 1024.0 * Unit("TB/s")
        elif self.mode == PssModes.FOLDED:
            data_rate = 0.0839 / 1024.0 * Unit("TB/s")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return data_rate

    @property
    def data_volume(self):
        return (self.data_rate * self.duration.to("s")).to("TB")


class FoldedPulsarArchive(DataProduct):
    """PST to SDP pulsar timing data product."""

    name = param.String(default="pst_product")
    description = param.String(default="PST to SDP data rate")

    n_bit = param.Integer(default=16)
    n_chan = param.Number(default=4096)
    n_stokes = param.Integer(default=4)
    n_beam = param.Integer(default=1, bounds=(1, None))
    n_phase_bin = param.Integer(default=128, bounds=(1, 100000))
    n_subint = param.Integer(default=16, bounds=(1, 100000))

    @property
    def data_volume(self):
        dv = self.n_beam * self.n_bit * self.n_stokes * self.n_phase_bin * self.n_subint * self.n_chan * Unit("bit")
        return dv.to("TB")


class DetectedFilterbank(DataProduct):
    """PST beam dynamic spectra data product."""

    name = param.String(default="dynamic_spectrum")
    description = param.String(default=" PST beam dynamic spectra data.")

    bw = param.Number()
    cbw = param.Number(default=2.0, bounds=(2, None))  # fixed default
    t_int = param.Number()
    n_beam = param.Integer(default=1, bounds=(1, 16))
    n_stokes = param.Integer(default=1, bounds=(1, 4))
    n_bit = param.Integer(default=8)
    duration = param.ClassSelector(class_=Quantity)

    @property
    def n_chan(self) -> int:
        return int(np.ceil(np.round(self.bw / self.cbw, 3)))

    @property
    def data_rate(self):
        dv = self.n_beam * self.n_chan * self.n_stokes * self.n_bit / self.t_int
        return (dv * Unit("b/s")).to("TB/s")

    @property
    def data_volume(self):
        return (self.data_rate * self.duration.to("s")).to("TB")

    @classmethod
    def from_pst_beam(cls, beam: dict, n_beam: int = 1, n_bit: int = 8) -> "DetectedFilterbank":
        """Create a DetectedFilterbank from a frontend pstSettings beam dict.

        Args:
            beam: A single entry from ``pstSettings.beamsList`` with keys
                  ``bandwidth`` (MHz), ``polarizations``, ``timeResolution``,
                  and ``frequencyResolution``.
            n_beam: Number of PST beams (default 1).
            n_bit: Bits per sample (default 8).

        Returns:
            DetectedFilterbank with ``bw`` and ``cbw`` both in Hz.
        """
        bw_hz = Quantity(beam["bandwidth"], "MHz").to("Hz").value
        freq_res = beam["frequencyResolution"]
        cbw_hz = Quantity(freq_res["value"], freq_res["unit"]).to("Hz").value
        time_res = beam["timeResolution"]
        t_int_s = Quantity(time_res["value"], time_res["unit"]).to("s").value
        n_stokes = len(beam.get("polarizations", ["I"]))
        n_bit = beam.get("bitDepth", n_bit)
        return cls(bw=bw_hz, cbw=cbw_hz, t_int=t_int_s, n_beam=n_beam, n_stokes=n_stokes, n_bit=n_bit)

    def check_data_rate(self, context: str, telescope: str) -> tuple[bool, Quantity, Quantity]:
        """Validate the computed data rate against the pst_dynamic_spectrum schema.

        Args:
            context: Observation context string, e.g. ``"Cycle 0"``.
            telescope: Telescope type string, e.g. ``"SKA-Mid"``.

        Returns:
            Tuple of ``(is_valid, computed_rate, max_rate)`` where both rates
            are astropy Quantities in GB/s (equivalent to Gbyte/s in the schema).
        """
        from .validation_schema import get_minmax

        _, max_val, _ = get_minmax(
            "odps/pst_dynamic_spectrum",
            "data_rate",
            context=context,
            telescope=telescope,
        )
        max_rate = Quantity(max_val, "GB/s")
        computed = self.data_rate.to("GB/s")
        return bool(computed <= max_rate), computed, max_rate


class FlowthroughArchive(DataProduct):
    """PST beam flowthrough data product."""

    name = param.String(default="flowthrough_product")
    description = param.String(default=" PST beam flowthrough.")

    bw = param.Number()
    n_beam = param.Integer(default=1, bounds=(1, None))
    osamp = param.Number(default=1.0)
    n_pol = param.Integer(default=2, bounds=(1, 2))
    n_bit = param.Integer(default=4, bounds=(1, 32))
    duration = param.ClassSelector(class_=Quantity)

    @param.depends("n_bit", watch=True)
    def _validate_n_bit_even(self) -> None:
        if self.n_bit % 2 != 0:
            raise ValueError("n_bit must be a multiple of 2")

    @property
    def data_rate(self):
        dv = 2 * self.n_beam * self.bw * self.n_pol * self.n_bit * self.osamp
        return (dv * Unit("b/s")).to("TB/s")

    @property
    def data_volume(self):
        return (self.data_rate * self.duration.to("s")).to("TB")

    @classmethod
    def from_pst_beam(cls, beam: dict, n_beam: int = 1) -> "FlowthroughArchive":
        """Create a FlowthroughArchive from a frontend pstSettings beam dict.

        Args:
            beam: A single entry from ``pstSettings.beamsList`` with keys
                  ``bandwidth`` (MHz), ``polarizations``, and ``bitDepth``.
            n_beam: Number of PST beams (default 1).

        Returns:
            FlowthroughArchive with ``bw`` in Hz.
        """
        bw_hz = Quantity(beam["bandwidth"], "MHz").to("Hz").value
        n_pol = len(beam.get("polarizations", ["X"]))
        n_bit = beam.get("bitDepth", 4)
        return cls(bw=bw_hz, n_pol=n_pol, n_bit=n_bit, n_beam=n_beam)

    def check_data_rate(self, context: str, telescope: str) -> tuple[bool, Quantity, Quantity]:
        """Validate the computed data rate against the pst_flowthrough schema.

        Args:
            context: Observation context string, e.g. ``"Cycle 0"``.
            telescope: Telescope type string, e.g. ``"SKA-Mid"``.

        Returns:
            Tuple of ``(is_valid, computed_rate, max_rate)`` where both rates
            are astropy Quantities in GB/s.
        """
        from .validation_schema import get_minmax

        _, max_val, _ = get_minmax(
            "odps/pst_flowthrough",
            "data_rate",
            context=context,
            telescope=telescope,
        )
        max_rate = Quantity(max_val, "GB/s")
        computed = self.data_rate.to("GB/s")
        return bool(computed <= max_rate), computed, max_rate


class VlbiProduct(FlowthroughArchive):
    """VLBI data product."""

    name = param.String(default="vlbi_product")
    description = param.String(default=" VLBI data product")


class TransientDump(DataProduct):
    """Transient buffer dump data product."""

    name = param.String(default="transient_dump")
    description = param.String(default=" Transient buffer dump")

    bw = param.Number()
    n_station = param.Integer(default=1, bounds=(1, None))
    n_pol = param.Integer(default=2, bounds=(1, 2))
    n_bit = param.Integer(default=2, bounds=(1, 8))
    n_dump = param.Integer(default=1)

    @param.depends("n_bit", watch=True)
    def _validate_n_bit_even(self) -> None:
        if self.n_bit % 2 != 0:
            raise ValueError("n_bit must be a multiple of 2")

    @property
    def data_rate(self):
        dr = 2 * self.n_station * self.bw * self.n_pol * self.n_bit
        return (dr * Unit("B/s")).to("TB/s")

    @property
    def data_volume(self):
        data_vol_B = self.data_rate.to("B/s").value * self.n_dump
        return (data_vol_B * Unit("B")).to("GB")


class CalibratedVisibilities(DataProduct):
    """Calibrated visibility data product."""

    name = param.String(default="calibrated_visibilities")
    description = param.String(default="Calibrated visibility data product")

    n_station = param.Integer(default=1, bounds=(1, None))
    t_int = param.Number()
    bw = param.Number()
    cbw = param.Number()
    n_stokes = param.Integer(default=1, bounds=(1, 4))
    duration = param.ClassSelector(class_=Quantity)
    storage_model = param.ObjectSelector(default="msv2", objects=["raw", "msv2", "msv4"])

    include_weights = param.Boolean(default=True, doc="Include per-channel weights")
    include_uvw = param.Boolean(default=True, doc="Include UVW coordinate array")
    include_uncalibrated = param.Boolean(default=False, doc="Include a copy of uncalibrated visibilities")
    include_model = param.Boolean(default=False, doc="Include model visibilities")

    @property
    def n_bls(self) -> float:
        return self.n_station * (self.n_station + 1) / 2

    @property
    def n_chan(self) -> int:
        return int(np.ceil(np.round(self.bw / self.cbw, 3)))

    @property
    def n_timestep(self) -> int:
        """Number of integration timesteps across the configured duration."""
        return int(self.duration.to("s").value / self.t_int)

    @property
    def data_volume(self):
        T = self.n_timestep
        B = int(self.n_bls)
        F = self.n_chan
        P = self.n_stokes

        # RAW (DATA only)
        if self.storage_model == "raw":
            dvol = B * T * F * P * 8
            if self.include_weights:
                dvol += B * T * F * P * 4
            if self.include_uvw:
                dvol += B * T * 3 * 4
            if self.include_uncalibrated:
                dvol += B * T * F * P * 8
            if self.include_model:
                dvol += B * T * F * P * 8
            return (dvol * Unit("B")).to("TB")

        # MSv2 (your existing formula)
        if self.storage_model == "msv2":
            n_rows = B * T
            bytes_per_row = (
                (7 * 8)  # TIMES + UVW (MSv2 spec)
                + (4 * 11)  # 11 Int keys
                + (8)  # TIME_EXTRA_PREC
                + (1)  # FLAG_ROW
                + P
                * (
                    (4 + 4)  # WEIGHT + SIGMA
                    + (8 * F)  # DATA Complex
                    + (4 * F)  # WEIGHT_SPECTRUM
                    + (1 * F)  # FLAG
                )
            )

            if not self.include_uvw:
                bytes_per_row -= 3 * 8

            if not self.include_weights:
                bytes_per_row -= P * (4 + 4 * F)

            if self.include_uncalibrated:
                bytes_per_row += (8 * F) * P

            if self.include_model:
                bytes_per_row += (8 * F) * P

            return (n_rows * bytes_per_row * Unit("B")).to("TB")

        # MSv4 (XRADIO schema)
        # Shapes from schema:
        #  VISIBILITY[T, B, F, P]  <c8 or <c16  (use 8B default)
        #  FLAG[T, B, F, P]        |b1 or small int (use 1B)
        #  WEIGHT[T, B, F, P]      <f2/<f4/<f8 (float32 -> 4B)
        #  UVW[T, B, 3]            <f2/<f4/<f8 (float32 -> 4B)
        if self.storage_model == "msv4":
            bytes_vis = 8  # complex64 (<c8)
            bytes_flag = 1  # bool
            bytes_wt = 4  # float32
            bytes_uvw = 4  # float32

            total = 0
            # always count VISIBILITY and FLAG (core arrays per schema)
            total += T * B * F * P * bytes_vis
            total += T * B * F * P * bytes_flag

            # include weights if requested (MSv4 WEIGHT has same 4D shape as VISIBILITY)
            if self.include_weights:
                total += T * B * F * P * bytes_wt

            # include UVW if requested (MSv4 UVW is [T,B,3])
            if self.include_uvw:
                total += T * B * 3 * bytes_uvw

            if self.include_uncalibrated:
                total += T * B * F * P * bytes_vis

            if self.include_model:
                total += T * B * F * P * bytes_vis

            return (total * Unit("B")).to("TB")


class GriddedVisibilities(DataProduct):
    """Storage model for the gridded-visibility data product described by Rozgonyi (2021).

    Section 2.2–2.3 (minimum viable data model + long-term storage estimation).

    It implements:
        V_bytes = κ * ν_vis * (N_pol * N_chan * N_beam * N_cell * (1 - B)) * (Δt / T_stack)

    Key references and assumptions:
    - W-projection is used (not IDG or w-stacking).
    - We store three grids per channel: V_grid (complex), W_grid (real or complex),
      and %0 (preconditioning grid, complex). κ captures their combined contribution
      and typical relative sparsity; κ ≈ 2.1 (minimum) or κ ≈ 1.6 (improved, real-only W_grid).
    - B is the grid sparseness (fraction of zero uv-cells). (1 - B) is the non-zero fraction
      actually stored (sparse representation). The thesis advocates sparse storage and uses
      (1 - B) explicitly in Table 2.1.
    """

    name = param.String(default="gridded-visibilities")
    description = param.String(default="Sparse gridded visibility data product volume")

    # --- Image dimensions (same convention as Image class) ---
    resolution = param.ClassSelector(class_=Quantity, doc="Angular resolution (astropy Quantity angle)")
    fov = param.ClassSelector(class_=Quantity, doc="Field of view (astropy Quantity angle)")
    n_psf_osamp = param.Integer(default=3, bounds=(1, 100), doc="PSF oversampling factor")

    # --- Spectral / polarisation ---
    n_chan = param.Integer(default=1, bounds=(1, 32768), doc="spectral channels")
    n_stokes = param.Integer(default=4, bounds=(1, 4), doc="number of coherency products (e.g., 4 for full-Stokes)")
    n_beam = param.Integer(default=36, bounds=(1, None), doc="number of formed beams")
    n_stack = param.Integer(default=1, bounds=(1, None), doc="number of stacked observation blocks to store")

    # --- Storage model parameters ---
    kappa = param.Number(
        default=2.1, bounds=(1, 3), doc="scaling factor for (V_grid, W_grid, %0); 2.1 minimum model, 1.6 improved"
    )
    B_sparseness = param.Number(
        default=0.90, bounds=(0, 1), doc="sparseness B (fraction of zero uv-cells). (1-B) is stored fraction"
    )

    @property
    def n_pix(self) -> int:
        """Number of pixels across one side of the image grid."""
        return int((self.fov / self.resolution)) * self.n_psf_osamp

    @property
    def n_cell(self) -> int:
        return self.n_pix**2

    @property
    def nonzero_fraction(self) -> float:
        """Stored (non-zero) cell fraction = (1 - B)."""
        return float(1.0 - self.B_sparseness)

    @property
    def volume_bytes_single_block(self) -> int:
        """Compute bytes for ONE observation block (e.g., one 'night').

        V_bytes = κ * ν_vis * (N_pol * N_chan * N_beam * N_cell * (1 - B)) * (Δt / T_stack)
        """
        kappa = float(self.kappa)

        bytes_per_complex = 8
        term_count = int(self.n_stokes) * int(self.n_chan) * int(self.n_beam) * self.n_cell

        # Core expression from the thesis (Section 2.3.2 / Table 2.1).
        v_bytes = kappa * bytes_per_complex * term_count * self.nonzero_fraction
        return int(np.ceil(v_bytes))

    @property
    def data_volume(self):
        """Total volume = per-block volume * n_blocks.

        Returned as astropy Quantity, auto-scaled to TB (or GB if < 0.1 TB).
        """
        total_bytes = self.volume_bytes_single_block * int(self.n_stack)
        vol = Quantity(total_bytes, Unit("B")).to("TB")
        if vol < 0.1 * Unit("TB"):
            vol = vol.to("GB")
        return vol

    def report(self) -> None:
        print(f"Data product     : {self.name}")
        print(f"Description      : {self.description}")
        print(f"uv grid (N_pix²) : {self.n_pix} × {self.n_pix}  (N_cell={self.n_cell:,})")
        print(f"N_stokes × N_chan: {self.n_stokes} × {self.n_chan}")
        print(f"Beams            : {self.n_beam}")
        print(f"Sparseness B     : {self.B_sparseness:.4f}  → stored fraction (1-B)={self.nonzero_fraction:.4f}")
        print(f"κ (kappa)        : {self.kappa}")
        print(f"Blocks stored    : {self.n_stack}")
        per_block = Quantity(self.volume_bytes_single_block, Unit("B")).to("TB")
        if per_block < 0.1 * Unit("TB"):
            per_block = per_block.to("GB")
        print(f"Per-block volume : {per_block:.3f}")
        print(f"Total volume     : {self.data_volume:.3f}")


__all__ = [
    "PssModes",
    "DataProduct",
    "Image",
    "ImageCutout",
    "ZoomImage",
    "ZoomImageCutout",
    "GriddedVisibilities",
    "PulsarSearchProduct",
    "FoldedPulsarArchive",
    "DetectedFilterbank",
    "FlowthroughArchive",
    "VlbiProduct",
    "TransientDump",
    "CalibratedVisibilities",
]
