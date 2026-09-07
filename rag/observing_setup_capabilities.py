"""
Capability introspection for the vendored SKA setup-validator schema
(rag/vendor/setup_validator/schema/). Answers "what are my options?"
questions (allowed subarray templates, min/max ranges, fixed values) by
reading the same validation schema the setup-validator rule engine enforces
— more authoritative than static capability documentation, since it's the
schema actually used to accept/reject real observing configurations.
"""

from __future__ import annotations

from pathlib import Path

from rag.vendor.setup_validator.frontend_defaults import get_defaults as _get_defaults
from rag.vendor.setup_validator.validation_schema import get_rules, load_validation_schema

_SCHEMA_ROOT = Path(__file__).resolve().parent / "vendor" / "setup_validator" / "schema"
_BASE_TELESCOPE_DIRS = {"ska_low": "SKA-Low", "ska_mid": "SKA-Mid"}


def _normalize(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_").replace("*", "star")


def list_capability_schemas(telescope: str = "ska_low") -> list[str]:
    """List schema names available to query for a telescope.

    Names are relative to schema/base/<telescope>/ with the .yaml suffix
    dropped, e.g. "subarray_config", "continuum_settings",
    "odps/calibrated_visibilities". Pass one of these as the `schema`
    argument to describe_schema().

    Args:
        telescope: "ska_low"/"SKA-Low" or "ska_mid"/"SKA-Mid".
    """
    telescope = _normalize(telescope)
    base_dir = _SCHEMA_ROOT / "base" / telescope
    if not base_dir.is_dir():
        raise ValueError(f"Unknown telescope {telescope!r}; expected one of {sorted(_BASE_TELESCOPE_DIRS)}")
    return sorted(str(p.relative_to(base_dir).with_suffix("")) for p in base_dir.rglob("*.yaml"))


def describe_schema(schema: str, context: str | None = None, telescope: str | None = None) -> dict:
    """Return per-parameter constraints for a schema, merged for a given context/telescope.

    Args:
        schema: Schema name, e.g. "subarray_config", "continuum_settings",
            "beam_config", "pss_settings", "pst_settings", "zoom_settings",
            "obs_config_container", "odps/calibrated_visibilities". See
            list_capability_schemas() for the full set for a telescope.
        context: Observing context, e.g. "base", "cycle_0", "cycle_1",
            "sv_aa2"/"SV-AA2", "sv_aastar"/"SV-AA*". None loads only the
            top-level (non-telescope-specific) schema, if one exists.
        telescope: "ska_low"/"SKA-Low" or "ska_mid"/"SKA-Mid". Required
            whenever context is given.

    Returns:
        dict mapping each parameter name to its constraint info, e.g.
        {"type": "oneof", "allowed_values": [...], "doc": "..."} or
        {"type": "minmax", "min": ..., "max": ..., "step": ..., "doc": "..."}
        or {"type": "equals", "ref_value": ..., "doc": "..."}.
    """
    tpl = load_validation_schema(schema, context=context, telescope=telescope)
    params = tpl.get("params") or {}
    docs = tpl.get("docs") or {}

    described = {}
    for param_id, spec in params.items():
        val_type, kws = spec[0], (spec[1] if len(spec) > 1 else {})
        entry = {"type": val_type, **kws}
        if param_id in docs:
            entry["doc"] = docs[param_id]
        described[param_id] = entry
    return described


def describe_rules(schema: str, context: str | None = None, telescope: str | None = None) -> dict:
    """Return the cross-field validation rules that apply to a schema.

    Unlike describe_schema() (per-parameter bounds), this explains *why* a
    combination of otherwise-individually-valid values can still fail — e.g.
    "number of PST beams across subarrays exceeds the maximum allowed".

    Args:
        schema: Schema name — see list_capability_schemas().
        context: Observing context, e.g. "cycle_0", "sv_aa2"/"SV-AA2".
        telescope: "ska_low"/"SKA-Low" or "ska_mid"/"SKA-Mid". Required
            whenever context is given.

    Returns:
        dict mapping rule name to {"description": ..., "errmsg": ..., and
        either "expr" (a numexpr expression over the schema's params) or
        "func" (a dotted Python function path evaluated against them)}.
    """
    return get_rules(schema, context=context, telescope=telescope)


def get_context_defaults(context: str) -> dict:
    """Return default/max values for continuum, PSS, and PST bandwidth, PST
    beam count, and zoom/spectral channel counts, per telescope and (for
    SKA-Mid) per receiver band, for a given observing context.

    This is the same data the setup-validator frontend GUI pre-fills its
    forms with — useful for constructing a new observing_setup from
    scratch rather than checking one that already exists.

    Args:
        context: Observing context, e.g. "base", "cycle_0", "cycle_1",
            "sv_aa2"/"SV-AA2", "sv_aastar"/"SV-AA*".

    Returns:
        dict with "skaMidDefaults" (per receiver band) and "skaLowDefaults".
    """
    return _get_defaults(context)
