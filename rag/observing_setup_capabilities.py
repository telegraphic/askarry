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

from rag.vendor.setup_validator.validation_schema import load_validation_schema

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
