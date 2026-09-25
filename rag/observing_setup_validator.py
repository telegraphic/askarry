"""
Thin client for the SKAO observing-setup validator core engine, vendored from
ska-sci-ops-setup-validator-backend (see rag/vendor/setup_validator/).

Validates a telescope observing configuration (subarrays, beams, continuum/
zoom/PSS/PST mode settings, output data products) against the SKA validation
schema/rules for a given context (e.g. "Cycle 0", "SV-AA*") and telescope
("SKA-Low" or "SKA-Mid"). Same JSON shape as the ska-sci-ops-setup-validator
frontend GUI submits to its backend API.
"""

from __future__ import annotations


def validate_observing_setup(obs_config: dict) -> dict:
    """Validate an SKA telescope observing configuration.

    Args:
        obs_config: Dict with keys "context", "telescopeType", and "subarrays"
            (per-subarray template/band/mode settings and output data
            products) — see rag/vendor/setup_validator/schema/obs_config.yaml
            for allowed context/telescope values.

    Returns:
        dict with "status" ("success" or "error"), "message", and (on error)
        "alerts" — a list of human-readable validation failure messages.
    """
    from rag.vendor.setup_validator.frontend_utils import validate

    return validate(obs_config)


# Contexts accepted by the validator schema (schema/obs_config.yaml).
CONTEXTS = ["SV-AA2", "SV-AA*", "Cycle 0", "Cycle 1"]


def validate_across_contexts(obs_config: dict, contexts: list[str] | None = None) -> dict:
    """Validate the same observing configuration under each observing context,
    showing at which array-assembly stage/cycle a requirement becomes feasible.
    The obs_config's own "context" value is ignored."""
    return {
        context: validate_observing_setup({**obs_config, "context": context})
        for context in contexts or CONTEXTS
    }
