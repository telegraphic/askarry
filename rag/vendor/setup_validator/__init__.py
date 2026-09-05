# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/__init__.py).
"""setup_validator - validate SKA telescope configurations.

Pared-back, framework-free vendor of the ska-sci-ops-setup-validator-backend
core validation engine (no FastAPI/pydantic/GUI). Entry point is
``frontend_utils.validate(obs_config_dict)`` — see rag/observing_setup_validator.py
for the thin wrapper used elsewhere in this repo.
"""
