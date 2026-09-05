# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/logging.py).
"""Configure logging for the vendored validator."""

import logging

# ponytail: upstream uses ska_ser_logging.configure_logging() for SKA SER log
# formatting; that's an SKA-internal package not on public PyPI and irrelevant
# to validation logic, so we fall back to stdlib logging config here.
logging.basicConfig()
logger = logging.getLogger("ska-sci-ops-setup-validator")
logger.setLevel(logging.INFO)
