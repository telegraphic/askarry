# Vendored from ska-sci-ops-setup-validator-backend (src/ska_sci_ops_setup_validator/telescope/receivers.py).
"""receivers.py - details on SKA-Mid receivers + SKA-Low frequencies."""

from astropy import units as u

# receiver details
rx_mid = {
    "Band 1": {
        "low": 0.35 * u.GHz,
        "high": 1.05 * u.GHz,
        "bw": 700 * u.MHz,
        "n_sub": 1,
        "reference": "SKAO-DISH_SRx_REQ-171",
    },
    "Band 2": {
        "low": 0.95 * u.GHz,
        "high": 1.76 * u.GHz,
        "bw": 810 * u.MHz,
        "n_sub": 1,
        "reference": "SKAO-DISH_SRx_REQ-172",
    },
    "Band 5a": {
        "low": 4.6 * u.GHz,
        "high": 8.5 * u.GHz,
        "bw": 3900 * u.MHz,
        "n_sub": 1,
        "reference": "SKAO-DISH_SRx_REQ-175",
    },
    "Band 5b": {
        "low": 8.3 * u.GHz,
        "high": 15.4 * u.GHz,
        "bw": 5000 * u.MHz,
        "n_sub": 1,  # WAR: actually 2x2500 MHz
        "reference": "SKAO-DISH_SRx_REQ-176",
    },
}

# SKA-Low receiver details
rx_low = {"SKALA": {"low": 50 * u.MHz, "high": 350 * u.MHz, "bw": 300 * u.MHz, "n_sub": 1}}

# Receiver fleet availability by array release
available_receivers = {
    "SKA-Mid": {
        "SV-AA2": ("Band 1", "Band 2", "Band 5a", "Band 5b"),
        "AASTAR": ("Band 1", "Band 2", "Band 5a", "Band 5b"),
        "AA4": ("Band 1", "Band 2", "Band 5a", "Band 5b"),
    },
    # Low only has SKALA antennas!
    "SKA-Low": {
        "SV-AA2": ("SKALA",),
        "AASTAR": ("SKALA",),
        "AA4": ("SKALA",),
    },
}

# Combined receiver dict
receiver_dict = {"SKA-Mid": rx_mid, "SKA-Low": rx_low}
