"""Structured requirement records and field-level quote checks, on quotes
from the AASKAII chapters (as PDF extraction leaves them)."""

from rag.requirements import check_fields, extract_draft, validate_record

# Guidi01, p. 14 (atlas requirement C-19).
C19 = ("In this example we show how we can detect and resolve the emission at SKA-Mid Band 5a and 5b, with 10 hours "
       "integration time, reaching a rms of about 1 . 1 and 1 . 3 \U0001d707 Jy/beam and achieving a peak SNR of about "
       "4-8 and 8-13 (depending from the disk inclination), respectively.")
# OSullivan01, p. 5 (atlas requirement E-03).
E03 = ("we consider a frequency resolution of ∼ 100 kHz and a bandwidth of ∼ 200 MHz (for context, the LoTSS RM "
       "Grid used 97.6 kHz channels and 48 MHz of bandwidth). Using the SKA-Low sensitivity calculator for AA4, an "
       "integration time of 8 hrs with a Briggs robust imaging weighting of -1 gives an rms sensitivity of ∼ 5 "
       "\U0001d707 Jy/beam at an angular resolution of ∼ 5 ′′ (2 to 3 times above the confusion noise).")
# Bianchi01, p. 17 (atlas requirement C-02).
C02 = ("Using the SKAO online sensitivity calculator in the AA* configuration in Band 5b, with 300 hours of observations "
       "we will obtain a spectral surfacebrightness sensitivity of 0.02 K with a synthesised beam-size of 8.17 ′′ "
       "at the central frequency of 11.85 GHz.")


def test_weighting_not_in_quote_is_inferred():
    # C-19: the atlas matched at robust -2, which the quote never states.
    record = {"weighting": {"value": "robust -2", "source_text": None, "inferred": False},
              "sensitivity": {"value": 1.1, "unit": "uJy/beam", "source_text": None, "inferred": False},
              "hours": {"value": 10, "unit": "h", "source_text": "10 hours", "inferred": False}}
    checks = check_fields(C19, record)
    assert checks["weighting"]["verdict"] == "inferred"
    assert checks["weighting"]["flag"] == "marked as stated, but the quote does not contain it"
    assert checks["sensitivity"]["verdict"] == "stated"  # "1.1 and 1.3 μJy/beam": shares the unit
    assert checks["hours"]["verdict"] == "stated"


def test_unstated_centre_frequency_is_inferred():
    # E-03: the 150 MHz centre came from the agent; the quote states a bandwidth.
    checks = check_fields(E03, {"frequency": {"value": 150, "unit": "MHz", "source_text": None, "inferred": True},
                                "bandwidth": {"value": 0.2, "unit": "GHz", "source_text": None, "inferred": False},
                                "weighting": {"value": "Briggs robust -1", "source_text": None, "inferred": False}})
    assert checks["frequency"]["verdict"] == "inferred" and "flag" not in checks["frequency"]
    assert checks["bandwidth"]["verdict"] == "stated"  # 0.2 GHz = "200 MHz"
    assert checks["weighting"]["verdict"] == "stated"


def test_extract_draft_finds_units():
    d = extract_draft(E03)
    assert d["bandwidth"]["value"] == 200 and d["bandwidth"]["unit"] == "MHz"
    assert d["channel_width"]["value"] == 100 and d["channel_width"]["unit"] == "kHz"
    assert d["weighting"]["value"] == "robust -1" and d["hours"]["value"] == 8
    assert d["angular_resolution"] == {"value": 5.0, "unit": "arcsec", "source_text": "5 ′′", "inferred": False}
    assert d["frequency"] is None  # 97.6 kHz is a channel width, not a centre frequency
    c = extract_draft(C02)
    assert (c["sensitivity"]["value"], c["sensitivity"]["unit"]) == (0.02, "K")
    assert c["array_stage"]["value"] == "AA*" and c["band"]["value"] == "Band 5b"
    assert c["frequency"]["value"] == 11.85 and c["hours"]["value"] == 300


def test_validate_record_rejects_bad_units_and_fields():
    problems = validate_record({"sensitivity": {"value": "1", "unit": "uJy/parsec"}, "colour": None,
                                "mode": {"value": "holography"}})
    assert "sensitivity: value must be a number" in problems
    assert any("uJy/parsec" in p for p in problems)
    assert "unknown field 'colour'" in problems and any(p.startswith("mode:") for p in problems)
    assert validate_record(extract_draft(C02)) == []
