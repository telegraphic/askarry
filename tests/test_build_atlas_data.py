import importlib.util
from pathlib import Path

import pytest

import rag.source_db as sd

_spec = importlib.util.spec_from_file_location(
    "build_atlas_data", Path(__file__).resolve().parents[1] / "scripts" / "build_atlas_data.py")
bad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bad)


def test_time_demand_tiers_and_placement(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "_live_lookup", lambda name: {"found": False})
    monkeypatch.setattr(sd, "find_quote", lambda q, p, b=None: {"source": f"/x/AASKAII/C/{p}.pdf", "page_no": 1})
    conn = sd.connect(tmp_path / "s.db")
    add = lambda *a, **k: sd.add_request(conn, *a, extracted_by="run", **k)["id"]  # noqa: E731
    tier = add("wide", 10000, "mid", "P1", "q", position_note="area only")
    again = add("wide again", 5000, "mid", "P2", "q", position_note="area only")
    psr = add("psr", 30, "mid", "P3", "q", ra=0, dec=-30)
    mode = add("always-on", 8760, "low", "P4", "q", ra=0, dec=-30)
    sd.review_request(conn, tier, "accepted", "defining", commensal_group="Wide tier")
    sd.review_request(conn, again, "duplicate", "restatement", duplicate_of=tier)
    sd.review_request(conn, psr, "accepted", "target")
    sd.review_request(conn, mode, "rejected", "commensal mode")

    out = bad.time_demand(conn, "run")
    mid = out["telescopes"]["mid"]
    assert (mid["accepted_hours"], mid["placed_hours"], mid["unplaced_hours"]) == (10030, 30, 10000)
    assert mid["lst_pressure"]["total_demand_h"] == pytest.approx(30)
    assert out["telescopes"]["low"]["accepted_hours"] == 0
    assert out["tiers"] == [{"group": "Wide tier", "telescope": "mid", "defining_paper": "P1", "hours": 10000,
                             "placed": False, "field": None, "restated_by": [{"paper": "P2", "hours": 5000}]}]
    assert out["status"]["rejected"] == {"n": 1, "hours": 8760}


def test_quote_check_accepts_split_sentences_and_rejects_invented(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "verify_atlas_quotes", Path(__file__).resolve().parents[1] / "scripts" / "verify_atlas_quotes.py")
    vq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vq)
    chunks = ["The first sentence of the chapter is long enough to count.", "The second sentence sits in the next chunk of text."]
    monkeypatch.setattr(vq.source_db, "find_quote",
                        lambda q, p, b=None: {"page_no": 3} if any(q.lower() in c.lower() for c in chunks) else None)
    monkeypatch.setattr(vq.corpus_tools, "verify_quote", lambda *a, **k: {"verdict": "not_found"})
    row = {"paper": "P1", "page": 9, "quote": " ".join(chunks)}
    assert vq.check(row) == {**row, "page": 3, "verification": "verbatim"}
    assert vq.check({**row, "quote": "A sentence that no chapter contains, invented for the test."}) is None


@pytest.mark.parametrize("text, spans", [
    ("Band 2, 0.95-1.76 GHz", [[0.95, 1.76], [0.95, 1.76]]),
    (">15 GHz up to ~24 GHz (Band 6)", [[15.0, 15.0], [15.4, 24.0], [24.0, 24.0]]),
    ("144 MHz, ~50 MHz bandwidth", [[0.144, 0.144]]),
    ("1.31, 6.55 and 11.85 GHz", [[1.31, 1.31], [6.55, 6.55], [11.85, 11.85]]),
    ("50 MHz to 15.4 GHz", [[0.05, 15.4]]),
    ("Bands 1 and 5a", [[0.35, 1.05], [4.6, 8.5]]),
    ("HI 21 cm, >=8 MHz bandwidth", [[1.4286, 1.4286]]),
    ("SKA-Low and SKA-Mid Band 1", [[0.05, 0.35], [0.35, 1.05]]),
    ("wideband", None),
    ("10 MHz bandwidth", None),
    ("two or more bands separated by factor >1.7", None),
])
def test_freq_range(text, spans):
    assert bad.freq_range(text) == spans


@pytest.mark.parametrize("name, cats", [
    ("PSR J0437-4715", ["PST"]),
    ("SKA-Mid AA4 pulsar search of all Galactic globular clusters", ["PSS"]),
    ("All-sky HI survey", ["Zoom"]),
    ("Medium wide HI/continuum survey", ["Continuum", "Zoom"]),
    ("Mid Band 1 Wide Survey", ["Continuum"]),
    ("SKA-VLBI methanol maser parallax survey", [bad.OUTSIDE]),
])
def test_classify_request(name, cats):
    assert bad.classify_request({"id": -1, "name": name}) == cats


def test_pss_volume_is_rate_times_duration():
    odps = bad.request_odps({"telescope": "mid", "hours": 1.0}, ["PSS"], {}, None)
    assert odps[0]["volume_TB"] == pytest.approx(0.0839 / 1024 * 3600)
