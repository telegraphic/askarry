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
