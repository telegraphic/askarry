import numpy as np
import pytest

import rag.source_db as sd


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Live SIMBAD lookups are stubbed out unless a test opts in."""
    monkeypatch.setattr(sd, "_live_lookup", lambda name: {"found": False})


def _names(text):
    return [raw for raw, _ in sd.extract_names(text)]


def test_extract_names_catalogues_and_common_names():
    text = "We observed NGC 5128 (Cen A), PSR J0437−4715, M87, FRB 20121102A and the LMC."
    assert _names(text) == ["NGC 5128", "Cen A", "PSR J0437-4715", "M87", "FRB 20121102A", "LMC"]


def test_extract_names_rejects_false_positives():
    assert _names("University of Manchester, Manchester M13 9PL, UK") == []
    assert _names("the Jet-M01/M22 + Flare-M22.AA4 model") == []
    assert _names("an M9 ultracool dwarf at M 10 kpc; limit of M 200 = 4e14; M01") == []


def test_name_key_and_query_form():
    assert sd.name_key("NGC5128") == sd.name_key("ngc 5128")
    assert sd.query_form("PSRs B1913+16") == "PSR B1913+16"


def _fake_simbad(names):
    known = {
        "NGC 5128": ("NAME Centaurus A", 201.365, -43.019, "BLL"),
        "Cen A": ("NAME Centaurus A", 201.365, -43.019, "BLL"),
        "NAME SGR 1935+2154": ("NAME Sgr 1935+2154", 293.73, 21.90, "Psr"),
        "GW150914": ("GrW 150914", np.ma.masked, np.ma.masked, "GrW"),
    }
    return [
        {"object_number_id": i, "main_id": known[n][0] if n in known else "",
         "ra": known[n][1] if n in known else np.ma.masked,
         "dec": known[n][2] if n in known else np.ma.masked,
         "otype": known.get(n, (0, 0, 0, None))[3], "rvz_redshift": np.ma.masked}
        for i, n in enumerate(names, 1)
    ]


def test_resolve_and_query_roundtrip(tmp_path):
    conn = sd.connect(tmp_path / "s.db")
    chunks = [
        {"text": "Cen A and NGC 5128 are one object. SGR 1935+2154 flared.",
         "meta": {"source": "/x/AASKAII/The Cosmos/A01.pdf", "doc_source": "aaskaii", "chunk_index": 0, "page_no": 2}},
        {"text": "GW150914 and PKS 9999-99 were not resolved.",
         "meta": {"source": "/x/AASKAII/The Cosmos/B01.pdf", "doc_source": "aaskaii", "chunk_index": 3, "page_no": 5}},
    ]
    assert sd.record_mentions(conn, chunks) == 5
    assert sd.record_mentions(conn, chunks) == 0  # idempotent
    assert sd.resolve_pending(conn, _fake_simbad) == {"resolved": 4, "unresolved": 1}

    top = sd.query_sources(conn)[0]
    assert top["main_id"] == "NAME Centaurus A" and top["names_as_written"] == ["Cen A", "NGC 5128"]
    # "NAME " retry resolved the magnetar; the GW event has no position so Dec filters drop it.
    assert [r["main_id"] for r in sd.query_sources(conn, dec_min=0)] == ["NAME Sgr 1935+2154"]
    assert len(sd.source_mentions(conn, "centaurus a")["mentions"]) == 2
    assert sd.unresolved_names(conn)[0]["raw_name"] == "PKS 9999-99"


def test_survey_fields_map_to_simbad_or_curated():
    assert sd.query_form("COSMOS") == "NAME COSMOS Field"
    assert sd.query_form("GOODS-South") == "NAME GOODS South Field"
    assert sd._field("EoR1")[1]["dec"] == -27.0
    assert _names("fields: COSMOS, ECDFS, EoR0 and ELAIS-N1") == ["COSMOS", "ECDFS", "EoR0", "ELAIS-N1"]


def test_curated_field_never_sent_to_simbad(tmp_path):
    conn = sd.connect(tmp_path / "s.db")
    chunks = [{"text": "Deep integrations on EoR0 and Cen A.",
               "meta": {"source": "/x/AASKAII/The Cosmos/A01.pdf", "doc_source": "aaskaii", "chunk_index": 0}}]
    sd.record_mentions(conn, chunks)
    sent = []
    sd.resolve_pending(conn, lambda names: sent.extend(names) or _fake_simbad(names))
    assert sent == ["Cen A"]
    row = conn.execute("SELECT * FROM sources WHERE main_id = 'EoR0 field'").fetchone()
    assert row["dec_deg"] == -27.0 and row["resolved_by"].startswith("curated: deLeraAcedo01")


def test_add_request_requires_verbatim_quote(tmp_path, monkeypatch):
    conn = sd.connect(tmp_path / "s.db")
    meta = {"source": "/x/AASKAII/The Cosmos/A01.pdf", "doc_source": "aaskaii", "page_no": 4}
    monkeypatch.setattr(sd, "find_quote", lambda q, p, b=None: meta if "stated" in q else None)
    with pytest.raises(ValueError, match="verbatim"):
        sd.add_request(conn, "X", 10, "low", "A01", "invented text", ra=0, dec=-30)
    with pytest.raises(ValueError, match="No position"):
        sd.add_request(conn, "Unknown", 10, "low", "A01", "stated text")
    out = sd.add_request(conn, "X", 10, "low", "A01", "stated text", ra=0, dec=-30, sun="night")
    assert out == {"id": 1, "paper": "A01", "page": 4, "placed": True}
    assert sd.list_requests(conn)[0]["ra"] == 0 and sd.list_requests(conn)[0]["sun"] == "night"


def test_find_quote_ignores_pdf_spacing(monkeypatch):
    import rag.corpus_tools as ct
    import rag.retrieval as retrieval

    src = "/x/AASKAII/The Cosmos/A01.pdf"
    chunks = [{"text": "EoR2 (RA = 10 . 3 h , DEC = -10 ◦ )", "meta": {"source": src, "doc_source": "aaskaii"}}]
    monkeypatch.setattr(retrieval, "get_all_chunks", lambda: chunks)
    monkeypatch.setattr(ct, "get_all_chunks", lambda: chunks)
    assert sd.find_quote("RA = 10 .3 h", "A01") is not None
    assert sd.find_quote("RA = 11 h", "A01") is None


def test_survey_field_positions_and_unplaced_requests(tmp_path, monkeypatch):
    from rag.scheduling import lst_pressure

    conn = sd.connect(tmp_path / "s.db")
    monkeypatch.setattr(sd, "find_quote", lambda q, p, b=None: (
        {"source": f"/x/AASKAII/The Cosmos/{p}.pdf", "doc_source": "aaskaii", "page_no": 4}
        if "stated" in q else None))
    # Field named without a centre: unplaced; a later paper stating the centre places it.
    assert sd.add_survey_field(conn, "A01", "stated field", name="ELAIS-N1")["geometry"] is None
    sd.add_survey_field(conn, "B01", "stated centre", name="ELAIS N1", ra=242.5, dec=55.0, area_deg2=10)
    fields = sd.list_fields(conn)
    assert fields[0]["n_papers"] == 2 and fields[0]["geometry"]["ra"] == 242.5
    with pytest.raises(ValueError, match="verbatim"):
        sd.add_survey_field(conn, "A01", "invented", name="X")

    assert sd.add_request(conn, "ELAIS-N1", 100, "mid", "A01", "stated time")["placed"]
    with pytest.raises(ValueError, match="position_note"):
        sd.add_request(conn, "deep field", 1000, "low", "A01", "stated time")
    assert not sd.add_request(conn, "deep field", 1000, "low", "A01", "stated time",
                              position_note="unnamed deep field")["placed"]
    out = lst_pressure(sd.list_requests(conn), "low", year_start="2027-01-01")
    assert out["unplaced_hours"] == 1000 and out["unplaced"][0]["position_note"] == "unnamed deep field"


def test_connect_migrates_old_requests_table(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY, name TEXT, hours REAL, telescope TEXT,"
                " paper TEXT, quote TEXT)")
    old.commit(); old.close()
    cols = {r[1] for r in sd.connect(path).execute("PRAGMA table_info(requests)")}
    assert "position_note" in cols


def test_unnamed_areas_and_galactic_plane(tmp_path, monkeypatch):
    from rag.scheduling import lst_pressure

    conn = sd.connect(tmp_path / "s.db")
    monkeypatch.setattr(sd, "find_quote", lambda q, p, b=None: (
        {"source": f"/x/AASKAII/The Cosmos/{p}.pdf", "doc_source": "aaskaii", "page_no": 9}
        if "stated" in q else None))
    with pytest.raises(ValueError, match="description"):
        sd.add_survey_field(conn, "C01", "stated")
    # Known only by size: recorded, but unplaced.
    small = sd.add_survey_field(conn, "C01", "stated 2% of the sky", description="several fields, 2% of the sky")
    assert small["field"] == "C01 area: several fields, 2% of the sky" and small["geometry"] is None
    south = sd.add_survey_field(conn, "V01", "stated southern sky", description="entire southern sky",
                                ra_range=[0, 360], dec_range=[-90, 0])
    plane = sd.add_survey_field(conn, "G01", "stated plane", description="Galactic plane |b| < 5",
                                gal_b_range=[-5, 5])
    assert plane["geometry"]["gal_l_range"] == [0.0, 360.0]

    assert not sd.add_request(conn, "wide", 1000, "low", "C01", "stated time", field=small["field"],
                              position_note="area size only")["placed"]
    assert sd.add_request(conn, "south", 500, "low", "V01", "stated time", field=south["field"])["placed"]
    assert sd.add_request(conn, "plane", 300, "low", "G01", "stated time", field=plane["field"])["placed"]
    out = lst_pressure(sd.list_requests(conn), "low", year_start="2027-01-01")
    assert out["unplaced_hours"] == 1000 and out["total_demand_h"] == pytest.approx(800)


def test_live_lookup_places_and_caches(tmp_path, monkeypatch):
    conn = sd.connect(tmp_path / "s.db")
    calls = []
    def fake(name):
        calls.append(name)
        if name == "OMC-2":
            return {"found": True, "service": "SIMBAD", "name": "NAME OMC-2", "ra_deg": 83.85,
                    "dec_deg": -5.17, "gal_l_deg": 209.0, "gal_b_deg": -19.5, "object_type": "MoC"}
        return {"found": False}
    monkeypatch.setattr(sd, "_live_lookup", fake)
    monkeypatch.setattr(sd, "find_quote", lambda q, p, b=None: {"source": f"/x/AASKAII/C/{p}.pdf", "page_no": 1})
    assert sd.add_request(conn, "OMC-2", 1000, "mid", "B01", "q")["placed"]
    assert sd.add_request(conn, "omc -2", 5, "mid", "B01", "q")["placed"] and calls == ["OMC-2"]  # cached
    with pytest.raises(ValueError, match="No position"):
        sd.add_request(conn, "Nonexistent-X", 5, "mid", "B01", "q")
    with pytest.raises(ValueError):
        sd.add_request(conn, "Nonexistent-X", 5, "mid", "B01", "q")
    assert calls.count("Nonexistent-X") == 1  # "not found" cached too
    assert len(sd.list_requests(conn, extracted_by=None)) == 2


def test_review_layer_overrides_without_losing_extraction(tmp_path, monkeypatch):
    from rag.scheduling import lst_pressure

    conn = sd.connect(tmp_path / "s.db")
    monkeypatch.setattr(sd, "find_quote", lambda q, p, b=None: {"source": f"/x/AASKAII/C/{p}.pdf", "page_no": 1})
    a = sd.add_request(conn, "wide A", 10000, "low", "P1", "q", position_note="area only", commensal_group="Wide A")["id"]
    b = sd.add_request(conn, "wide A again", 10000, "low", "P2", "q", position_note="area only")["id"]
    c = sd.add_request(conn, "psr", 30, "low", "P3", "q", ra=0, dec=-30, commensal_group="programme")["id"]
    d = sd.add_request(conn, "always-on", 8760, "low", "P4", "q", ra=0, dec=-30)["id"]
    south = sd.add_survey_field(conn, "P1", "q", description="southern sky", ra_range=[0, 360], dec_range=[-90, 0])["field"]

    with pytest.raises(ValueError, match="note"):
        sd.review_request(conn, a, "accepted", "")
    with pytest.raises(ValueError, match="duplicate_of"):
        sd.review_request(conn, b, "duplicate", "same survey")
    with pytest.raises(ValueError, match="no position"):
        sd.review_request(conn, a, "accepted", "x", field="P9 area: nowhere")
    sd.review_request(conn, a, "accepted", "footprint stated in P1", field=south)
    sd.review_request(conn, b, "duplicate", "same survey as P1", duplicate_of=a)
    sd.review_request(conn, c, "accepted", "separate target, not commensal", commensal_group="", hours=35)
    sd.review_request(conn, d, "rejected", "commensal mode, no dedicated time")

    reqs = {r["id"]: r for r in sd.list_requests(conn)}
    assert set(reqs) == {a, c}
    assert "ra_range" in reqs[a] and reqs[c]["hours"] == 35 and reqs[c]["extracted_hours"] == 30
    assert "commensal_group" not in reqs[c]
    assert len(sd.list_requests(conn, include_rejected=True)) == 4
    raw = conn.execute("SELECT hours, commensal_group FROM requests WHERE id = ?", (c,)).fetchone()
    assert tuple(raw) == (30, "programme")  # extraction untouched
    out = lst_pressure(sd.list_requests(conn, reviewed_only=True), "low", year_start="2027-01-01", years=5)
    assert out["total_demand_h"] == pytest.approx(10035) and out["planning_years"] == 5
