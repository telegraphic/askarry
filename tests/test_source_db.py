import numpy as np
import pytest

import rag.source_db as sd


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
    assert out == {"id": 1, "paper": "A01", "page": 4}
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
