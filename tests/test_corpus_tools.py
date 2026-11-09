import rag.bibliography as bibliography
import rag.corpus_tools as ct
import rag.sensitivity_calculator as sc
from rag.config import HERE
from rag.ingestion import classify_acronym
from rag.retrieval import _chroma_where, matches_where

A = str(HERE / "pdfs/AASKAII/The Cosmos/Smith01.pdf")
B = str(HERE / "pdfs/AASKAII/Methods and Techniques/Jones01.pdf")
OLD = str(HERE / "pdfs/AASKA2015/Magnetism/Smith01.pdf")

CHUNKS = [
    {"text": "Faraday rotation and RM grids. RM synthesis.", "meta": {"source": A, "doc_source": "aaskaii", "page_no": 3}},
    {"text": "We measure the rotation measure; hi there.", "meta": {"source": B, "doc_source": "aaskaii", "page_no": 1}},
    {"text": "sigma = 12.6 uJy/beam at 50 hours", "meta": {"source": B, "doc_source": "aaskaii", "page_no": 7}},
    {"text": "Faraday rotation in 2015.", "meta": {"source": OLD, "doc_source": "aaska2015", "page_no": 2}},
]
# Stem shared by both books: the 2015 entry is keyed with a book prefix.
BIB = {
    "Smith01": {"title": "New Cosmos", "authors": ["J. Smith"], "section": "The Cosmos"},
    "AASKA2015/Smith01": {"title": "Old Magnetism", "authors": ["J. Smith"], "section": "Magnetism", "year": 2015},
    "Jones01": {"title": "Methods Paper", "authors": ["A. Jones"], "section": "Methods and Techniques"},
}


def _fake_corpus(monkeypatch):
    monkeypatch.setattr(bibliography, "load_bibliography", lambda: BIB)
    monkeypatch.setattr(ct, "load_bibliography", lambda: BIB)
    monkeypatch.setattr(ct, "get_all_chunks", lambda: CHUNKS)
    monkeypatch.setattr(ct, "get_acronyms", lambda: {"RM": "Rotation Measure"})


def test_lookup_is_book_aware(monkeypatch):
    _fake_corpus(monkeypatch)
    assert bibliography.lookup_citation(A)["title"] == "New Cosmos"
    assert bibliography.lookup_citation(OLD)["title"] == "Old Magnetism"
    assert bibliography.section_of(A) == "The Cosmos"
    assert bibliography.section_of(OLD) == "Magnetism"
    assert bibliography.section_of(B) == "Methods and Techniques"
    assert [p["book"] for p in ct.paper_info("Smith01")] == [
        "Advancing Astrophysics with the SKA, 2015", "AASKAII, 2026"]


def test_keyword_coverage_counts_alternatives_expansion_and_book(monkeypatch):
    _fake_corpus(monkeypatch)
    out = ct.keyword_coverage(["Faraday rotation|RM", "HI"], book="aaskaii")
    rm, hi = out["terms"]
    # 3 mentions in A (Faraday rotation, RM x2) + 1 expansion in B; 2015 excluded.
    assert rm["total_mentions"] == 4 and rm["n_sections"] == 2
    assert hi["total_mentions"] == 0  # all-caps term is case-sensitive: "hi" ≠ "HI"
    strict = ct.keyword_coverage(["Faraday rotation|RM"], book="aaskaii", min_hits_per_paper=2)
    assert list(strict["terms"][0]["sections"]) == ["The Cosmos"]


def test_build_filter_rejects_unknown_section(monkeypatch):
    _fake_corpus(monkeypatch)
    assert ct.build_filter(book="aaskaii", section="the cosmos") == {"doc_source": "aaskaii", "source": [A]}
    try:
        ct.build_filter(section="Nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_verify_quote_flags_citation_swap(monkeypatch):
    _fake_corpus(monkeypatch)
    monkeypatch.setattr(ct, "retrieve", lambda *a, **k: [])
    out = ct.verify_quote("12.6 uJy/beam in 50 hours", paper="Smith01")
    assert out["verdict"] == "found_in_other_paper"
    assert out["literal_number_matches"][0]["paper"] == "Jones01"
    assert ct.verify_quote("12.6 uJy/beam in 50 hours", paper="Jones01")["verdict"] == "supported_by_claimed_paper"


def test_where_helpers():
    assert _chroma_where({"doc_source": "x"}) == {"doc_source": "x"}
    assert _chroma_where({"a": 1, "b": [2, 3]}) == {"$and": [{"a": 1}, {"b": {"$in": [2, 3]}}]}
    assert matches_where({"a": 1, "b": 3}, {"a": 1, "b": [2, 3]})
    assert not matches_where({"a": 1, "b": 4}, {"a": 1, "b": [2, 3]})


def test_classify_acronym_whole_words():
    assert classify_acronym("Field-Programmable Gate Array") == "Telescopes & Instruments"
    assert classify_acronym("SKA Regional Centres") == "Organizations & Programs"


def test_compare_to_calculator_picks_closest(monkeypatch):
    def fake(telescope, endpoint, params):
        r = params.get("robustness", 5)
        total = None if params["weighting_mode"] == "natural" else {"value": (1 + abs(r)) * 1e-6}
        return {
            "transformed_result": {"total_continuum_sensitivity": total,
                                   "weighted_continuum_sensitivity": {"value": 0.9e-6}},
            "weighting": {"continuum_weighting": {"beam_size": {"beam_maj_scaled": 1 / 3600, "beam_min_scaled": 1 / 3600}}},
        }
    monkeypatch.setattr(sc, "query_sensitivity_calculator", fake)
    out = sc.compare_to_calculator("mid", {}, 1.0, claimed_beam_arcsec=1.0)
    assert out["closest"]["robustness"] == 0 and out["closest"]["pct_diff"] == 0.0
    assert out["variants"][0]["sensitivity_ujy"] == 0.9  # natural falls back to weighted_*


def test_paper_info_pdf_url_only_for_aaskaii(monkeypatch):
    _fake_corpus(monkeypatch)
    urls = {p["book"]: p["pdf_url"] for p in ct.paper_info("Smith01")}
    assert [u for u in urls.values() if u] == ["https://www.skao.int/sites/default/files/documents/Smith01.pdf"]
    assert None in urls.values()  # the AASKA2015 twin has no stored URL


def test_build_filter_book_restricts_shared_paper_id(monkeypatch):
    _fake_corpus(monkeypatch)
    assert ct.build_filter(book="aaskaii", paper="Smith01")["source"] == [A]


def test_verify_quote_checks_exact_chunk(monkeypatch):
    _fake_corpus(monkeypatch)
    # Same chunk number in both books' Smith01 (index 0 and 3 → both 0).
    chunks = [dict(c, meta={**c["meta"], "chunk_index": i % 3}) for i, c in enumerate(CHUNKS)]
    monkeypatch.setattr(ct, "get_all_chunks", lambda: chunks)
    monkeypatch.setattr(ct, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(ct, "rerank_score", lambda q, t: 0.99)
    out = ct.verify_quote("12.6 uJy/beam in 50 hours", paper="Jones01", chunk_index=2)
    assert out["chunk_match"]["page"] == 7 and out["verdict"] == "supported_by_claimed_paper"
    try:
        ct.verify_quote("x", paper="Smith01", chunk_index=0)  # ID in both books, no book=
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for ambiguous paper ID")


def test_facility_patterns_guard_common_words():
    regexes = ct._facility_regexes()
    hits = lambda name, text: any(r.search(text) for r in regexes[name])  # noqa: E731
    assert hits("Planck", "Planck 2018 results") and not hits("Planck", "Planck's constant h")
    assert hits("Rubin/LSST", "the Vera C. Rubin Observatory") and not hits("Rubin/LSST", "Rubin et al. (1980)")
    assert hits("VLA", "the VLA and ngVLA") and not hits("VLA", "only ngVLA")
    assert not hits("Fermi", "Fermi acceleration at shocks") and hits("Fermi", "Fermi-LAT sources")
    assert not hits("FAST", "the FAST EM solver") and hits("FAST", "FAST telescope pulsars")
    assert not hits("LIGO/Virgo/KAGRA", "the Virgo cluster")


def test_verify_citation_matches_author_and_year(monkeypatch, tmp_path):
    import rag.citation_db as cdb

    conn = cdb.connect(tmp_path / "c.db")
    conn.execute("INSERT INTO refs (ref_key, raw, first_author, year) VALUES ('k1', 'F. Govoni et al. A&A 2019', 'govoni', 2019)")
    conn.execute("INSERT INTO refs (ref_key, raw, first_author, year) VALUES ('k2', 'F. Govoni et al. A&A 2005', 'govoni', 2005)")
    for k in ("k1", "k2"):
        conn.execute("INSERT INTO cites (ref_key, paper, source_path) VALUES (?, 'Jones01', ?)", (k, B))
    conn.commit()
    _fake_corpus(monkeypatch)
    monkeypatch.setattr(cdb, "connect", lambda *a: conn)
    assert ct.verify_citation("Jones01", "Govoni et al. 2019")["found"]
    miss = ct.verify_citation("Jones01", "(Govoni et al., 2006a)")
    assert not miss["found"] and miss["same_author_other_years"][0].startswith("2005")


def test_compare_to_calculator_zoom_and_pss(monkeypatch):
    def fake(telescope, endpoint, params):
        if endpoint == "pss/calculate":
            return {"folded_pulse_sensitivity": {"value": 4.2, "unit": "uJy"}, "warnings": []}
        r = params.get("robustness", 5)
        return {
            "transformed_result": [{"total_spectral_sensitivity": {"value": (1 + abs(r)) * 1e-4},
                                    "weighted_spectral_sensitivity": {"value": 9e-5}}],
            "weighting": {"spectral_weighting": [{"beam_size": {"beam_maj_scaled": 1 / 3600, "beam_min_scaled": 1 / 3600}}]},
        }
    monkeypatch.setattr(sc, "query_sensitivity_calculator", fake)
    z = sc.compare_to_calculator("mid", {}, 100, endpoint="zoom")
    assert z["closest"]["robustness"] == 0 and z["closest"]["pct_diff"] == 0.0
    p = sc.compare_to_calculator("low", {}, 4.0, unit="uJy", endpoint="pss")
    assert len(p["variants"]) == 1 and p["closest"]["pct_diff"] == 5.0 and p["unit"] == "uJy"


# --- Keyword coverage: zones, inflection, exclusions (real chunks) ----------
import json as _json
from pathlib import Path as _Path

TERMS = _json.loads((_Path(__file__).parent / "fixtures/chunks/terms.json").read_text())
for _c in TERMS:
    _c["meta"]["source"] = str(HERE / _c["meta"]["source"])


def _term_corpus(monkeypatch, chunks=TERMS):
    monkeypatch.setattr(ct, "get_all_chunks", lambda: chunks)
    monkeypatch.setattr(ct, "get_acronyms", lambda: {})
    monkeypatch.setattr(ct, "section_of", lambda s: "S")


def _papers(result, term):
    row = next(t for t in result["terms"] if t["term"] == term)
    return sorted(p for s in row["sections"].values() for p in s["papers"])


def test_planck_body_zone_excludes_max_planck_references_and_constants(monkeypatch):
    _term_corpus(monkeypatch)
    old = ct.keyword_coverage(["Planck"], zones=None, strict=True)
    assert {"Baczko01", "Bourke01", "Bertacca01"} <= set(_papers(old, "Planck"))
    new = ct.keyword_coverage(["Planck"])
    # Wolz01 names the Planck satellite; the others are an affiliation,
    # a reference list and "Planck and Boltzmann constants".
    assert _papers(new, "Planck") == ["Wolz01"]


def test_chandra_body_excludes_surname_citations(monkeypatch):
    _term_corpus(monkeypatch)
    assert "Baldi01" in _papers(ct.keyword_coverage(["Chandra"], strict=True), "Chandra")
    assert _papers(ct.keyword_coverage(["Chandra"]), "Chandra") == ["Colombo01"]


def test_plural_and_hyphen_variants_match(monkeypatch):
    _term_corpus(monkeypatch)
    r = ct.keyword_coverage(["pulsar survey", "data rate|data volume", "zoom mode|zoom window"])
    assert "Joshi01" in _papers(r, "pulsar survey")
    assert "Prandoni01" in _papers(r, "data rate|data volume")
    assert "Emig01" in _papers(r, "zoom mode|zoom window")


def test_strict_restores_literal_matching(monkeypatch):
    _term_corpus(monkeypatch)
    r = ct.keyword_coverage(["pulsar survey"], strict=True)
    assert "Joshi01" not in _papers(r, "pulsar survey")  # Joshi01 writes "pulsar surveys"
    assert r["matching"].startswith("strict")


def test_user_exclude_and_unknown_zone(monkeypatch):
    _term_corpus(monkeypatch)
    r = ct.keyword_coverage(["Chandra"], exclude=["Chandra X-ray Observatory"])
    assert _papers(r, "Chandra") == []
    try:
        ct.keyword_coverage(["Chandra"], zones=["bodies"])
    except ValueError as exc:
        assert "Unknown zones" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_return_contexts_and_usage_stats(monkeypatch):
    _term_corpus(monkeypatch)
    r = ct.keyword_coverage(["pulsar survey"], return_contexts=1, usage_stats=True)
    row = r["terms"][0]
    ctx = row["contexts"]["Joshi01"][0]
    assert "pulsar survey" in ctx["text"].lower() and ctx["zone"] == "body"
    usage = row["usage"]["Joshi01"]
    assert usage["mentions_per_1k_body_words"] > 0 and usage["in_abstract"] is False


def test_facility_network_applies_zones_and_exclusions(monkeypatch):
    _term_corpus(monkeypatch)
    every = ct.facility_network(min_hits_per_paper=1, zones=None, use_curated_exclusions=False)
    planck = next(f for f in every["facilities"] if f["facility"] == "Planck")
    assert "Bourke01" in planck["papers"]  # "Planck Collaboration" in its reference list
    default = ct.facility_network(min_hits_per_paper=1)
    planck = next(f for f in default["facilities"] if f["facility"] == "Planck")
    assert planck["papers"] == ["Wolz01"]
    chandra = next(f for f in default["facilities"] if f["facility"] == "Chandra")
    assert chandra["papers"] == ["Colombo01"]


def test_inflection_rules():
    rx = lambda alt, txt: bool(ct._alternative_regex(alt, False).search(txt))
    assert rx("frequency", "frequencies") and rx("FRB", "FRBs") and rx("surveys", "survey")
    assert rx("analysis", "analysis") and not rx("HI", "hi there")
