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
    assert out["variants"][0]["sensitivity_ujy_beam"] == 0.9  # natural falls back to weighted_*
