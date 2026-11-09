"""Duplicate detection, the index fingerprint and its cache invalidation."""

import json
from pathlib import Path

import rag.bibliography as bibliography
import rag.corpus_tools as ct
from rag import store, zones
from rag.config import HERE

FIXTURE = json.loads((Path(__file__).parent / "fixtures/chunks/zones.json").read_text())
FES = "pdfs/AASKAII/Formation and Evolution of Stars"
SWG = "pdfs/AASKAII/Science Working Group Overviews"
SEP = "pdfs/AASKAII/Sun, Earth and Planets"


def _doc(path: str, chunks: list[dict]) -> list[dict]:
    """Copy fixture chunks under another file path."""
    return [{"text": c["text"], "meta": {**c["meta"], "source": str(HERE / path), "doc_source": "aaskaii"}}
            for c in chunks]


def _info(path: str, listed: bool, section_match: bool) -> dict:
    return {"doc_source": "aaskaii", "stem": Path(path).stem, "book": True, "listed": listed,
            "section_match": section_match}


def test_near_duplicate_copy_marked_noncanonical():
    # Curiel01c is a re-download of Curiel01 filed in a real section.
    real, copy = f"{SEP}/Curiel01.pdf", f"{SEP}/Curiel01c.pdf"
    chunks = _doc(real, FIXTURE["Curiel01"]) + _doc(copy, FIXTURE["Curiel01"])
    status = zones.find_duplicates(chunks, {str(HERE / real): _info(real, True, True),
                                            str(HERE / copy): _info(copy, False, False)})
    assert status[str(HERE / real)]["status"] == "canonical"
    assert status[str(HERE / copy)] == {"status": "duplicate", "duplicate_of": str(HERE / real),
                                        "reason": "same text as another indexed file"}


def test_same_stem_in_two_folders_counted_once():
    # Bianchi01 exists in two folders; the SWG copy actually holds Bianchi02's text.
    real, misfiled, other = f"{FES}/Bianchi01.pdf", f"{SWG}/Bianchi01.pdf", f"{SWG}/Bianchi02.pdf"
    chunks = (_doc(real, FIXTURE["Maccagni01"]) + _doc(misfiled, FIXTURE["Nasirudin01"])
              + _doc(other, FIXTURE["Nasirudin01"]))
    status = zones.find_duplicates(chunks, {str(HERE / real): _info(real, True, True),
                                            str(HERE / misfiled): _info(misfiled, True, False),
                                            str(HERE / other): _info(other, True, True)})
    assert status[str(HERE / real)]["status"] == "canonical"
    assert status[str(HERE / other)]["status"] == "canonical"
    assert status[str(HERE / misfiled)]["duplicate_of"] == str(HERE / other)


def test_same_stem_different_text_keeps_folder_that_matches_bibliography():
    real, misfiled = f"{FES}/Bianchi01.pdf", f"{SWG}/Bianchi01.pdf"
    chunks = _doc(real, FIXTURE["Maccagni01"]) + _doc(misfiled, FIXTURE["Hota01"])
    status = zones.find_duplicates(chunks, {str(HERE / real): _info(real, True, True),
                                            str(HERE / misfiled): _info(misfiled, True, False)})
    assert status[str(HERE / misfiled)]["status"] == "duplicate"
    assert status[str(HERE / real)]["status"] == "canonical"


def test_unlisted_book_document_flagged():
    page = "pdfs/contents_webpage.pdf"
    status = zones.find_duplicates(_doc(page, FIXTURE["Hota01"]), {str(HERE / page): _info(page, False, False)})
    assert status[str(HERE / page)]["status"] == "unlisted"


def test_fingerprint_stable_and_changes_with_text():
    chunks = _doc(f"{SEP}/Curiel01.pdf", FIXTURE["Curiel01"])
    fp = zones.index_fingerprint(chunks, HERE)
    assert fp == zones.index_fingerprint(chunks[::-1], HERE)
    edited = [{**chunks[0], "text": chunks[0]["text"] + " edited"}] + chunks[1:]
    assert zones.index_fingerprint(edited, HERE) != fp


def test_keyword_coverage_counts_each_paper_once_and_skips_copies(monkeypatch):
    real, misfiled = f"{FES}/Bianchi01.pdf", f"{SWG}/Bianchi01.pdf"
    chunks = [{"text": "Planck maps of the dust.", "meta": {"source": str(HERE / real), "doc_source": "aaskaii",
                                                          "doc_status": "canonical"}},
              {"text": "Planck maps again.", "meta": {"source": str(HERE / misfiled), "doc_source": "aaskaii",
                                                      "doc_status": "duplicate", "duplicate_of": "Bianchi02"}},
              {"text": "Planck too.", "meta": {"source": str(HERE / f"{SEP}/Curiel01c.pdf"),
                                               "doc_source": "aaskaii", "doc_status": "duplicate"}}]
    bib = {"Bianchi01": {"title": "t", "authors": [], "section": "Formation and Evolution of Stars"}}
    monkeypatch.setattr(bibliography, "load_bibliography", lambda: bib)
    monkeypatch.setattr(ct, "get_all_chunks", lambda: chunks)
    monkeypatch.setattr(ct, "get_acronyms", lambda: {})
    term = ct.keyword_coverage(["Planck"])["terms"][0]
    assert term["n_papers"] == 1
    assert term["sections"] == {"Formation and Evolution of Stars": {"n_papers": 1, "papers": ["Bianchi01"]}}
    every = ct.keyword_coverage(["Planck"], include_noncanonical=True)["terms"][0]
    assert every["n_papers"] == 3


def test_cache_invalidates_when_manifest_rewritten(tmp_path):
    class FakeCollection:
        def count(self):
            return 5

    coll = FakeCollection()
    store._collections[str(tmp_path)] = coll
    try:
        builds = []
        cache = store.CountCache()
        cache.get(coll, lambda: builds.append(1) or len(builds))
        cache.get(coll, lambda: builds.append(1) or len(builds))
        assert len(builds) == 1
        zones.write_manifest(tmp_path, {"index_fingerprint": "x"})  # a retag: same count, new metadata
        cache.get(coll, lambda: builds.append(1) or len(builds))
        assert len(builds) == 2
    finally:
        store._collections.pop(str(tmp_path))
