"""
Deterministic corpus-level tools for the MCP server: book-section/paper
filters for search, keyword coverage sweeps, quote/figure verification, paper
metadata, and acronym lookup.

Built for multi-agent analyses like the AASKAII Atlas, where agents otherwise
hand-tally "which sections mention X" from ranked search results and can
misattribute a figure to the wrong paper. Everything here scans the same
chunks the retrieval index holds, so counts are exact and repeatable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from rag import store
from rag.bibliography import bib_key, book_of, format_citation, load_bibliography, lookup_citation, section_of
from rag.config import (
    DOC_SOURCE_AASKA2015,
    DOC_SOURCE_AASKAII,
    DOC_SOURCE_LABELS,
)
from rag.ingestion import find_acronym_candidates
from rag.retrieval import get_acronyms, get_all_chunks, matches_where, retrieve

# Book names accepted by the tools, mapped to chunk doc_source tags.
BOOKS = {
    "aaskaii": DOC_SOURCE_AASKAII,
    "aaska2015": DOC_SOURCE_AASKA2015,
    "ska_capabilities": "ska_capabilities",
}

_BOOK_NAMES = {tag: name for name, tag in BOOKS.items()}

# ponytail: fixed cut-off on the 0-1 cross-encoder score, used only when a
# claim contains no numbers to match literally; tune if verdicts look off.
VERIFY_MIN_SCORE = 0.5

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _stem(source: str) -> str:
    return Path(source).stem


def _book_tag(book: str | None) -> str | None:
    if book is None:
        return None
    if book not in BOOKS:
        raise ValueError(f"Unknown book {book!r}; expected one of {sorted(BOOKS)}")
    return BOOKS[book]


def _indexed_sources() -> set[str]:
    return {c["meta"]["source"] for c in get_all_chunks()}


def resolve_papers(name: str) -> list[str]:
    """Indexed source paths whose filename stem equals *name* (case-insensitive),
    or failing that whose title contains it."""
    sources = _indexed_sources()
    wanted = Path(name).stem.lower()
    exact = [s for s in sources if _stem(s).lower() == wanted]
    if exact:
        return sorted(exact)
    return sorted(
        s for s in sources
        if name.lower() in (lookup_citation(s) or {}).get("title", "").lower()
    )


def build_filter(book: str | None = None, section: str | None = None,
                 paper: str | None = None) -> dict | None:
    """Flat metadata filter for rag.retrieval.retrieve (list value = any of).

    Raises ValueError when *section* or *paper* matches nothing, so a typo
    surfaces as an error instead of silently searching the whole corpus.
    """
    where: dict = {}
    tag = _book_tag(book)
    if tag:
        where["doc_source"] = tag
    if paper:
        sources = resolve_papers(paper)
        if not sources:
            raise ValueError(f"No indexed paper matches {paper!r} (see list_documents)")
        where["source"] = sources
    if section:
        in_section = [s for s in _indexed_sources()
                      if section_of(s).lower() == section.lower()]
        if "source" in where:
            in_section = [s for s in in_section if s in where["source"]]
        if not in_section:
            raise ValueError(f"No indexed papers in section {section!r} (see list_documents)")
        where["source"] = sorted(in_section)
    return where or None


def _term_regexes(term: str, acronyms: dict[str, str]) -> list[re.Pattern]:
    """One regex per "|"-separated alternative, plus the full form of any
    all-caps acronym alternative. All-caps alternatives match case-sensitively
    (so "HI" doesn't match "hi"); everything else ignores case."""
    alts = [a.strip() for a in term.split("|") if a.strip()]
    alts += [acronyms[a] for a in list(alts) if a.isupper() and a in acronyms]
    return [
        re.compile(rf"\b{re.escape(a)}\b", 0 if a.isupper() else re.IGNORECASE)
        for a in dict.fromkeys(alts)
    ]


def keyword_coverage(terms: list[str], book: str | None = "aaskaii",
                     min_hits_per_paper: int = 1) -> dict:
    """Exact per-section / per-paper mention counts for each term.

    A paper counts as covering a term when the term (or an alternative /
    acronym expansion) appears at least *min_hits_per_paper* times in it.
    """
    where = {"doc_source": _book_tag(book)} if book else None
    chunks = [c for c in get_all_chunks() if matches_where(c["meta"], where)]
    acronyms = get_acronyms()

    section_sizes: dict[str, set] = defaultdict(set)
    for c in chunks:
        section_sizes[section_of(c["meta"]["source"]) or "(unknown)"].add(_stem(c["meta"]["source"]))

    results = []
    for term in terms:
        regexes = _term_regexes(term, acronyms)
        hits: dict[str, int] = defaultdict(int)
        for c in chunks:
            n = sum(len(r.findall(c["text"])) for r in regexes)
            if n:
                hits[c["meta"]["source"]] += n
        by_section: dict[str, list[str]] = defaultdict(list)
        for source, n in hits.items():
            if n >= min_hits_per_paper:
                by_section[section_of(source) or "(unknown)"].append(_stem(source))
        results.append({
            "term": term,
            "total_mentions": sum(hits.values()),
            "n_papers": sum(len(p) for p in by_section.values()),
            "n_sections": len(by_section),
            "sections": {
                s: {"n_papers": len(p), "papers": sorted(p)}
                for s, p in sorted(by_section.items(), key=lambda kv: -len(kv[1]))
            },
        })
    results.sort(key=lambda r: (-r["n_sections"], -r["n_papers"]))
    return {
        "book": book or "all",
        "min_hits_per_paper": min_hits_per_paper,
        "papers_per_section": {s: len(p) for s, p in sorted(section_sizes.items())},
        "terms": results,
    }


def _numbers_in(numbers: list[str], text: str) -> list[str]:
    return [n for n in numbers if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text)]


def _describe(source: str, page: int, text: str, numbers: list[str], score: float | None) -> dict:
    found = _numbers_in(numbers, text)
    supports = len(found) == len(numbers) if numbers else (score or 0) >= VERIFY_MIN_SCORE
    out = {
        "paper": _stem(source),
        "title": (lookup_citation(source) or {}).get("title", ""),
        "section": section_of(source),
        "page": page,
        "numbers_found": found,
        "supports": supports,
        "excerpt": text[:400],
    }
    if score is not None:
        out["score"] = score
    return out


def verify_quote(text: str, paper: str | None = None, top_k: int = 5) -> dict:
    """Check where a claimed sentence/figure actually appears in the corpus.

    Combines ranked search (semantic + BM25 + rerank) with a literal scan for
    chunks containing every number in *text*. With *paper*, reports whether
    that paper supports the claim or whether it only appears elsewhere
    (a likely citation swap).
    """
    numbers = list(dict.fromkeys(_NUMBER_RE.findall(text)))

    def _ranked(where: dict | None, k: int) -> list[dict]:
        chunks = retrieve(text, top_k=k, use_hyde=False, expansion_window=0, where=where)
        return [_describe(c["source"], c["page_no"], c["text"], numbers, c["score"]) for c in chunks]

    corpus = _ranked(None, top_k)
    claimed = _ranked(build_filter(paper=paper), 3) if paper else []

    literal = []
    if numbers:
        for c in get_all_chunks():
            if len(_numbers_in(numbers, c["text"])) == len(numbers):
                literal.append(_describe(c["meta"]["source"], c["meta"].get("page_no", 0),
                                         c["text"], numbers, None))
    candidates = corpus + literal

    if paper:
        claimed_stems = {_stem(s) for s in resolve_papers(paper)}
        if any(c["supports"] for c in claimed) or any(
            c["supports"] and c["paper"] in claimed_stems for c in literal
        ):
            verdict = "supported_by_claimed_paper"
        elif any(c["supports"] for c in candidates):
            verdict = "found_in_other_paper"
        else:
            verdict = "not_found"
    else:
        verdict = "found" if any(c["supports"] for c in candidates) else "not_found"

    return {
        "verdict": verdict,
        "numbers_checked": numbers,
        "claimed_paper_matches": claimed,
        "search_matches": corpus,
        "literal_number_matches": literal[:10],
        "literal_number_match_count": len(literal),
    }


def _chunk_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for c in get_all_chunks():
        counts[bib_key(c["meta"]["source"])] += 1
    return counts


def paper_info(name: str, limit: int = 10) -> list[dict]:
    """Bibliography entries matching a filename stem (exact) or title substring."""
    bibliography = load_bibliography()
    wanted = name.lower()
    keys = [k for k in bibliography if _stem(k).lower() == Path(wanted).stem] or [
        k for k, e in bibliography.items()
        if wanted in e["title"].lower()
        or any(wanted in a.lower() for a in e.get("authors", []))
    ]
    counts = _chunk_counts()
    files: dict[str, set] = defaultdict(set)
    for c in get_all_chunks():
        files[bib_key(c["meta"]["source"])].add(c["meta"]["source"])
    return [
        {
            "indexed_files": sorted(files.get(k, ())),
            "paper": _stem(k),
            "title": bibliography[k]["title"],
            "authors": bibliography[k].get("authors", []),
            "section": bibliography[k]["section"],
            "book": DOC_SOURCE_LABELS[book_of(bibliography[k])],
            "citation": format_citation(bibliography[k]),
            "indexed_chunks": counts.get(k, 0),
        }
        for k in sorted(keys)[:limit]
    ]


def missing_papers(book: str | None = None) -> dict:
    """Bibliography entries with no indexed chunks, and indexed files with no
    bibliography entry — makes corpus gaps explicit instead of silent."""
    bibliography = load_bibliography()
    counts = _chunk_counts()
    not_indexed = [
        {"paper": _stem(k), "title": e["title"], "section": e["section"], "book": _BOOK_NAMES[book_of(e)]}
        for k, e in sorted(bibliography.items())
        if counts.get(k, 0) == 0 and (book is None or book_of(e) == _book_tag(book))
    ]
    tag = _book_tag(book)
    no_entry = sorted({
        _stem(c["meta"]["source"]) for c in get_all_chunks()
        if bib_key(c["meta"]["source"]) not in bibliography
        and (tag is None or c["meta"].get("doc_source") == tag)
    })
    return {"not_indexed": not_indexed, "indexed_without_bibliography_entry": no_entry}


_acronym_caches: defaultdict[str, store.CountCache] = defaultdict(store.CountCache)


def _acronyms_for(book: str) -> dict[str, dict]:
    tag = _book_tag(book)
    collection = store.get_chroma_collection()
    return _acronym_caches[tag].get(collection, lambda: find_acronym_candidates(collection, tag))


def _acronym_row(key: str, entry: dict, max_papers: int) -> dict:
    return {
        "acronym": key,
        "expansion": max(entry["expansions"], key=entry["expansions"].get),
        "variants": entry["expansions"],
        "category": entry["category"],
        "occurrence_count": entry["occurrence_count"],
        "n_papers": len(entry["sources"]),
        "papers": [_stem(s) for s in entry["sources"][:max_papers]],
    }


def lookup_acronym(acronym: str, book: str = "aaskaii") -> list[dict]:
    """Every sense of *acronym* defined in the corpus (ambiguous acronyms like
    DM come back as one row per meaning)."""
    key = acronym.upper()
    return [
        _acronym_row(k, e, max_papers=50)
        for k, e in _acronyms_for(book).items()
        if k == key or k.startswith(f"{key} (")
    ]


def list_acronyms(book: str = "aaskaii", category: str | None = None, limit: int = 100) -> list[dict]:
    """Most-used acronyms in a book, optionally filtered by category."""
    rows = [
        _acronym_row(k, e, max_papers=0)
        for k, e in _acronyms_for(book).items()
        if category is None or e["category"] == category
    ]
    rows.sort(key=lambda r: -r["occurrence_count"])
    return rows[:limit]
