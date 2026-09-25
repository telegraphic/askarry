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
    doc_source_for as doc_source_of,
)
from rag.ingestion import find_acronym_candidates
from rag.retrieval import get_acronyms, get_all_chunks, matches_where, rerank_score, retrieve

# Book names accepted by the tools, mapped to chunk doc_source tags.
BOOKS = {
    "aaskaii": DOC_SOURCE_AASKAII,
    "aaska2015": DOC_SOURCE_AASKA2015,
    "ska_capabilities": "ska_capabilities",
}

_BOOK_NAMES = {tag: name for name, tag in BOOKS.items()}

# Cut-off on the 0-1 cross-encoder score, used only when a claim contains no
# numbers to match literally. Tuned 2026-09-25 with `python -m rag.corpus_tools
# --tune-verify` on 31 paraphrased claims (tests/data/verify_claims.json), each
# scored against its true paper and a same-section other paper: 0.97 gave
# precision 1.0 / recall 0.97 (0.5 let 8/31 wrong-paper claims through,
# precision 0.80). ponytail: small sample; re-tune if the claim set grows.
VERIFY_MIN_SCORE = 0.97

# Where pdfs/download.py fetches AASKAII chapters from (AASKA2015 URLs aren't stored).
AASKAII_PDF_URL = "https://www.skao.int/sites/default/files/documents/{stem}.pdf"

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
        sources = [s for s in resolve_papers(paper) if not tag or doc_source_of(s) == tag]
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


def _mention_counts(patterns: dict[str, list[re.Pattern]], chunks: list[dict]) -> dict[str, dict[str, int]]:
    """{label: {source path: mentions}} for each label's regexes over *chunks*."""
    counts: dict[str, dict[str, int]] = {label: defaultdict(int) for label in patterns}
    for c in chunks:
        for label, regexes in patterns.items():
            n = sum(len(r.findall(c["text"])) for r in regexes)
            if n:
                counts[label][c["meta"]["source"]] += n
    return counts


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
    all_hits = _mention_counts({t: _term_regexes(t, acronyms) for t in terms}, chunks)
    for term in terms:
        hits = all_hits[term]
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


# External facilities named as SKA partners/synergies: (name, category, regex).
# Matched case-sensitively as whole tokens. Guards keep common words, authors
# and physics terms out: "Rubin et al.", the Hubble constant, Planck's
# constant, Fermi acceleration, the FAST EM solver, GMT (the time zone).
FACILITIES: list[tuple[str, str, str]] = [
    # optical / infrared
    ("Rubin/LSST", "optical/IR", r"Rubin Observatory|Vera C\.? Rubin|LSST|Legacy Survey of Space and Time"),
    ("JWST", "optical/IR", r"JWST|James Webb"),
    ("Euclid", "optical/IR", r"Euclid(?! et al)"),
    ("Roman", "optical/IR", r"Roman Space Telescope|Nancy Grace Roman|Roman Observatory"),
    ("HST", "optical/IR", r"HST|Hubble Space Telescope"),
    ("ELT", "optical/IR", r"E-ELT|ELT|Extremely Large Telescope"),
    ("TMT", "optical/IR", r"TMT|Thirty Meter Telescope"),
    ("GMT", "optical/IR", r"Giant Magellan Telescope"),
    ("VLT", "optical/IR", r"VLT(?!I)|Very Large Telescope"),
    ("Keck", "optical/IR", r"Keck(?! et al)"),
    ("Subaru", "optical/IR", r"Subaru"),
    ("Gaia", "optical/IR", r"Gaia"),
    ("SDSS", "optical/IR", r"SDSS|Sloan Digital Sky Survey"),
    ("DESI", "optical/IR", r"DESI"),
    ("4MOST", "optical/IR", r"4MOST"),
    ("WEAVE", "optical/IR", r"WEAVE"),
    ("Pan-STARRS", "optical/IR", r"Pan-STARRS"),
    ("ZTF", "optical/IR", r"ZTF|Zwicky Transient Facility"),
    ("WISE", "optical/IR", r"WISE|unWISE|Wide-field Infrared Survey Explorer"),
    ("Herschel", "optical/IR", r"Herschel(?! et al)"),
    ("Spitzer", "optical/IR", r"Spitzer(?! et al)"),
    ("SPHEREx", "optical/IR", r"SPHEREx"),
    # X-ray
    ("Chandra", "X-ray", r"Chandra(?! et al)"),
    ("XMM-Newton", "X-ray", r"XMM-Newton|XMM"),
    ("eROSITA", "X-ray", r"eROSITA"),
    ("Athena/NewAthena", "X-ray", r"NewAthena|Athena"),
    ("XRISM", "X-ray", r"XRISM"),
    ("Swift", "X-ray", r"Swift(?! et al)"),
    ("Einstein Probe", "X-ray", r"Einstein Probe"),
    ("SVOM", "X-ray", r"SVOM"),
    ("NuSTAR", "X-ray", r"NuSTAR"),
    # gamma-ray
    ("Fermi", "gamma-ray", r"Fermi[-/ ](?:LAT|GBM)|Fermi Gamma|Fermi (?:satellite|telescope|mission|Space)|4FGL|3FGL"),
    ("CTAO", "gamma-ray", r"CTAO?|Cherenkov Telescope Array"),
    ("H.E.S.S.", "gamma-ray", r"H\.E\.S\.S\.?|HESS"),
    ("LHAASO", "gamma-ray", r"LHAASO"),
    ("MAGIC", "gamma-ray", r"MAGIC"),
    ("VERITAS", "gamma-ray", r"VERITAS"),
    ("SWGO", "gamma-ray", r"SWGO"),
    # gravitational waves / neutrinos
    ("LIGO/Virgo/KAGRA", "GW", r"LIGO|KAGRA|LVK|Virgo (?:detector|interferometer|Collaboration)"),
    ("LISA", "GW", r"LISA"),
    ("Einstein Telescope", "GW", r"Einstein Telescope"),
    ("Cosmic Explorer", "GW", r"Cosmic Explorer"),
    ("IceCube", "neutrino", r"IceCube(?:-Gen2)?"),
    ("KM3NeT", "neutrino", r"KM3NeT"),
    # radio (non-SKA)
    ("ALMA", "radio", r"ALMA"),
    ("ngVLA", "radio", r"ngVLA"),
    ("VLA", "radio", r"VLA|Karl G\. Jansky"),
    ("VLBA", "radio", r"VLBA"),
    ("EVN", "radio", r"EVN|European VLBI Network"),
    ("EHT/ngEHT", "radio", r"ngEHT|EHT|Event Horizon Telescope"),
    ("LOFAR", "radio", r"LOFAR(?:2\.0)?"),
    ("MeerKAT", "radio", r"MeerKAT\+?"),
    ("ASKAP", "radio", r"ASKAP"),
    ("MWA", "radio", r"MWA|Murchison Widefield Array"),
    ("HERA", "radio", r"HERA"),
    ("CHIME", "radio", r"CHIME"),
    ("FAST", "radio", r"Five-hundred-meter Aperture Spherical|FAST(?= (?:telescope|radio telescope|observations|survey|CRAFTS|GPPS))"),
    ("DSA", "radio", r"DSA-?2000|DSA-110"),
    ("GBT", "radio", r"GBT|Green Bank Telescope"),
    ("Parkes/Murriyang", "radio", r"Parkes|Murriyang"),
    ("Effelsberg", "radio", r"Effelsberg"),
    ("GMRT", "radio", r"u?GMRT"),
    ("ATCA", "radio", r"ATCA|Australia Telescope Compact Array"),
    ("Apertif", "radio", r"APERTIF|Apertif"),
    ("NenuFAR", "radio", r"NenuFAR"),
    ("Pulsar timing arrays", "radio", r"IPTA|NANOGrav|EPTA|PPTA|InPTA|CPTA|MPTA"),
    # CMB
    ("Planck", "CMB", r"Planck (?:satellite|mission|[Cc]ollaboration|20\d\d|data|results|maps?)"),
    ("Simons Observatory", "CMB", r"Simons Observatory"),
    ("CMB-S4", "CMB", r"CMB-S4"),
    ("SPT", "CMB", r"SPT(?:-3G)?|South Pole Telescope"),
    ("ACT", "CMB", r"Atacama Cosmology Telescope"),
    ("LiteBIRD", "CMB", r"LiteBIRD"),
    # solar / heliospheric
    ("Solar Orbiter", "solar", r"Solar Orbiter"),
    ("Parker Solar Probe", "solar", r"Parker Solar Probe"),
    ("DKIST", "solar", r"DKIST|Inouye Solar Telescope"),
    ("SDO", "solar", r"SDO|Solar Dynamics Observatory"),
    ("SOHO", "solar", r"SOHO|LASCO"),
    ("STEREO", "solar", r"STEREO"),
]


def _facility_regexes() -> dict[str, list[re.Pattern]]:
    return {name: [re.compile(rf"(?<!\w)(?:{pat})(?!\w)")] for name, _, pat in FACILITIES}


def facility_network(book: str | None = "aaskaii", min_hits_per_paper: int = 2,
                     min_shared_papers: int = 2) -> dict:
    """Which external facilities the papers name, across which sections, and
    which facilities are named together (co-mention edges) — the data behind
    a synergy network, computed rather than reconstructed.

    A paper counts for a facility when it names it at least
    *min_hits_per_paper* times (2 by default, to drop passing mentions).
    """
    where = {"doc_source": _book_tag(book)} if book else None
    chunks = [c for c in get_all_chunks() if matches_where(c["meta"], where)]
    counts = _mention_counts(_facility_regexes(), chunks)
    category = {name: cat for name, cat, _ in FACILITIES}

    papers: dict[str, set[str]] = {
        name: {s for s, n in hits.items() if n >= min_hits_per_paper} for name, hits in counts.items()
    }
    facilities = []
    for name, sources in papers.items():
        by_section: dict[str, int] = defaultdict(int)
        for s in sources:
            by_section[section_of(s) or "(unknown)"] += 1
        facilities.append({
            "facility": name, "category": category[name],
            "n_papers": len(sources), "n_sections": len(by_section),
            "total_mentions": sum(counts[name].values()),
            "sections": dict(sorted(by_section.items(), key=lambda kv: -kv[1])),
        })
    facilities.sort(key=lambda f: (-f["n_sections"], -f["n_papers"]))

    names = [f["facility"] for f in facilities if f["n_papers"]]
    edges = [
        {"a": a, "b": b, "shared_papers": len(papers[a] & papers[b])}
        for i, a in enumerate(names) for b in names[i + 1:]
        if len(papers[a] & papers[b]) >= min_shared_papers
    ]
    edges.sort(key=lambda e: -e["shared_papers"])
    return {
        "book": book or "all",
        "min_hits_per_paper": min_hits_per_paper,
        "facilities": [f for f in facilities if f["n_papers"]],
        "not_named": sorted(f["facility"] for f in facilities if not f["n_papers"]),
        "edges": edges,
    }


def verify_citation(paper: str, citation: str, book: str | None = None) -> dict:
    """Does *paper*'s reference list contain the work cited as *citation*
    (e.g. "Vacca et al. 2025", "(Condon & Ransom, 2016a)")? Matches first
    author and year against the citation database (rag/citation_db.py)."""
    from rag import citation_db

    m = re.search(r"((?:[a-z]+\s)*[A-Z][\w'\u2019-]+).*?\b((?:18|19|20)\d{2})", citation)
    if not m:
        raise ValueError("Give the citation as 'Surname [et al.] YEAR'")
    surname = citation_db._fold(m[1].split()[-1])  # "van der Hulst" → "hulst", as parse_ref keys it
    year = int(m[2])
    sources = build_filter(book=book, paper=paper)["source"]
    conn = citation_db.connect()
    marks = ",".join("?" * len(sources))
    n_refs = conn.execute(f"SELECT COUNT(*) FROM cites WHERE source_path IN ({marks})", sources).fetchone()[0]
    rows = conn.execute(
        f"""SELECT r.first_author, r.year, COALESCE(c.raw, r.raw) AS raw
            FROM cites c JOIN refs r ON r.ref_key = c.ref_key
            WHERE c.source_path IN ({marks}) AND r.first_author = ?""", (*sources, surname)).fetchall()
    exact = [r["raw"] for r in rows if r["year"] == year]
    return {
        "paper": paper, "citation": citation, "first_author": surname, "year": year,
        "found": bool(exact),
        "matches": exact,
        "same_author_other_years": sorted({f"{r['year']}: {r['raw'][:160]}" for r in rows if r["year"] != year}),
        "references_parsed_for_paper": n_refs,
        "note": "" if n_refs else "No reference list parsed for this paper, so absence proves nothing.",
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


def paper_chunk(paper: str, chunk_index: int, book: str | None = None) -> dict:
    """One indexed chunk ({"text", "meta"}) by paper ID and chunk number, as
    shown on each search passage ("Paper ID: Vacca01 (chunk 12)"). *book*
    picks between the two books when a paper ID exists in both."""
    sources = build_filter(book=book, paper=paper)["source"]
    matches = [c for c in get_all_chunks()
               if c["meta"]["source"] in sources and c["meta"].get("chunk_index") == chunk_index]
    if not matches:
        raise ValueError(f"No chunk {chunk_index} in paper {paper!r}")
    if len(matches) > 1:
        raise ValueError(f"Paper ID {paper!r} exists in both books; pass book=")
    return matches[0]


def verify_quote(text: str, paper: str | None = None, top_k: int = 5,
                 chunk_index: int | None = None, book: str | None = None) -> dict:
    """Check where a claimed sentence/figure actually appears in the corpus.

    Combines ranked search (semantic + BM25 + rerank) with a literal scan for
    chunks containing every number in *text*. With *paper*, reports whether
    that paper supports the claim or whether it only appears elsewhere
    (a likely citation swap). With *paper* and *chunk_index*, also checks that
    exact passage.
    """
    numbers = list(dict.fromkeys(_NUMBER_RE.findall(text)))
    if chunk_index is not None and not paper:
        raise ValueError("chunk_index needs paper")

    def _ranked(where: dict | None, k: int) -> list[dict]:
        chunks = retrieve(text, top_k=k, use_hyde=False, expansion_window=0, where=where)
        return [_describe(c["source"], c["page_no"], c["text"], numbers, c["score"]) for c in chunks]

    corpus = _ranked(None, top_k)
    claimed = _ranked(build_filter(book=book, paper=paper), 3) if paper else []
    chunk_match = None
    if chunk_index is not None:
        c = paper_chunk(paper, chunk_index, book)
        chunk_match = _describe(c["meta"]["source"], c["meta"].get("page_no", 0), c["text"],
                                numbers, rerank_score(text, c["text"]))
        claimed.insert(0, chunk_match)

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
        "chunk_match": chunk_match,
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
            "pdf_url": (AASKAII_PDF_URL.format(stem=_stem(k))
                        if book_of(bibliography[k]) == DOC_SOURCE_AASKAII else None),
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


def tune_verify_threshold(claims_path: Path, seed: int = 0) -> dict:
    """Score labelled paraphrased claims against their true paper (positive)
    and a random other paper from the same book section (negative, i.e. a
    citation swap), the way verify_quote judges number-free claims: the best
    cross-encoder score among the claimed paper's top 3 chunks. Returns the
    best-F1 cut-off with its precision/recall."""
    import json
    import random

    rng = random.Random(seed)
    claims = json.loads(claims_path.read_text())
    by_section: dict[str, list[str]] = defaultdict(list)
    for s in _indexed_sources():
        if doc_source_of(s) == DOC_SOURCE_AASKAII:
            by_section[section_of(s)].append(_stem(s))

    def best(claim: str, paper: str) -> float:
        chunks = retrieve(claim, top_k=3, use_hyde=False, expansion_window=0,
                          where=build_filter(book="aaskaii", paper=paper))
        return max((c["score"] for c in chunks), default=0.0)

    scored = []  # (score, is_true_paper)
    for c in claims:
        section = section_of(build_filter(book="aaskaii", paper=c["paper"])["source"][0])
        other = rng.choice([p for p in by_section[section] if p != c["paper"]])
        scored += [(best(c["claim"], c["paper"]), True), (best(c["claim"], other), False)]

    def f1_at(t: float) -> tuple[float, float, float]:
        tp = sum(1 for s, y in scored if s >= t and y)
        fp = sum(1 for s, y in scored if s >= t and not y)
        fn = sum(1 for s, y in scored if s < t and y)
        p, r = tp / (tp + fp or 1), tp / (tp + fn or 1)
        return (2 * p * r / (p + r) if p + r else 0.0), p, r

    t_best = max((s for s, _ in scored), key=lambda t: f1_at(t)[0])
    f1, p, r = f1_at(t_best)
    return {
        "n_cases": len(scored),
        "best_threshold": round(t_best, 3),
        "f1": round(f1, 3), "precision": round(p, 3), "recall": round(r, 3),
        "current_threshold": VERIFY_MIN_SCORE,
        "current_f1_precision_recall": [round(x, 3) for x in f1_at(VERIFY_MIN_SCORE)],
        "true_paper_scores": sorted(round(s, 3) for s, y in scored if y),
        "other_paper_scores": sorted(round(s, 3) for s, y in scored if not y),
    }


if __name__ == "__main__":
    import json
    import sys

    if "--tune-verify" in sys.argv:
        print(json.dumps(tune_verify_threshold(
            Path(__file__).resolve().parent.parent / "tests/data/verify_claims.json"), indent=1))
