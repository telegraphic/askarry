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
from rag.zones import CONTENT_ZONES, ZONES

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


def doc_status(meta: dict) -> str:
    """"canonical", "duplicate" or "unlisted" (see rag/zones.find_duplicates).
    Chunks from an index not yet retagged count as canonical."""
    return meta.get("doc_status", "canonical")


def corpus_chunks(book: str | None = "aaskaii", include_noncanonical: bool = False,
                  zones: list[str] | None = None) -> list[dict]:
    """Chunks the corpus tools count: one book (None = all), canonical
    documents only unless *include_noncanonical* (re-downloaded copies and
    unlisted book documents are otherwise left out), and only the given
    *zones* (None = every zone; an index not yet retagged has no zones, so
    its chunks all count as "body")."""
    where = {"doc_source": _book_tag(book)} if book else None
    return [
        c for c in get_all_chunks()
        if matches_where(c["meta"], where)
        and (include_noncanonical or doc_status(c["meta"]) == "canonical")
        and (zones is None or c["meta"].get("zone", "body") in zones)
    ]


# Words that end in "s" but are not plurals, so _singular leaves them alone.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "ous", "ics", "ays")

# Other meanings of words that are also facility names, left out of counts
# unless strict matching is asked for. Keys match a term alternative (or a
# FACILITIES name) case-insensitively. Each guard is a regex matched
# case-insensitively; a mention inside a guard's match is not counted.
TERM_EXCLUSIONS: dict[str, list[str]] = {
    "Planck": [r"Max[- ]Planck\w*", r"Planck(?:'s)?\s+(?:and\s+\w+\s+)?(?:constants?|function|law|mass|scale|length|time|units?|"
               r"spectrum|distribution|curve|formula|radiation|regime)", r"Planck\s+Collaborations?",
               r"Planck\s+et\s+al\.?", r"Planck,?\s+\(?(?:19|20)\d\d"],
    "Chandra": [r"Chandra\s+et\s+al\.?", r"Chandra,?\s+\(?(?:19|20)\d\d", r"Chandra\s+and\s+[A-Z]\w+,?\s+\(?(?:19|20)\d\d"],
    "Swift": [r"Swift\s+et\s+al\.?", r"Swift,?\s+\(?(?:19|20)\d\d"],
    "Fermi": [r"Fermi(?:[- ]Dirac|\s+(?:energy|level|surface|acceleration|gas|momentum|sea|statistics|bubbles?|"
              r"mechanism|process))", r"Fermi\s+et\s+al\.?", r"Fermi,?\s+\(?(?:19|20)\d\d"],
    "Gaia": [r"Gaia[- ](?:Enceladus|Sausage)\w*", r"Gaia\s+Collaboration", r"Gaia\s+et\s+al\.?"],
    "Euclid": [r"Euclid's", r"Euclid\s+Collaboration", r"Euclid\s+et\s+al\.?"],
}


def _singular(word: str) -> str:
    """Strip a plural ending ("surveys" -> "survey", "frequencies" ->
    "frequency"), leaving "analysis", "mass", "radius" alone."""
    low = word.lower()
    if len(word) > 4 and low.endswith("ies"):
        return word[:-3] + ("Y" if word[-3:].isupper() else "y")
    if len(word) > 4 and low.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and low.endswith("s") and not low.endswith(_NOT_PLURAL_ENDINGS) and not word.isupper():
        return word[:-1]
    return word


def _inflected(word: str) -> str:
    """Regex for one word in singular or plural form."""
    if word.isupper() and len(word) > 1:  # acronym: FRB, FRBs
        return re.escape(word) + "s?"
    base = _singular(word)
    low = base.lower()
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return re.escape(base[:-1]) + "(?:y|ies)"
    if low.endswith(("s", "x", "ch", "sh", "z")):
        return re.escape(base) + "(?:es)?"
    return re.escape(base) + "(?:s|es)?"


def _alternative_regex(alt: str, strict: bool) -> re.Pattern:
    """One "|"-separated alternative. All-caps alternatives match
    case-sensitively (so "HI" doesn't match "hi"); everything else ignores case.

    strict: the literal, whole-word match keyword_coverage always used.
    Otherwise words also match in singular or plural, and words may be joined
    by spaces or hyphens ("zoom mode" = "zoom-mode" = "zoom modes").
    """
    flags = 0 if alt.isupper() else re.IGNORECASE
    if strict:
        return re.compile(rf"\b{re.escape(alt)}\b", flags)
    words = [w for w in re.split(r"[\s\-]+", alt) if w]
    return re.compile(r"(?<!\w)" + r"[\s\-]+".join(_inflected(w) for w in words) + r"(?!\w)", flags)


def _term_regexes(term: str, acronyms: dict[str, str], strict: bool = True) -> list[re.Pattern]:
    """One regex per "|"-separated alternative, plus the full form of any
    all-caps acronym alternative (see _alternative_regex for matching)."""
    alts = [a.strip() for a in term.split("|") if a.strip()]
    alts += [acronyms[a] for a in list(alts) if a.isupper() and a in acronyms]
    return [_alternative_regex(a, strict) for a in dict.fromkeys(alts)]


def _exclusion_regex(names: list[str], extra: list[str] | None, curated: bool) -> re.Pattern | None:
    """One regex for every guard that applies: the curated TERM_EXCLUSIONS for
    any of *names*, plus the caller's own patterns in *extra* (each a regex,
    or "|"-separated phrases matched as in _alternative_regex)."""
    keys = {k.lower(): k for k in TERM_EXCLUSIONS}
    guards = [g for n in names if curated and n.strip().lower() in keys for g in TERM_EXCLUSIONS[keys[n.strip().lower()]]]
    guards += list(extra or [])
    return re.compile("|".join(f"(?:{g})" for g in guards), re.IGNORECASE) if guards else None


def _matches(regexes: list[re.Pattern], exclude: re.Pattern | None, text: str) -> list[re.Match]:
    """Matches of any regex in *text*, minus those overlapping an exclusion."""
    found = [m for r in regexes for m in r.finditer(text)]
    if exclude is None or not found:
        return found
    spans = [m.span() for m in exclude.finditer(text)]
    return [m for m in found if not any(a < m.end() and m.start() < b for a, b in spans)]


def _mention_counts(patterns: dict[str, list[re.Pattern]], chunks: list[dict],
                    excludes: dict[str, re.Pattern | None] | None = None) -> dict[str, dict[str, int]]:
    """{label: {source path: mentions}} for each label's regexes over *chunks*,
    not counting matches inside that label's exclusion regex."""
    excludes = excludes or {}
    counts: dict[str, dict[str, int]] = {label: defaultdict(int) for label in patterns}
    for c in chunks:
        for label, regexes in patterns.items():
            n = len(_matches(regexes, excludes.get(label), c["text"]))
            if n:
                counts[label][c["meta"]["source"]] += n
    return counts


def _snippet(text: str, m: re.Match, width: int = 120) -> str:
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("…" if a else "") + re.sub(r"\s+", " ", text[a:b]).strip() + ("…" if b < len(text) else "")


def keyword_coverage(terms: list[str], book: str | None = "aaskaii",
                     min_hits_per_paper: int = 1, include_noncanonical: bool = False,
                     zones: list[str] | None = CONTENT_ZONES, strict: bool = False,
                     exclude: list[str] | None = None, use_curated_exclusions: bool = True,
                     return_contexts: int = 0, usage_stats: bool = False) -> dict:
    """Exact per-section / per-paper mention counts for each term.

    A paper counts as covering a term when the term (or an alternative /
    acronym expansion) appears at least *min_hits_per_paper* times in it.
    Papers are counted by file path, once each; copies of another indexed
    file and unlisted book documents are left out unless
    *include_noncanonical*.

    Only chunks in *zones* are counted (default abstract and body, so
    reference lists, affiliations and acknowledgements don't count; None for
    every zone). Words match in singular or plural and with spaces or hyphens
    unless *strict*. Mentions inside an *exclude* pattern don't count, nor,
    unless *strict* or not *use_curated_exclusions*, inside the curated
    TERM_EXCLUSIONS for that word ("Max-Planck", "Chandra et al."). With
    *return_contexts* k, each paper lists its first k matched snippets; with
    *usage_stats*, mentions per 1,000 body words and whether the term is in a
    heading or the abstract.
    """
    if zones is not None:
        unknown = sorted(set(zones) - set(ZONES))
        if unknown:
            raise ValueError(f"Unknown zones {unknown}; expected some of {list(ZONES)}")
    every = corpus_chunks(book, include_noncanonical)
    chunks = [c for c in every if zones is None or c["meta"].get("zone", "body") in zones]
    acronyms = get_acronyms()

    section_sizes: dict[str, set] = defaultdict(set)
    for c in every:
        section_sizes[section_of(c["meta"]["source"]) or "(unknown)"].add(c["meta"]["source"])

    patterns = {t: _term_regexes(t, acronyms, strict) for t in terms}
    excludes = {t: _exclusion_regex(t.split("|"), exclude, use_curated_exclusions and not strict) for t in terms}
    all_hits = _mention_counts(patterns, chunks, excludes)
    stats = _usage_stats(patterns, excludes, every) if usage_stats else {}

    results = []
    for term in terms:
        hits = all_hits[term]
        counted = {s for s, n in hits.items() if n >= min_hits_per_paper}  # paths: each file counts once
        by_section: dict[str, list[str]] = defaultdict(list)
        for source in counted:
            by_section[section_of(source) or "(unknown)"].append(_stem(source))
        row = {
            "term": term,
            "total_mentions": sum(hits.values()),
            "n_papers": len(counted),
            "n_sections": len(by_section),
            "sections": {
                s: {"n_papers": len(p), "papers": sorted(p)}
                for s, p in sorted(by_section.items(), key=lambda kv: -len(kv[1]))
            },
        }
        if return_contexts:
            row["contexts"] = _contexts(patterns[term], excludes[term], chunks, counted, return_contexts)
        if usage_stats:
            row["usage"] = {_stem(s): stats[term][s] for s in sorted(counted, key=_stem)}
        results.append(row)
    results.sort(key=lambda r: (-r["n_sections"], -r["n_papers"]))
    return {
        "book": book or "all",
        "min_hits_per_paper": min_hits_per_paper,
        "zones": zones or "all",
        "matching": "strict (literal, whole word)" if strict else "inflected (plurals, space/hyphen variants)",
        "exclusions": {t: excludes[t].pattern for t in terms if excludes[t] is not None},
        "papers_per_section": {s: len(p) for s, p in sorted(section_sizes.items())},
        "terms": results,
    }


def _contexts(regexes: list[re.Pattern], exclude: re.Pattern | None, chunks: list[dict],
              sources: set[str], k: int) -> dict[str, list[dict]]:
    """First *k* matched snippets per counted paper, in reading order."""
    out: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(chunks, key=lambda c: (c["meta"]["source"], c["meta"].get("chunk_index", 0))):
        src = c["meta"]["source"]
        if src not in sources or len(out[_stem(src)]) >= k:
            continue
        for m in _matches(regexes, exclude, c["text"])[:k - len(out[_stem(src)])]:
            out[_stem(src)].append({"text": _snippet(c["text"], m), "zone": c["meta"].get("zone", "body"),
                                    "heading": c["meta"].get("heading", ""), "page": c["meta"].get("page_no")})
    return dict(out)


def _usage_stats(patterns: dict[str, list[re.Pattern]], excludes: dict[str, re.Pattern | None],
                 chunks: list[dict]) -> dict[str, dict[str, dict]]:
    """{term: {source: {mentions_per_1k_body_words, in_heading, in_abstract}}},
    over every zone of each paper (body words count the body zone only)."""
    body_words: dict[str, int] = defaultdict(int)
    for c in chunks:
        if c["meta"].get("zone", "body") == "body":
            body_words[c["meta"]["source"]] += len(c["text"].split())
    out: dict[str, dict] = {}
    for term, regexes in patterns.items():
        per: dict[str, dict] = defaultdict(lambda: {"body_mentions": 0, "in_heading": False, "in_abstract": False})
        for c in chunks:
            m, src, zone = c["meta"], c["meta"]["source"], c["meta"].get("zone", "body")
            n = len(_matches(regexes, excludes[term], c["text"]))
            if zone == "body":
                per[src]["body_mentions"] += n
            if zone == "abstract" and n:
                per[src]["in_abstract"] = True
            if _matches(regexes, None, m.get("section_path") or m.get("heading") or ""):
                per[src]["in_heading"] = True
        out[term] = {
            src: {"mentions_per_1k_body_words": round(1000 * v["body_mentions"] / body_words[src], 2) if body_words[src] else None,
                  "in_heading": v["in_heading"], "in_abstract": v["in_abstract"]}
            for src, v in per.items()
        }
    return out


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
                     min_shared_papers: int = 2, include_noncanonical: bool = False,
                     zones: list[str] | None = CONTENT_ZONES, exclude: list[str] | None = None,
                     use_curated_exclusions: bool = True) -> dict:
    """Which external facilities the papers name, across which sections, and
    which facilities are named together (co-mention edges) — the data behind
    a synergy network, computed rather than reconstructed.

    A paper counts for a facility when it names it at least
    *min_hits_per_paper* times (2 by default, to drop passing mentions).
    Copies and unlisted documents are left out unless *include_noncanonical*.
    Only chunks in *zones* count (default abstract and body). Mentions inside
    the curated TERM_EXCLUSIONS for a facility (Max-Planck, "Chandra et al.",
    the Fermi level) or an *exclude* pattern don't count.
    """
    chunks = corpus_chunks(book, include_noncanonical, zones)
    regexes = _facility_regexes()
    counts = _mention_counts(regexes, chunks, {name: _exclusion_regex([name], exclude, use_curated_exclusions)
                                               for name in regexes})
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
            "papers": sorted(_stem(s) for s in sources),
        })
    facilities.sort(key=lambda f: (-f["n_sections"], -f["n_papers"]))

    names = [f["facility"] for f in facilities if f["n_papers"]]
    edges = [
        {"a": a, "b": b, "shared_papers": len(papers[a] & papers[b]),
         "papers": sorted(_stem(s) for s in papers[a] & papers[b])}
        for i, a in enumerate(names) for b in names[i + 1:]
        if len(papers[a] & papers[b]) >= min_shared_papers
    ]
    edges.sort(key=lambda e: -e["shared_papers"])
    return {
        "book": book or "all",
        "min_hits_per_paper": min_hits_per_paper,
        "zones": zones or "all",
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


def index_health(book: str | None = None) -> dict:
    """State of the index: duplicates, unlisted documents, chapters with no
    PDF, chunk and zone counts per document, documents whose reference list
    wasn't found (a sign the zone rules missed it), and whether the manifest
    still matches the chunks."""
    from rag import zones as zone_rules
    from rag.config import CHROMA_DIR, HERE

    tag = _book_tag(book)
    chunks = [c for c in get_all_chunks() if tag is None or c["meta"].get("doc_source") == tag]
    docs: dict[str, dict] = {}
    for c in chunks:
        m = c["meta"]
        d = docs.setdefault(m["source"], {"paper": bib_key(m["source"]), "book": _BOOK_NAMES.get(m.get("doc_source"), m.get("doc_source")),
                                          "section": section_of(m["source"]) or "(unknown)",
                                          "status": doc_status(m), "duplicate_of": m.get("duplicate_of", ""),
                                          "n_chunks": 0, "zones": defaultdict(int)})
        d["n_chunks"] += 1
        d["zones"][m.get("zone", "(untagged)")] += 1
    for d in docs.values():
        d["zones"] = dict(d["zones"])

    stems: dict[tuple, list[str]] = defaultdict(list)
    for s_, d in docs.items():
        stems[d["book"], d["paper"]].append(str(Path(s_).relative_to(HERE)) if Path(s_).is_relative_to(HERE) else s_)
    manifest = zone_rules.read_manifest(CHROMA_DIR)
    every = get_all_chunks()
    return {
        "manifest": manifest,
        "manifest_matches_index": bool(manifest) and manifest["index_fingerprint"] == zone_rules.index_fingerprint(every, HERE),
        "untagged_chunks": sum(1 for c in chunks if "zone" not in c["meta"]),
        "duplicate_paper_ids": {f"{b}:{p}": paths for (b, p), paths in stems.items() if len(paths) > 1},
        "duplicates": sorted(({"paper": d["paper"], "copy_of": d["duplicate_of"], "section": d["section"]}
                              for d in docs.values() if d["status"] == "duplicate"), key=lambda x: x["paper"]),
        "unlisted": sorted(d["paper"] for d in docs.values() if d["status"] == "unlisted"),
        "unclassified": sorted(d["paper"] for d in docs.values() if d["section"] in ("Unclassified", "(unknown)")),
        "no_references_zone": sorted(d["paper"] for d in docs.values()
                                     if d["status"] == "canonical" and d["book"] in ("aaskaii", "aaska2015")
                                     and "references" not in d["zones"]),
        "chapters_without_pdf": missing_papers(book)["not_indexed"],
        "documents": sorted(docs.values(), key=lambda d: d["paper"]),
    }


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
