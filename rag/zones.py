"""
Document structure for indexed chunks: which part of a paper each chunk is
(its *zone*), which indexed files are copies of another, and a fingerprint of
the whole index.

Corpus counts used to treat every chunk alike, so reference lists ("ApJ",
"Planck Collaboration 2020"), affiliations ("Max-Planck-Institut") and
author lists ("Chandra, P.") were counted as if the paper discussed them.
Zones let the corpus tools count the abstract and body only. Duplicate
detection keeps re-downloaded copies (Curiel01c, Kurahara01_3, ...) from
counting as extra papers.

Everything here is a pure function of the chunks' text and metadata, so
`python ingest.py --retag` gives the same zones and fingerprint every time it
runs over the same chunks, and new rules only need a new ZONE_RULES_VERSION.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ZONES = ("front_matter", "abstract", "body", "acknowledgements", "references", "appendix")
CONTENT_ZONES = ["abstract", "body"]  # what the corpus tools count by default
ZONE_RULES_VERSION = "1"
DEDUP_RULES_VERSION = "1"
MANIFEST_NAME = "manifest.json"

# Documents sharing at least this fraction of their (normalised) chunk texts
# are copies of one another (Jaccard similarity of the chunk-hash sets).
# ponytail: exact chunk hashes only; switch to word-shingle MinHash if a
# re-download ever chunks differently and a known copy is missed.
NEAR_DUPLICATE_JACCARD = 0.5

# --- Zone rules -------------------------------------------------------------

_REFERENCES_HEADING = re.compile(r"^\W*(?:\d+\.?\s*)?(references|bibliography|literature cited|works cited)\b", re.I)
_ACK_HEADING = re.compile(r"\backnowledge?ments?\b|\bfunding\b", re.I)
_FRONT_HEADING = re.compile(
    r"\b(author list|affiliations?|table of contents|contents|list of (figures|tables)|author contributions?)\b", re.I)
_APPENDIX_HEADING = re.compile(r"^\W*(appendix|appendices|index)\b|^[A-H](\.\d+)*\s+[A-Z]", re.I)
_ABSTRACT_HEADING = re.compile(r"^\W*(abstract|summary of the chapter)\W*$", re.I)
# A numbered section heading ("1 Introduction", "2.3 Results"): the body has started.
_SECTION_HEADING = re.compile(r"^\W*\d+(\.\d+)*\.?\s+\S")

# One reference-list entry: a DOI, an arXiv id, "vol(issue):page" or
# "Journal , vol:page". Body text cites "(Smith et al., 2020)", which matches none.
_REF_ENTRY = re.compile(
    r"doi:\s*10\.|\barXiv[:\s]\s*\d|\b\d{1,4}\s*\(\d{1,3}\)\s*:\s*[A-Z]?\d+"
    r"|\b(?:ApJ|MNRAS|A&A|AJ|PASA|PASP|ARA&A|Nature|Science|JCAP|Phys\.\s*Rev)\w*\s*,\s*\d{1,4}\b")
# Affiliation and author-list cues.
_AFFILIATION = re.compile(
    r"\b(Universit\w*|Institut\w*|Observator\w*|Department|Dipartimento|Laborator\w*|School of|"
    r"Faculty of|Centre for|Center for|Max[- ]Planck|INAF|CSIRO|CNRS|Academy of Sciences|Research Cent\w*)\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_AFFIL_MARKS = re.compile(r"\w\s*,\s*\d{1,2}\s*(?:,\s*\d{1,2}\s*)*,\s*[A-Z]")  # "Maccagni , 1 , 2 Vincenzo"


def _is_reference_text(text: str) -> bool:
    """A chunk that reads as a reference list: one entry-like match per
    ~250 characters (at least three)."""
    return len(_REF_ENTRY.findall(text)) >= max(3, len(text) / 250)


def _is_front_matter_text(text: str) -> bool:
    """Author lists, affiliation blocks and contact e-mails."""
    affil = len(_AFFILIATION.findall(text)) + 2 * len(_EMAIL.findall(text)) + len(_AFFIL_MARKS.findall(text))
    return affil >= max(3, len(text) / 300)


def assign_zones(chunks: list[dict]) -> list[str]:
    """Zone of each chunk of ONE document, given as {"text", "meta"} dicts.

    Walks the chunks in reading order (meta["chunk_index"]) as a small state
    machine, because a zone is mostly a property of where a chunk sits:

    - Before the first numbered section heading, a chunk is front matter
      (title, authors, affiliations, e-mails) or, if it reads as prose, the
      abstract. Docling gives the paper title as the heading of that region.
    - Headings (Docling's section structure) switch zone: References,
      Acknowledgements, Appendix, Contents, "Full author list and affiliations".
    - References are sticky: Docling often promotes reference fragments
      ("0004-6361/202348444.") to headings, so only a heading that starts a
      recognised later part (acknowledgements, appendix, author list) or a
      chunk that no longer reads as references leaves the zone.
    - A body chunk that reads as a reference list (an un-headed bibliography)
      is a reference chunk.

    Returns zones in the same order as *chunks*.
    """
    order = sorted(range(len(chunks)), key=lambda i: chunks[i]["meta"].get("chunk_index", 0))
    zones = [""] * len(chunks)
    title_heading = (chunks[order[0]]["meta"].get("heading") or "") if chunks else ""
    state = "front"  # front -> body; references/acknowledgements/appendix/front_matter as headings say
    last_heading = None
    for i in order:
        text, meta = chunks[i]["text"], chunks[i]["meta"]
        heading = (meta.get("heading") or "").strip()
        path = meta.get("section_path") or heading
        new_heading = heading != last_heading
        last_heading = heading

        if new_heading or state == "front":
            if _REFERENCES_HEADING.search(heading):
                state = "references"
            elif _ACK_HEADING.search(heading):
                state = "acknowledgements"
            elif _FRONT_HEADING.search(path):
                state = "front_matter"
            elif _ABSTRACT_HEADING.search(heading):
                state = "abstract"
            elif _APPENDIX_HEADING.search(heading) and state in ("references", "acknowledgements", "body", "appendix"):
                state = "appendix"
            elif state == "references" and new_heading and _is_reference_text(text):
                pass  # a reference fragment Docling took for a heading
            elif state == "front" and heading == title_heading and not _SECTION_HEADING.search(heading):
                pass  # still in the title/author/abstract region
            elif new_heading or state in ("abstract", "front"):
                state = "body"
        # A reference list printed under another heading (Hota01 prints its
        # references under "12 Acknowledgement"): switch, and stay switched
        # until a heading that isn't a reference fragment.
        if state in ("body", "acknowledgements", "appendix") and _is_reference_text(text):
            state = "references"

        if state == "front":
            zone = "front_matter" if (_is_front_matter_text(text) or meta.get("chunk_index", 0) == 0) else "abstract"
        elif state == "references" and not new_heading and not _is_reference_text(text) and _is_front_matter_text(text):
            zone = "front_matter"  # affiliations printed after the reference list
        elif state == "body" and _is_front_matter_text(text) and _EMAIL.search(text):
            zone = "front_matter"
        else:
            zone = state
        zones[i] = zone
    return zones


# --- Duplicates --------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha1(_norm(text).encode()).hexdigest()


def find_duplicates(chunks: list[dict], doc_info: dict[str, dict]) -> dict[str, dict]:
    """{source: {"status", "duplicate_of", "reason"}} for every source in *chunks*.

    status is "canonical", "duplicate" (a copy of another indexed file) or
    "unlisted" (a book document with no bibliography entry: contents pages,
    stray PDFs). *doc_info* gives, per source, {"book": bool (a book chapter
    source), "listed": bool (has a bibliography entry), "section_match": bool
    (its folder agrees with its bibliography section), "doc_source", "stem"}.

    Copies are found two ways:
    1. Content: documents whose chunk-hash sets overlap by at least
       NEAR_DUPLICATE_JACCARD. Within each group the copy kept is the one
       with a bibliography entry whose folder agrees with it, then the
       shorter stem ("Curiel01" over "Curiel01c").
    2. Paper ID: two files with the same stem in the same book (Bianchi01 in
       two folders). Run after (1), so a file already shown to be a copy of
       another paper doesn't make the real one look like a duplicate.
    """
    hashes: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        hashes[c["meta"]["source"]].add(text_hash(c["text"]))

    # Candidate pairs share at least one chunk hash; compare only those.
    by_hash: dict[str, list[str]] = defaultdict(list)
    for src, hs in hashes.items():
        for h in hs:
            by_hash[h].append(src)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for srcs in by_hash.values():
        if 1 < len(srcs) < 20:  # boilerplate shared by many files is not evidence
            for a in srcs:
                for b in srcs:
                    if a < b:
                        shared[a, b] += 1

    parent = {s: s for s in hashes}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for (a, b), n in shared.items():
        if n / len(hashes[a] | hashes[b]) >= NEAR_DUPLICATE_JACCARD:
            parent[find(a)] = find(b)

    def rank(src: str) -> tuple:
        info = doc_info.get(src, {})
        return (not info.get("listed", True), not info.get("section_match", True), len(info.get("stem", src)), src)

    out = {s: {"status": "canonical", "duplicate_of": "", "reason": ""} for s in hashes}
    groups: dict[str, list[str]] = defaultdict(list)
    for s in hashes:
        groups[find(s)].append(s)
    for members in groups.values():
        keep = min(members, key=rank)
        for s in members:
            if s != keep:
                out[s] = {"status": "duplicate", "duplicate_of": keep,
                          "reason": "same text as another indexed file"}

    by_id: dict[tuple, list[str]] = defaultdict(list)
    for s in hashes:
        if out[s]["status"] == "canonical":
            info = doc_info.get(s, {})
            by_id[info.get("doc_source"), info.get("stem", Path(s).stem)].append(s)
    for members in by_id.values():
        keep = min(members, key=rank)
        for s in members:
            if s != keep:
                out[s] = {"status": "duplicate", "duplicate_of": keep,
                          "reason": "same paper ID as another indexed file, filed in a section its bibliography entry disagrees with"}

    for s, info in doc_info.items():
        if s in out and out[s]["status"] == "canonical" and info.get("book") and not info.get("listed"):
            out[s] = {"status": "unlisted", "duplicate_of": "", "reason": "no bibliography entry"}
    return out


# --- Fingerprint and manifest ------------------------------------------------


def _rel(source: str, root: Path) -> str:
    try:
        return Path(source).resolve().relative_to(root).as_posix()
    except ValueError:
        return source


def index_fingerprint(chunks: list[dict], root: Path) -> str:
    """sha256 over every chunk's (path relative to *root*, chunk_index, text
    hash), sorted, so moving the repository or reordering storage doesn't
    change it but any added, removed or edited chunk does."""
    lines = sorted(f"{_rel(c['meta']['source'], root)}\t{c['meta'].get('chunk_index', 0)}\t{text_hash(c['text'])}"
                   for c in chunks)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_manifest(chunks: list[dict], root: Path, embedding_model: str, max_chunk_tokens: int,
                   git_sha: str | None = None) -> dict:
    """What a rebuilt index must match to reproduce today's counts."""
    return {
        "index_fingerprint": index_fingerprint(chunks, root),
        "n_docs": len({c["meta"]["source"] for c in chunks}),
        "n_chunks": len(chunks),
        "zone_rules_version": ZONE_RULES_VERSION,
        "dedup_rules_version": DEDUP_RULES_VERSION,
        # Versions installed when the manifest was written. Retagging keeps the
        # original parse, so these record the tagging run, not the first parse.
        "parser": {"docling": _package_version("docling"), "docling-core": _package_version("docling-core")},
        "embedding_model": embedding_model,
        "max_chunk_tokens": max_chunk_tokens,
        "git_sha": git_sha,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(db_dir: Path, manifest: dict) -> None:
    (db_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")


def read_manifest(db_dir: Path) -> dict | None:
    path = db_dir / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else None


def manifest_mtime(db_dir: Path) -> float:
    path = db_dir / MANIFEST_NAME
    return path.stat().st_mtime if path.exists() else 0.0
