"""Title/author lookup for AASKAII chapter PDFs.

Backed by the static `pdfs/bibliography.json` file generated offline by
`pdfs/build_bibliography.py` (run manually, never at app runtime). Used to
render nicer citations in the Source Passages section and to build the
searchable Table of Contents tab.
"""

import json
from functools import lru_cache
from pathlib import Path

from rag import store
from rag.config import (
    AASKA2015_YEAR,
    AASKAII_YEAR,
    DOC_SOURCE_AASKA2015,
    DOC_SOURCE_AASKAII,
    BIBLIOGRAPHY_PATH,
    PDF_DIRS,
    SECTION_ORDER,
    SUPPORTED_SUFFIXES,
    doc_source_for,
)


@lru_cache(maxsize=1)
def load_bibliography() -> dict:
    """Load and cache pdfs/bibliography.json. Returns {} if missing."""
    if not BIBLIOGRAPHY_PATH.exists():
        return {}
    return json.loads(BIBLIOGRAPHY_PATH.read_text())


def bib_key(source: str | Path) -> str:
    """bibliography.json key for a file: its filename stem, prefixed
    "AASKA2015/" for AASKA2015 chapters, since 15 stems (e.g. Vacca01) exist
    in both books."""
    stem = Path(source).stem
    return f"AASKA2015/{stem}" if doc_source_for(source) == DOC_SOURCE_AASKA2015 else stem


def lookup_citation(source: str | Path) -> dict | None:
    """Look up the bibliography entry for a chunk's source path."""
    return load_bibliography().get(bib_key(source))


def book_of(entry: dict) -> str:
    """doc_source tag of the book a bibliography entry belongs to."""
    return DOC_SOURCE_AASKA2015 if entry.get("year") == AASKA2015_YEAR else DOC_SOURCE_AASKAII


def section_of(source: str) -> str:
    """Book section of a file ("" if unknown).

    Prefers the bibliography entry, then falls back to the
    pdfs/<book>/<section>/ folder name (not preferred outright:
    pdfs/download.py filed every "From the Milky Way to Distant Galaxies"
    paper under "Formation and Evolution of Stars").
    """
    entry = lookup_citation(source)
    if entry:
        return entry["section"]
    path = Path(source)
    return path.parent.name if path.parent.parent.name in ("AASKAII", "AASKA2015") else ""


def format_citation(entry: dict) -> str:
    """Format a bibliography entry as "{Surname} (et al) ({year}) {title}".

    *year* comes from the entry itself when present (e.g. AASKA2015 chapters,
    tagged by pdfs/build_bibliography_aaska2015.py), else defaults to
    AASKAII_YEAR for the original (year-less) AASKAII entries.
    """
    authors = entry.get("authors") or []
    title = entry.get("title", "")
    year = entry.get("year", AASKAII_YEAR)
    if authors:
        surname = authors[0].split()[-1]
        who = f"{surname} et al" if len(authors) > 1 else surname
    else:
        who = "Unknown"
    return f"{who} ({year}) {title}"


def list_toc_entries(doc_source: str | None = None) -> dict[str, list[dict]]:
    """
    Build the Table of Contents grouped by section (in book order), sorted
    alphabetically by title within each group. Local files with no matching
    bibliography entry are excluded.

    When *doc_source* is given (see config.DOC_SOURCE_*), only chapters from
    that report are included — so AASKA2015 and AASKAII chapters can be
    browsed as separate tabs without mixing.
    """
    grouped: dict[str, list[dict]] = {}
    for path in store.discover_files(PDF_DIRS, SUPPORTED_SUFFIXES):
        entry = lookup_citation(path)
        if entry is None:
            continue
        if doc_source is not None and doc_source_for(path) != doc_source:
            continue
        grouped.setdefault(section_of(path), []).append(
            {
                "title": entry["title"],
                "authors": entry.get("authors", []),
                "path": path,
            }
        )

    ordered_sections = SECTION_ORDER + sorted(set(grouped) - set(SECTION_ORDER))
    return {
        section: sorted(grouped[section], key=lambda e: e["title"].lower())
        for section in ordered_sections
        if section in grouped
    }
