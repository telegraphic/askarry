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
    AASKAII_YEAR,
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


def lookup_citation(source: str) -> dict | None:
    """Look up the bibliography entry for a chunk's source path, by filename stem."""
    bibliography = load_bibliography()
    return bibliography.get(Path(source).stem)


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


def _discover_pdf_stems() -> dict[str, Path]:
    """Recursively find all supported files under PDF_DIRS, keyed by filename stem."""
    return {path.stem: path for path in store.discover_files(PDF_DIRS, SUPPORTED_SUFFIXES)}


def list_toc_entries(doc_source: str | None = None) -> dict[str, list[dict]]:
    """
    Build the Table of Contents grouped by section (in book order), sorted
    alphabetically by title within each group. Local files with no matching
    bibliography entry are excluded.

    When *doc_source* is given (see config.DOC_SOURCE_*), only chapters from
    that report are included — so AASKA2015 and AASKAII chapters can be
    browsed as separate tabs without mixing.
    """
    bibliography = load_bibliography()
    stems = _discover_pdf_stems()

    grouped: dict[str, list[dict]] = {}
    for chapter_id, path in stems.items():
        entry = bibliography.get(chapter_id)
        if entry is None:
            continue
        if doc_source is not None and doc_source_for(path) != doc_source:
            continue
        grouped.setdefault(entry["section"], []).append(
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
