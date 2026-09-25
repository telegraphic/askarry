"""Reorganize the flat, numerically-named AASKA2015 (PoS vol. 215, "Advancing
Astrophysics with the SKA", AASKA14 conference, published 2015) PDFs that
download_pos215.py drops into pdfs/PoS215/ (e.g. "001.pdf") into a proper
AASKAII-style layout: one subfolder per conference session, files renamed to
"<FirstAuthorSurname><NN>.pdf" (e.g. "Koopmans01.pdf", matching e.g.
pdfs/AASKAII/The Cosmos/Harrison01.pdf, Harrison02.pdf for repeats), moved
into pdfs/AASKA2015/<session>/. Also (re)writes the AASKA2015 entries in the
shared pdfs/bibliography.json, keyed "AASKA2015/<stem>" (see
rag/bibliography.py::bib_key) so stems shared with AASKAII (e.g. Vacca01)
don't overwrite the AASKAII entries — replacing any numeric-id- or
bare-stem-keyed AASKA2015 entries from a prior run of this script.

The contribution list, titles, authors and session groupings all come from
scraping the live https://pos.sissa.it/215/ contents page (same page
download_pos215.py scrapes for PDF links) — see scrape_contributions().

This is a one-time, offline build/migration step -- the app itself never
fetches this page or moves files at runtime.

Usage:
    source .venv/bin/activate
    python pdfs/build_bibliography_aaska2015.py
"""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

PAGE_URL = "https://pos.sissa.it/215/"
HERE = Path(__file__).parent
SOURCE_DIR = HERE / "PoS215"      # flat, id-named PDFs from download_pos215.py
TARGET_DIR = HERE / "AASKA2015"   # reorganized destination
OUTPUT_PATH = HERE / "bibliography.json"
AASKA2015_YEAR = 2015

_IDCODE_RE = re.compile(r"\)(\d+)\s*$")
_AUTHOR_SPLIT_RE = re.compile(r",\s*|\s+and\s+")
_SESSION_PREFIX_RE = re.compile(r"^Session\s+\d+:\s*(.+)$")


def _split_authors(text: str) -> list[str]:
    text = text.replace("\xa0", " ")
    return [a.strip() for a in _AUTHOR_SPLIT_RE.split(text) if a.strip()]


def _clean_section(raw: str) -> str:
    """Strip the "Session N: " numbering prefix so folder names read like
    AASKAII's topical folders (e.g. "The Cosmos") rather than "Session 2: ..".
    """
    match = _SESSION_PREFIX_RE.match(raw)
    return match.group(1).strip() if match else raw.strip()


def _surname(author: str) -> str:
    """Last whitespace-separated token, accents stripped for a safe filename
    (e.g. "L. Koopmans" -> "Koopmans"; "M. Della Valle" -> "Valle")."""
    name = unicodedata.normalize("NFKD", author.split()[-1])
    return "".join(c for c in name if not unicodedata.combining(c))


def scrape_contributions(html: str) -> list[dict]:
    """Return every contribution in page order:
    [{"contrib_id", "title", "authors", "section"}, ...].

    The contents page lists sessions as `<tr id="session-..."><th>...</th>
    </tr>` rows followed by one `<tr><td>` per paper (title/idcode/authors);
    some author lists are truncated behind a <details>/<summary> toggle —
    get_text() pulls the full list regardless.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="contribs-public")

    contributions = []
    current_section = "Unclassified"
    for row in table.find_all("tr"):
        header = row.find("th")
        if header is not None:
            current_section = header.get_text(strip=True)
            continue

        title_el = row.find("div", class_="title")
        idcode_el = row.find("div", class_="idcode")
        authors_el = row.find("div", class_="authors")
        if title_el is None or idcode_el is None:
            continue

        match = _IDCODE_RE.search(idcode_el.get_text(strip=True))
        if not match:
            continue

        contributions.append(
            {
                "contrib_id": match.group(1),
                "title": title_el.get_text(strip=True),
                "authors": _split_authors(authors_el.get_text(" ", strip=True)) if authors_el else [],
                "section": _clean_section(current_section),
            }
        )
    return contributions


def assign_filenames(contributions: list[dict]) -> dict[str, str]:
    """Map contrib_id -> "<Surname><NN>" filename stem, numbering repeats of
    the same first-author surname in page order (e.g. Harrison01, Harrison02)
    — the same convention already used under pdfs/AASKAII/."""
    counts: dict[str, int] = defaultdict(int)
    stems: dict[str, str] = {}
    for c in contributions:
        surname = _surname(c["authors"][0]) if c["authors"] else "Unknown"
        counts[surname] += 1
        stems[c["contrib_id"]] = f"{surname}{counts[surname]:02d}"
    return stems


def reorganize_files(contributions: list[dict], stems: dict[str, str]) -> None:
    """Move each flat id-named PDF from SOURCE_DIR into
    TARGET_DIR/<section>/<stem>.pdf."""
    for c in contributions:
        src = SOURCE_DIR / f"{c['contrib_id']}.pdf"
        if not src.exists():
            logger.warning(f"Missing PDF for contribution {c['contrib_id']} ({c['title']}) — skipping")
            continue
        dest_dir = TARGET_DIR / c["section"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{stems[c['contrib_id']]}.pdf"
        src.rename(dest)
        logger.info(f"{src.relative_to(HERE)} -> {dest.relative_to(HERE)}")


def main() -> None:
    html = requests.get(PAGE_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text
    contributions = scrape_contributions(html)

    stems = assign_filenames(contributions)
    reorganize_files(contributions, stems)
    if SOURCE_DIR.exists() and not any(SOURCE_DIR.iterdir()):
        SOURCE_DIR.rmdir()

    bibliography = json.loads(OUTPUT_PATH.read_text()) if OUTPUT_PATH.exists() else {}
    # Drop AASKA2015 entries from prior runs of this script: numeric-id-keyed
    # (pre-reorganize) or bare-stem-keyed (pre-"AASKA2015/" prefix). The
    # AASKAII entries those bare stems clobbered come back from
    # pdfs/build_bibliography.py.
    for c in contributions:
        bibliography.pop(c["contrib_id"], None)
    for key in [k for k, e in bibliography.items() if e.get("year") == AASKA2015_YEAR and "/" not in k]:
        del bibliography[key]
    for c in contributions:
        bibliography[f"AASKA2015/{stems[c['contrib_id']]}"] = {
            "title": c["title"],
            "authors": c["authors"],
            "section": c["section"],
            "year": AASKA2015_YEAR,
        }
    OUTPUT_PATH.write_text(json.dumps(bibliography, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {len(contributions)} AASKA2015 entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
