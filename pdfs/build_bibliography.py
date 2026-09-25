"""Build pdfs/bibliography.json from the live SKAO AASKAII contents page.

The "Advancing Astrophysics with the SKA II" contents page
(https://www.skao.int/en/aaskaii) lists every chapter as: an `<h3><strong>`
title, a `<p>` paragraph with the author list followed by an
`AASKAII/<chapter-id>` marker (the same id used as the local PDF filename,
e.g. `Ma01.pdf`), grouped under numbered `<h2>` section headings. A
previously saved local PDF snapshot of this page (contents_webpage.pdf) was
found to be missing most entries (the page appears to lazy-load content that
a "print to PDF" doesn't capture), so this script fetches the live page HTML
instead. This is a one-time, offline build step -- the app itself never
fetches this page at runtime; it only reads the generated JSON.

Usage:
    source .venv/bin/activate
    python pdfs/build_bibliography.py
"""

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

PAGE_URL = "https://www.skao.int/en/aaskaii"
OUTPUT_PATH = Path(__file__).parent / "bibliography.json"

_SECTION_RE = re.compile(r"^\d+\.\s*(.+)$")
_ID_RE = re.compile(r"AASKAII/([A-Za-z][\w\-]*)")
_NOISE_RE = re.compile(r"https?://\S+|\S+\.pdf\)?|ArXiv\([^)]*\)|\(/EN\)|\b(abstract|PDF|ArXiv)\b")


def _clean_authors(paragraph_text: str) -> list[str]:
    text = _ID_RE.sub("", paragraph_text)
    text = _NOISE_RE.sub("", text)
    text = text.replace("|", " ").replace("(", " ").replace(")", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return [a.strip() for a in text.split(",") if a.strip()]


def build_bibliography(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    entries: dict[str, dict] = {}
    current_section = "Unclassified"
    current_title: str | None = None

    for el in soup.find_all(["h2", "h3", "p"]):
        if el.name in ("h2", "h3"):
            text = el.get_text(strip=True)
            section_match = _SECTION_RE.match(text)
            if section_match:
                current_section = section_match.group(1).strip()
            elif el.find("strong"):
                # Titles are usually <p><strong>, but the very first (front
                # matter) entry uses <h3><strong>.
                current_title = text
            continue

        # el.name == "p"
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if "AASKAII/" not in text:
            if el.find("strong") and not el.find("em"):
                current_title = text
            continue
        if current_title is None:
            continue
        match = _ID_RE.search(text)
        if not match:
            continue
        chapter_id = match.group(1).rstrip(").,")
        authors = _clean_authors(text)
        if authors:
            entries[chapter_id] = {
                "title": current_title,
                "authors": authors,
                "section": current_section,
            }
        current_title = None

    return entries


def main() -> None:
    html = requests.get(PAGE_URL, timeout=60).text
    entries = build_bibliography(html)
    # Keep the "AASKA2015/"-keyed entries written by build_bibliography_aaska2015.py.
    existing = json.loads(OUTPUT_PATH.read_text()) if OUTPUT_PATH.exists() else {}
    entries |= {k: e for k, e in existing.items() if k.startswith("AASKA2015/")}
    OUTPUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {len(entries)} bibliography entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
