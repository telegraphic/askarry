"""Download AASKAII chapter PDFs from the SKAO website into AASKAII/<section>/.

Re-runnable: chapters already present anywhere under AASKAII/ (whichever
section folder they ended up in) are skipped, so only new papers download.

    python pdfs/download.py
"""

import re
import sys
from pathlib import Path

import requests
from loguru import logger

# Allow running as `python pdfs/download.py` from the repo root (or from
# inside pdfs/) while still importing the shared `rag` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import SECTION_ORDER

PAGE_URL = "https://www.skao.int/en/science-users/aaskaii"

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "AASKAII"
ERRORS_PATH = HERE / "download_errors.txt"


def build_topic_map(html: str) -> dict[str, str]:
    """Map each chapter PDF URL on the live page to the section heading that
    precedes it. (The old version parsed a saved PDF of the page, where long
    headings wrapped across lines and went unmatched, which misfiled every
    "From the Milky Way to Distant Galaxies" chapter under the previous
    section.)"""
    headings = sorted(
        (m.start(), h) for h in SECTION_ORDER for m in re.finditer(re.escape(h), html)
    )
    topic_map = {}
    for m in re.finditer(r'https?://[^"\']+\.pdf', html, flags=re.IGNORECASE):
        before = [h for pos, h in headings if pos < m.start()]
        topic_map.setdefault(m.group(0), before[-1] if before else "Unclassified")
    return topic_map


def main() -> None:
    html = requests.get(PAGE_URL, timeout=60).text
    topic_map = build_topic_map(html)
    logger.info(f"Found {len(topic_map)} candidate PDFs")

    existing = {p.name.lower() for p in OUTPUT_DIR.rglob("*.pdf")}
    failures = []

    for url, topic in sorted(topic_map.items()):
        filename = url.split("/")[-1]
        if filename.lower() in existing:
            continue

        try:
            r = requests.get(url, timeout=120, allow_redirects=True)
            if r.status_code == 404:
                failures.append(url)
                logger.warning(f"404  {filename}")
                continue
            r.raise_for_status()
            folder = OUTPUT_DIR / topic
            folder.mkdir(parents=True, exist_ok=True)
            (folder / filename).write_bytes(r.content)
            logger.info(f"OK   {filename} -> {topic}")
        except Exception as e:
            failures.append(f"{url} : {e}")
            logger.error(f"FAIL {filename}")

    ERRORS_PATH.write_text("".join(f"{item}\n" for item in failures))
    logger.info(f"Skipped {len(existing)} already downloaded; failures: {len(failures)}")


if __name__ == "__main__":
    main()
