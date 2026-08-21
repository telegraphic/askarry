import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger
from pypdf import PdfReader

# Allow running as `python pdfs/download.py` from the repo root (or from
# inside pdfs/) while still importing the shared `rag` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import SECTION_ORDER

PDF_INDEX = "Advancing Astrophysics with the SKA II _ SKAO.pdf"
PAGE_URL = "https://www.skao.int/en/aaskaii"

OUTPUT_DIR = Path("AASKAII")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Extract topic mapping from downloaded webpage PDF
# ------------------------------------------------------------

def build_topic_map(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += "\n" + (page.extract_text() or "")

    topic_map = {}

    current_topic = "Unclassified"

    for line in text.splitlines():

        line = line.strip()

        for heading in SECTION_ORDER:
            if heading in line:
                current_topic = heading
                break

        m = re.findall(r"AASKAII/([A-Za-z0-9_-]+)", line)

        for chapter in m:
            topic_map[chapter] = current_topic

    return topic_map


topic_map = build_topic_map(PDF_INDEX)

logger.info(f"Mapped {len(topic_map)} chapters")

# ------------------------------------------------------------
# Scrape PDF links
# ------------------------------------------------------------

html = requests.get(PAGE_URL, timeout=60).text

pdf_links = set(
    re.findall(
        r'https?://[^"\']+\.pdf',
        html,
        flags=re.IGNORECASE,
    )
)

# fallback URLs taken from chapter identifiers
for chapter in topic_map:
    pdf_links.add(
        f"https://www.skao.int/sites/default/files/documents/{chapter}.pdf"
    )

logger.info(f"Found {len(pdf_links)} candidate PDFs")

# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

failures = []

for url in sorted(pdf_links):

    filename = url.split("/")[-1]

    stem = Path(filename).stem

    topic = topic_map.get(stem, "Unclassified")

    folder = OUTPUT_DIR / topic
    folder.mkdir(parents=True, exist_ok=True)

    outfile = folder / filename

    if outfile.exists():
        logger.info(f"SKIP {filename} (already downloaded)")
        continue

    try:

        r = requests.get(
            url,
            timeout=120,
            allow_redirects=True,
        )

        if r.status_code == 404:
            failures.append(url)
            logger.warning(f"404  {filename}")
            continue

        r.raise_for_status()

        outfile.write_bytes(r.content)

        logger.info(f"OK   {filename} -> {topic}")

    except Exception as e:

        failures.append(f"{url} : {e}")

        logger.error(f"FAIL {filename}")

# ------------------------------------------------------------
# Save error log
# ------------------------------------------------------------

with open("download_errors.txt", "w") as f:

    for item in failures:
        f.write(str(item) + "\n")

logger.info("Downloaded successfully")
logger.info(f"Failures: {len(failures)}")