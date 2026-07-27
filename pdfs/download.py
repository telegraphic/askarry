import re
import shutil
import requests
from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup
from urllib.parse import urljoin

PDF_INDEX = "Advancing Astrophysics with the SKA II _ SKAO.pdf"
PAGE_URL = "https://www.skao.int/en/aaskaii"

OUTPUT_DIR = Path("AASKAII")
OUTPUT_DIR.mkdir(exist_ok=True)

TOPIC_HEADINGS = [
    "Science Working Group Overviews",
    "Sun, Earth and Planets",
    "Formation and Evolution of Stars",
    "From the Milky Way to Distant Galaxies",
    "The Cosmos",
    "The Extreme Universe",
    "Methods and Techniques",
]

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

        for heading in TOPIC_HEADINGS:
            if heading in line:
                current_topic = heading
                break

        m = re.findall(r"AASKAII/([A-Za-z0-9_-]+)", line)

        for chapter in m:
            topic_map[chapter] = current_topic

    return topic_map


topic_map = build_topic_map(PDF_INDEX)

print(f"Mapped {len(topic_map)} chapters")

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

print(f"Found {len(pdf_links)} candidate PDFs")

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

    try:

        r = requests.get(
            url,
            timeout=120,
            allow_redirects=True,
        )

        if r.status_code == 404:
            failures.append(url)
            print(f"404  {filename}")
            continue

        r.raise_for_status()

        outfile.write_bytes(r.content)

        print(f"OK   {filename} -> {topic}")

    except Exception as e:

        failures.append(f"{url} : {e}")

        print(f"FAIL {filename}")

# ------------------------------------------------------------
# Save error log
# ------------------------------------------------------------

with open("download_errors.txt", "w") as f:

    for item in failures:
        f.write(str(item) + "\n")

print()
print(f"Downloaded successfully")
print(f"Failures: {len(failures)}")