"""
Database of the references cited by the indexed papers.

    python -m rag.citation_db        # rebuild citations.db, print top 20

Pipeline: take each paper's "References" chunks (joined in order, since a
reference can straddle a chunk boundary) → split into entries → parse first
author, year, volume/page, DOI, arXiv id → key each entry so the same work
cited by different papers collapses to one row. Offline and cheap, so every
build starts from scratch.

Two reference styles occur:
  AASKA2015  Kaiser, N., 1987, MNRAS, 227, 1
  AASKAII    - M. A. Zwaan et al. MNRAS , 359(1):L30-L34, May 2005. doi: 10...

Tables
------
refs   one row per distinct cited work (ref_key), with one example raw string
cites  which corpus paper cites which ref_key, with that paper's raw string
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

from rag.bibliography import bib_key, section_of
from rag.config import CITATIONS_DB_PATH

CORPUS_BOOKS = ("aaskaii", "aaska2015")  # capability docs' "Reference Documents" aren't papers
REF_HEADING_RE = re.compile(r"(?:^|[\s:])(?:references|reference list|bibliography)\s*$", re.I)

YEAR_RE = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
# PDF extraction line-wraps DOIs after "/", ".", "-" or ":", leaving a space
# ("10.1111/j.1365-2966. 2009.16188.x", "10.1103/ PhysRevD.88.021302").
DOI_RE = re.compile(r"(?i:doi:\s*|doi\.org/)(10\.\d{4,9}(?=/)(?:/\s(?=\w)|[.:-]\s(?=[a-z0-9])|\S)+)")
ARXIV_RE = re.compile(r"arXiv(?:\.org/(?:abs|pdf)/|[:\s]*)((?:astro-ph|gr-qc|hep-\w+|physics)/\d{7}|\d{4}\.\d{4,5})", re.I)
GIVEN_FIRST_RE = re.compile(r"^(?:[A-Z][a-z]?\.[\s-]*)+(?=[^\W\d])")  # "M. A. Zwaan", "X.-N. Bai"
AUTHOR_START_RE = re.compile(r"^[A-Z][^\s,]*,?\s+(?:[A-Z][a-z]?\.|et al)")  # "Condon J.J", "Kaiser, N."
# Chapters of the two books themselves, cited by report number or PoS id.
AASKAII_RE = re.compile(r"Report number\s*AASKAII/([\w-]+)", re.I)
AASKA14_RE = re.compile(r"PoS\s*\(\s*AASKA14\s*\)\s*0*(\d+)|\(AASKA14\)\s*,\s*page\s*(\d+)", re.I)
# AASKA2015 lines where several references ran together: "...147, 73 Schwarz, D. et al."
RUN_ON_RE = re.compile(r"(?<=\d)\s+(?=(?:[a-z]+\s)?[A-Z][^\s,]*,\s+[A-Z]\.)")
VOL_PAGE_GIVEN_FIRST = re.compile(r"\b(\d+)(?:\(\d+\))?:\s*([A-Za-z]*\d+)")  # 359(1):L30
VOL_PAGE_SURNAME_FIRST = re.compile(r",\s*([A-Z]?\d+)\s*,\s*(?:(?:article id\.|art\.|p\.)\s*)?([A-Za-z]?\d+)\b")

# Journals as cited -> (abbreviation, ADS bibstem). Keys are the journal
# name lowercased with spaces and full stops removed. Used to name the
# journal and to build ADS bibcodes offline (no network): references are
# grouped by DOI or bibcode, so the two books' citation styles meet.
JOURNALS: dict[str, tuple[str, str]] = {}
for _abbr, _stem, _names in (
    ("ApJ", "ApJ", "apj astrophysj theastrophysicaljournal astrophysicaljournal"),
    ("ApJL", "ApJ", "apjl apjlett astrophysjlett apjletters astrophysjl theastrophysicaljournalletters"),
    ("ApJS", "ApJS", "apjs astrophysjsuppl astrophysjsupplser theastrophysicaljournalsupplementseries"),
    ("MNRAS", "MNRAS", "mnras monnotrastronsoc monthlynoticesoftheroyalastronomicalsociety"),
    ("A&A", "A&A", "a&a aa astronastrophys astronomyandastrophysics astronomy&astrophysics"),
    ("AJ", "AJ", "aj astronj theastronomicaljournal astronomicaljournal"),
    ("ARA&A", "ARA&A", "ara&a araa annurevastronastrophys"),
    ("A&ARv", "A&ARv", "a&arv aarv astronastrophysrev"),
    ("PASA", "PASA", "pasa publastronsocaust"),
    ("PASP", "PASP", "pasp publastronsocpac"),
    ("PASJ", "PASJ", "pasj publastronsocjpn"),
    ("Nature", "Natur", "nature"),
    ("Science", "Sci", "science"),
    ("NatAs", "NatAs", "natastron natureastronomy"),
    ("PhRvD", "PhRvD", "physrevd prd physicalreviewd"),
    ("PhRvL", "PhRvL", "physrevlett prl physicalreviewletters"),
    ("JCAP", "JCAP", "jcap jcosmolastropartphys"),
    ("SSRv", "SSRv", "ssrv spacescirev spacesciencereviews"),
    ("SoPh", "SoPh", "solphys"),
    ("Icarus", "Icar", "icarus"),
):
    for _name in _names.split():
        JOURNALS[_name] = (_abbr, _stem)
# Folded forms a journal takes when a parser mistakes it for a surname
# ("S. P. Reynolds. ARA&A , 46:89" once gave first author "araa").
JOURNAL_ABBREVIATIONS = {re.sub(r"[^a-z0-9]", "", k) for k in JOURNALS} | {"d", "physics", "lett", "rev"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS refs (
    ref_key TEXT PRIMARY KEY,
    raw TEXT NOT NULL,
    first_author TEXT, year INTEGER, volume TEXT, page TEXT, doi TEXT, arxiv TEXT,
    journal TEXT, bibcode TEXT, first_author_suspect INTEGER
);
CREATE TABLE IF NOT EXISTS cites (
    ref_key TEXT NOT NULL REFERENCES refs(ref_key),
    paper TEXT, source_path TEXT NOT NULL, book TEXT, section TEXT, raw TEXT,
    UNIQUE (source_path, ref_key)
);
CREATE INDEX IF NOT EXISTS cites_ref ON cites(ref_key);
CREATE INDEX IF NOT EXISTS refs_doi ON refs(doi);
CREATE INDEX IF NOT EXISTS refs_arxiv ON refs(arxiv);
CREATE INDEX IF NOT EXISTS refs_bibcode ON refs(bibcode);
"""


# Columns added after the first release; older databases get them on connect
# (empty until the next build_citation_db).
_ADDED_COLUMNS = {"journal": "TEXT", "bibcode": "TEXT", "first_author_suspect": "INTEGER"}


def connect(path: Path | str = CITATIONS_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    have = {r[1] for r in conn.execute("PRAGMA table_info(refs)")}
    if have:  # an existing database: add new columns before indexing them
        for col, kind in _ADDED_COLUMNS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE refs ADD COLUMN {col} {kind}")
    conn.executescript(SCHEMA)
    return conn


def _fold(s: str) -> str:
    """Lowercase ASCII alphanumerics only: "Martínez-Henares" -> "martinezhenares"."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def split_entries(text: str) -> list[str]:
    """One string per reference. Bulleted lists ("- ...") start an entry per
    bullet; otherwise per line, gluing a line onto a year-less predecessor
    unless it looks like a new author (heals chunk-boundary splits)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bulleted = sum(line.startswith("- ") for line in lines) > len(lines) / 2
    entries: list[str] = []
    for line in lines:
        if bulleted:
            new = line.startswith("- ")
        else:
            new = not entries or bool(YEAR_RE.search(entries[-1])) or bool(AUTHOR_START_RE.match(line))
        line = line.removeprefix("- ")
        if new or not entries:
            entries.append(line)
        else:
            entries[-1] += " " + line
    if not bulleted:
        entries = [part for entry in entries for part in RUN_ON_RE.split(entry)]
    return entries


# End of a given-first author list: "Reynolds. ", "et al. ", "Gerhard. "
# (a full stop after a whole word, not after an initial like "J. ").
AUTHORS_END_RE = re.compile(r"et al\.?\s|(?<=[^\W\d_]{2})\.\s")
VOL_PAGE_RAW_GIVEN = re.compile(r"\b(\d+)(?:\(\d+\))?:\s*([A-Za-z]?\d+)")


def journal_of(name: str | None) -> tuple[str, str] | None:
    """(abbreviation, ADS bibstem) for a journal as cited, if known."""
    if not name:
        return None
    return JOURNALS.get(re.sub(r"[\s.]", "", name.lower()))


def bibcode(year: int | None, journal: str | None, volume: str | None, page: str | None,
            initial: str | None) -> str | None:
    """ADS bibcode built offline (YYYYJJJJJVVVVMPPPPA), or None when a part is
    missing or doesn't fit the format (five-digit pages, unknown journal).
    Letter page prefixes (ApJL "L24", A&A "A68") go in the qualifier."""
    j = journal_of(journal)
    if not (year and j and volume and page and initial) or not volume.isdigit():
        return None
    abbr, stem = j
    m = re.fullmatch(r"([A-Za-z]?)(\d+)", page)
    if not m or len(m[2]) > 4 or len(volume) > 4:
        return None
    qualifier = (m[1] or ("L" if abbr == "ApJL" else ".")).upper()
    return f"{year}{stem:.<5}{volume:.>4}{qualifier}{m[2]:.>4}{initial.upper()}"


def parse_ref(entry: str) -> dict | None:
    """Fields and ref_key for one reference string (None if unparseable).

    Key preference: AASKAII report number / AASKA14 PoS id (chapters of the
    books themselves), then the ADS bibcode built from journal, year, volume,
    page and first initial (the same in both citation styles), then
    surname|year|volume|page for journals without a known bibstem, then arXiv
    id, then DOI, then surname|year|journal, then surname|year|start of the
    text. Refs sharing a DOI, arXiv id or bibcode are merged after parsing
    (_merge_shared_ids), so a citation with a DOI and one without still meet.
    # ponytail: bibcodes are built offline, not looked up; an arXiv-only
    # citation and a journal citation without the arXiv id/DOI stay separate.
    """
    doi_m = DOI_RE.search(entry)
    doi = re.sub(r"\s+", "", doi_m[1]).rstrip(".").lower() if doi_m else None
    if doi and doi.endswith("/"):  # prefix only: would merge every paper from that publisher
        doi = None
    arxiv_m = ARXIV_RE.search(entry)
    arxiv = arxiv_m[1] if arxiv_m else None
    if doi and doi.startswith("10.48550/arxiv") and arxiv:
        doi = f"10.48550/arxiv.{arxiv.lower()}"
    body = entry[: doi_m.start()] if doi_m else entry
    body = ARXIV_RE.sub("", body)

    given = GIVEN_FIRST_RE.match(body)
    years = list(YEAR_RE.finditer(body))
    year_m = (years[-1] if given else years[0]) if years else None
    year = int(year_m[1]) if year_m else None
    journal = None
    if given:
        # The first author's name runs to the first ",", " et al", " and" or
        # the full stop ending the author list ("S. P. Reynolds. ARA&A , 46"
        # is Reynolds, not ARA&A); the journal follows the author list.
        name = re.split(r",| et al\b| and |(?<=[^\W\d_]{2})\.\s", body[given.end():], maxsplit=1)[0].strip()
        first = name.split()
        surname = first[-1] if first else ""
        end = AUTHORS_END_RE.search(body, given.end())
        if end:
            jm = re.match(r"\s*(.+?)\s*,\s*(?:art\.|\d)", body[end.end():])
            journal = jm[1].strip() if jm else None
    else:
        m = re.match(r"(?:[a-z]+\s+)*([^\s,.]+)", body)  # skip particles: "de Blok" -> "Blok"
        surname = m[1] if m else ""
        name = body.split(",")[0].strip()
        if year_m:
            jm = re.match(r"[a-z]?[,.]?\s*([^,\d][^,]*?)\s*,\s*[A-Z]?\d", body[year_m.end():])
            journal = jm[1].strip(" .'") if jm else None
    initial = next((ch for ch in name if ch.isalpha()), None)
    surname = _fold(surname)

    if given:
        vp = VOL_PAGE_GIVEN_FIRST.search(body)
        raw_vp = VOL_PAGE_RAW_GIVEN.search(body)
    else:
        vp = raw_vp = VOL_PAGE_SURNAME_FIRST.search(body, year_m.end() if year_m else 0)
    volume, page = (_fold(vp[1]).lstrip("0"), _fold(vp[2]).lstrip("0")) if vp else (None, None)
    known = journal_of(journal)
    bib = bibcode(year, journal, volume, raw_vp[2].lstrip("0") if raw_vp else None,
                  unicodedata.normalize("NFKD", initial).encode("ascii", "ignore").decode() if initial else None)

    aaskaii, aaska14 = AASKAII_RE.search(entry), AASKA14_RE.search(entry)
    if aaskaii:
        key = f"aaskaii:{aaskaii[1].lower()}"
    elif aaska14:
        key = f"aaska14:{int(aaska14[1] or aaska14[2])}"
    elif bib:
        key = f"bib:{bib}"
    elif surname and year and volume and page:
        key = f"{surname}|{year}|{volume}|{page}"
    elif arxiv:  # before DOI: arXiv DOIs (10.48550/arXiv.NNNN) often extract broken
        key = f"arxiv:{arxiv.lower()}"
    elif doi:
        key = f"doi:{doi}"
    elif surname and year and known:
        key = f"{surname}|{year}|{_fold(known[0])}"
    elif surname and year:
        tail = body[given.end():] if given else body[year_m.end():]
        key = f"{surname}|{year}|{_fold(tail)[:40]}"
    else:
        return None
    return {"ref_key": key, "raw": entry, "first_author": surname or None, "year": year,
            "volume": volume, "page": page, "doi": doi, "arxiv": arxiv,
            "journal": known[0] if known else journal, "bibcode": bib,
            "first_author_suspect": int(surname in JOURNAL_ABBREVIATIONS)}


def reference_texts(chunks: list[dict]) -> dict[str, tuple[str, str]]:
    """{source_path: (book, joined reference-section text)} for corpus papers."""
    by_source: dict[str, list[tuple[int, str]]] = {}
    books: dict[str, str] = {}
    for chunk in chunks:
        meta = chunk["meta"]
        if meta.get("doc_source") not in CORPUS_BOOKS or not REF_HEADING_RE.search(meta.get("heading") or ""):
            continue
        by_source.setdefault(meta["source"], []).append((meta.get("chunk_index", 0), chunk["text"]))
        books[meta["source"]] = meta["doc_source"]
    return {src: (books[src], "\n".join(t for _, t in sorted(parts))) for src, parts in by_source.items()}


def record_citations(conn: sqlite3.Connection, chunks: list[dict]) -> int:
    """Replace the tables with citations parsed from *chunks*; returns cites."""
    conn.executescript("DROP TABLE IF EXISTS cites; DROP TABLE IF EXISTS refs;" + SCHEMA)
    for source, (book, text) in reference_texts(chunks).items():
        paper, section = bib_key(source), section_of(source)
        for entry in split_entries(text):
            ref = parse_ref(entry)
            if ref is None:
                continue
            conn.execute("INSERT OR IGNORE INTO refs VALUES (:ref_key, :raw, :first_author, :year,"
                         " :volume, :page, :doi, :arxiv, :journal, :bibcode, :first_author_suspect)", ref)
            conn.execute("INSERT OR IGNORE INTO cites VALUES (?, ?, ?, ?, ?, ?)",
                         (ref["ref_key"], paper, source, book, section, entry))
    _merge_shared_ids(conn)
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM cites").fetchone()[0]


def _key_rank(key: str) -> tuple:
    """Which ref_key survives a merge: a book chapter, then a bibcode, then a DOI, then any."""
    return (0 if key.startswith("aask") else 1 if key.startswith("bib:") else 2 if key.startswith("doi:") else 3, key)


def _merge_shared_ids(conn: sqlite3.Connection) -> None:
    """Fold refs that are the same work into one ref_key: sharing a DOI, a
    bibcode, an arXiv id, or first author + year + volume + page (the last
    joins a citation whose journal the PDF text lost to one with a bibcode,
    and a work cited under two journal names to itself). The surviving key
    prefers a bibcode, and inherits identifiers the others had."""
    rows = [dict(r) for r in conn.execute("SELECT * FROM refs")]
    parent = {r["ref_key"]: r["ref_key"] for r in rows}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    seen: dict[tuple, str] = {}
    for r in rows:
        ids = [("doi", r["doi"]), ("bibcode", r["bibcode"]), ("arxiv", r["arxiv"])]
        if all(r[c] for c in ("first_author", "year", "volume", "page")):
            ids.append(("avp", f"{r['first_author']}|{r['year']}|{r['volume']}|{r['page']}"))
        for ident in ids:
            if ident[1] is None:
                continue
            if ident in seen:
                a, b = find(seen[ident]), find(r["ref_key"])
                if a != b:
                    keep, drop = sorted((a, b), key=_key_rank)
                    parent[drop] = keep
            else:
                seen[ident] = r["ref_key"]

    by_key = {r["ref_key"]: r for r in rows}
    groups: dict[str, list[str]] = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    for keep, members in groups.items():
        if len(members) == 1:
            continue
        for col in ("doi", "bibcode", "arxiv", "journal"):
            if by_key[keep][col] is None:
                value = next((by_key[m][col] for m in sorted(members, key=_key_rank) if by_key[m][col]), None)
                if value:
                    conn.execute(f"UPDATE refs SET {col} = ? WHERE ref_key = ?", (value, keep))
        for m in members:
            if m != keep:
                # A paper citing both variants keeps one row (UNIQUE source_path, ref_key).
                conn.execute("UPDATE OR IGNORE cites SET ref_key = ? WHERE ref_key = ?", (keep, m))
                conn.execute("DELETE FROM cites WHERE ref_key = ?", (m,))
                conn.execute("DELETE FROM refs WHERE ref_key = ?", (m,))


def build_citation_db(path: Path = CITATIONS_DB_PATH) -> dict:
    """Rebuild from all indexed chunks; also lists corpus papers with no
    parsed references (missing reference section or a parse failure)."""
    from rag.retrieval import get_all_chunks

    chunks = get_all_chunks()
    conn = connect(path)
    record_citations(conn, chunks)
    corpus = {bib_key(c["meta"]["source"]) for c in chunks if c["meta"].get("doc_source") in CORPUS_BOOKS}
    cited = {r[0] for r in conn.execute("SELECT DISTINCT paper FROM cites")}
    return {**stats(conn), "papers_without_refs": sorted(corpus - cited)}


def stats(conn: sqlite3.Connection) -> dict:
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "refs": one("SELECT COUNT(*) FROM refs"),
        "cites": one("SELECT COUNT(*) FROM cites"),
        "papers_with_refs": one("SELECT COUNT(DISTINCT source_path) FROM cites"),
        "refs_with_doi": one("SELECT COUNT(*) FROM refs WHERE doi IS NOT NULL"),
        "refs_with_bibcode": one("SELECT COUNT(*) FROM refs WHERE bibcode IS NOT NULL"),
        "first_author_suspect": one("SELECT COUNT(*) FROM refs WHERE first_author_suspect"),
    }


def top_cited(conn: sqlite3.Connection, book: str | None = None, section: str | None = None,
              since_year: int | None = None, min_papers: int = 1, limit: int = 50) -> list[dict]:
    """Cited works, most-citing-papers first, with the corpus papers citing each."""
    where, params = ["1"], []
    for column, value in (("c.book", book), ("c.section", section)):
        if value is not None:
            where.append(f"{column} = ?")
            params.append(value)
    if since_year is not None:
        where.append("r.year >= ?")
        params.append(since_year)
    rows = conn.execute(
        f"""SELECT r.*, COUNT(DISTINCT c.source_path) AS n_papers,
                   GROUP_CONCAT(DISTINCT c.paper) AS papers
            FROM refs r JOIN cites c ON c.ref_key = r.ref_key
            WHERE {' AND '.join(where)}
            GROUP BY r.ref_key HAVING n_papers >= ?
            ORDER BY n_papers DESC, r.year LIMIT ?""",
        (*params, min_papers, limit),
    ).fetchall()
    return [{**dict(r), "papers": sorted(r["papers"].split(","))} for r in rows]


def citing_papers(conn: sqlite3.Connection, text: str, limit: int = 100) -> list[dict]:
    """Corpus papers whose reference lists contain *text* (case-insensitive
    substring, e.g. "Condon" or "1998, AJ, 115"), grouped by cited work."""
    rows = conn.execute(
        """SELECT r.ref_key, r.raw, COUNT(DISTINCT c.source_path) AS n_papers,
                  GROUP_CONCAT(DISTINCT c.paper) AS papers
           FROM cites c JOIN refs r ON r.ref_key = c.ref_key
           WHERE c.raw LIKE ? GROUP BY r.ref_key
           ORDER BY n_papers DESC LIMIT ?""",
        (f"%{text}%", limit),
    ).fetchall()
    return [{**dict(r), "papers": sorted(r["papers"].split(","))} for r in rows]


if __name__ == "__main__":
    result = build_citation_db()
    print({k: v for k, v in result.items() if k != "papers_without_refs"})
    print(f"{len(result['papers_without_refs'])} papers without refs: {result['papers_without_refs']}")
    for row in top_cited(connect(), limit=20):
        print(f"{row['n_papers']:4d}  {row['raw'][:110]}")
