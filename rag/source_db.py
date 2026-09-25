"""
Database of astronomical sources named in the indexed papers.

    python -m rag.source_db                      # build/update sources.db (incremental)
    python -m rag.source_db --retry-unresolved   # also re-query past SIMBAD misses

fields    survey fields the papers name or define (one row per field per
          paper, with the paper's stated centre/area and a verbatim quote)
requests  observing requests stated in the papers (hours, position,
          constraints), each backed by a verbatim quote; input to
          rag.scheduling.lst_pressure

Pipeline: regex-scan every indexed chunk for catalogue-style and common
source names → look each distinct name up once in SIMBAD (batched, cached,
including "not found") → store resolved sources and every mention in SQLite.
Re-running only queries SIMBAD for names it has never seen.

Tables
------
sources   one row per SIMBAD object (unique main_id): position, type, redshift
names     every distinct name seen (the SIMBAD cache): source_id or unresolved
mentions  where each name appears: paper, book, section, chunk, page, context
"""

from __future__ import annotations

import re
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from rag.bibliography import section_of
from rag.config import SOURCES_DB_PATH

# Catalogue-style names. Unicode minus/en-dash are normalised to "-" first.
_SIGN = r"[+-]"
NAME_PATTERNS: list[str] = [
    r"(?:NGC|IC)\s?\d{1,4}[A-Z]?",
    # Messier 1-110 only; skip "M 10 kpc"-style quantities and UK postcodes ("M13 9PL").
    r"M\s?(?:110|10\d|[1-9]\d?)(?!\d|[.,]\d|\s?(?:yr|kpc|pc|Mpc|K|km)\b|\s\d[A-Z]{2}\b|\s?(?:V|dwarf|ultracool|star|type)\b)",
    # ponytail: citation shorthand like "(hereafter M22)" still matches; review via mentions.context
    r"UGC\s?\d{1,5}",
    r"PSRs?\s?[JB]\d{4}" + _SIGN + r"\d{2,4}[A-Za-z]?",
    r"FRB\s?\d{6,8}[A-Z]?",
    r"GRB\s?\d{6}[A-Z]?",
    r"SN\s?\d{4}[a-z]{1,3}",
    r"AT\s?20\d{2}[a-z]{2,3}",
    r"GW\d{6}(?:_\d{6})?",
    r"3C\s?\d{1,3}(?:\.\d)?",
    r"4C\s?" + _SIGN + r"?\d{2}\.\d{2}",
    r"PKS\s?[JB]?\d{4}" + _SIGN + r"\d{2,3}",
    r"Abell\s?\d{2,4}",
    r"(?:HD|HIP)\s?\d{3,6}",
    r"(?:SGR|XTE|GRS|MAXI|Swift)\s?J?\d{4}(?:\.\d)?" + _SIGN + r"\d{2,4}",
    r"(?:NVSS|SUMSS|2MASS|SDSS|TGSS|FIRST|RACS|VLASS|4FGL|3FGL)\s?J\d{4,6}(?:\.\d+)?" + _SIGN + r"\d{4,6}(?:\.\d+)?",
    r"(?:Cyg|Sco|Cen|Her|Vela|Cir|Aql|GX)\s?X-\d",
    # Common names SIMBAD knows (planets and the Sun are omitted: not in SIMBAD).
    r"Cen(?:taurus)? A|Cyg(?:nus)? A|Virgo A|Fornax A|Hydra A|Pictor A|Hercules A",
    r"Sgr A\*|Sagittarius A\*|Galactic Cent(?:re|er)",
    r"LMC|SMC|Large Magellanic Cloud|Small Magellanic Cloud|Magellanic Stream|Magellanic Bridge",
    r"47 Tuc(?:anae)?|Omega Cen(?:tauri)?|Crab (?:Nebula|pulsar)|Vela pulsar|Orion Nebula",
    r"Andromeda Galaxy|Coma Cluster|Virgo Cluster|Perseus Cluster|Bullet Cluster|Fornax Cluster",
    r"Proxima Cen(?:tauri)?|TRAPPIST-1|Barnard's Star",
]
# Survey/deep fields: (pattern, SIMBAD name, curated position). SIMBAD knows
# most fields only under a "NAME …" form, so the name as written is mapped
# to that. Fields SIMBAD lacks carry a curated position with its source;
# positions are never typed from memory. ELAIS-S1/N1 and Euclid Deep Field
# South are extracted but stay unresolved: neither SIMBAD nor the corpus
# gives a centre for them.
SURVEY_FIELDS: list[tuple[str, str | None, dict | None]] = [
    (r"COSMOS(?: [Ff]ield)?", "NAME COSMOS Field", None),
    (r"E-?CDF-?S|Extended Chandra Deep Field[- ]South", "NAME ECDFS", None),
    (r"CDF-?S|Chandra Deep Field[- ]South", "NAME CDFS", None),
    (r"GOODS-?S|GOODS[- ]South", "NAME GOODS South Field", None),
    (r"GOODS-?N|GOODS[- ]North", "NAME GOODS North Field", None),
    (r"Lockman Hole", "NAME Lockman Hole", None),
    (r"Hubble Deep Field[- ]South|HDF-?S", "NAME HDF-S", None),
    (r"Hubble Deep Field(?:[- ]North)?|HDF(?:-?N)?", "NAME HDF", None),
    (r"South Galactic Pole|SGP", "NAME SGP", None),
    (r"Bo(?:ö|o)tes (?:[Ff]ield|Deep Field)|NDWFS", "NAME NOAO Deep Wide Field", None),
    # deLeraAcedo01 p.13: "EoR0 (RA = 0 h, DEC = -27°), EoR1 (RA = 4 h,
    # DEC = -27°) and EoR2 (RA = 10.3 h, DEC = -10°)".
    (r"EoR0", None, {"ra": 0.0, "dec": -27.0, "source": "deLeraAcedo01 p.13"}),
    (r"EoR1", None, {"ra": 60.0, "dec": -27.0, "source": "deLeraAcedo01 p.13"}),
    (r"EoR2", None, {"ra": 154.5, "dec": -10.0, "source": "deLeraAcedo01 p.13"}),
    (r"XMM-LSS", "NAME XMM-LSS Field", None),
    (r"ELAIS[- ]?[NS]1", None, None),
    (r"Euclid Deep Field[- ]South|EDF-?S", None, None),
]
NAME_PATTERNS += [pattern for pattern, _, _ in SURVEY_FIELDS]


def _field(raw: str) -> tuple[str | None, dict | None] | None:
    """(SIMBAD name, curated position) if *raw* is a known survey field."""
    for pattern, simbad_name, curated in SURVEY_FIELDS:
        if re.fullmatch(pattern, raw):
            return simbad_name, curated
    return None


# Not glued to other tokens: rejects model/label names like "Jet-M01/M22".
NAME_RE = re.compile(r"(?<![\w*/-])(?:" + "|".join(NAME_PATTERNS) + r")(?![\w*/])")

SIMBAD_BATCH = 200
SIMBAD_PAUSE_S = 1.0  # ponytail: fixed pause between batches; SIMBAD asks for ≲5 queries/s
CONTEXT_CHARS = 150

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    main_id TEXT UNIQUE NOT NULL,
    ra_deg REAL, dec_deg REAL, gal_l_deg REAL, gal_b_deg REAL,
    object_type TEXT, redshift REAL,
    resolved_by TEXT, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS names (
    name_key TEXT PRIMARY KEY,
    raw_name TEXT NOT NULL,
    source_id INTEGER REFERENCES sources(id),
    status TEXT NOT NULL,            -- 'resolved' | 'unresolved'
    checked_at TEXT
);
CREATE TABLE IF NOT EXISTS mentions (
    name_key TEXT NOT NULL REFERENCES names(name_key),
    paper TEXT, source_path TEXT NOT NULL, book TEXT, section TEXT,
    chunk_index INTEGER NOT NULL, page INTEGER, context TEXT,
    UNIQUE (source_path, chunk_index, name_key)
);
CREATE INDEX IF NOT EXISTS mentions_name ON mentions(name_key);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_id INTEGER REFERENCES sources(id),
    ra_deg REAL, dec_deg REAL,                              -- point target, or
    ra_min REAL, ra_max REAL, dec_min REAL, dec_max REAL,   -- area (survey)
    hours REAL NOT NULL, telescope TEXT NOT NULL, band TEXT, freq_mhz REAL,
    sun TEXT, min_elevation REAL, commensal_group TEXT,
    paper TEXT NOT NULL, book TEXT, page INTEGER, quote TEXT NOT NULL,
    extracted_by TEXT, created_at TEXT, position_note TEXT,
    gal_l_min REAL, gal_l_max REAL, gal_b_min REAL, gal_b_max REAL, field TEXT
);
CREATE TABLE IF NOT EXISTS fields (
    field_key TEXT NOT NULL,
    name TEXT NOT NULL,              -- as written, or "<paper> area: <description>" if unnamed
    description TEXT,                -- stated characteristics ("2% of the sky", "|b| < 5 deg")
    ra_deg REAL, dec_deg REAL,       -- centre, if stated
    ra_min REAL, ra_max REAL, dec_min REAL, dec_max REAL,          -- RA/Dec box, if stated
    gal_l_min REAL, gal_l_max REAL, gal_b_min REAL, gal_b_max REAL,  -- Galactic box, if stated
    area_deg2 REAL,
    paper TEXT NOT NULL, book TEXT, page INTEGER, quote TEXT NOT NULL,
    extracted_by TEXT, created_at TEXT,
    PRIMARY KEY (field_key, paper)
);
"""


def name_key(name: str) -> str:
    """Cache key: case- and spacing-insensitive, so "NGC5128" == "NGC 5128"."""
    return re.sub(r"\s+", "", name).upper()


def query_form(raw: str) -> str:
    """SIMBAD-friendly spelling of a name as written ("PSRs B1913+16" →
    "PSR B1913+16"; "COSMOS" → "NAME COSMOS Field")."""
    field = _field(raw)
    if field and field[0]:
        return field[0]
    return re.sub(r"^PSRs\b", "PSR", raw)


def extract_names(text: str) -> list[tuple[str, str]]:
    """(raw name, context snippet) for every source name in *text*."""
    text = text.replace("−", "-").replace("–", "-")
    return [
        (m.group(0), text[max(0, m.start() - CONTEXT_CHARS): m.end() + CONTEXT_CHARS])
        for m in NAME_RE.finditer(text)
    ]


def connect(path: Path = SOURCES_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Databases built before these columns existed.
    for table, cols in _ADDED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, sql_type in cols:
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")
    return conn


_ADDED_COLUMNS = {
    "requests": [("position_note", "TEXT"), ("gal_l_min", "REAL"), ("gal_l_max", "REAL"),
                 ("gal_b_min", "REAL"), ("gal_b_max", "REAL"), ("field", "TEXT")],
    "fields": [("description", "TEXT"), ("ra_min", "REAL"), ("ra_max", "REAL"), ("dec_min", "REAL"),
               ("dec_max", "REAL"), ("gal_l_min", "REAL"), ("gal_l_max", "REAL"),
               ("gal_b_min", "REAL"), ("gal_b_max", "REAL")],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_mentions(conn: sqlite3.Connection, chunks: list[dict]) -> int:
    """Insert every name mention found in *chunks*; returns new mentions."""
    count = lambda: conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]  # noqa: E731
    before = count()
    for chunk in chunks:
        meta = chunk["meta"]
        for raw, context in extract_names(chunk["text"]):
            key = name_key(raw)
            conn.execute(
                "INSERT OR IGNORE INTO names (name_key, raw_name, status) VALUES (?, ?, 'pending')",
                (key, raw),
            )
            conn.execute(
                "INSERT OR IGNORE INTO mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, Path(meta["source"]).stem, meta["source"], meta.get("doc_source"),
                 section_of(meta["source"]), meta.get("chunk_index", 0),
                 meta.get("page_no", 0), context),
            )
    conn.commit()
    return count() - before


def _query_simbad(names: list[str]):
    from rag.sky_lookup import _simbad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _simbad().query_objects(names)


def _lookup(names: list[str], query_fn) -> list:
    """One SIMBAD row (or None) per name, in order. Names SIMBAD doesn't know
    as written are retried once as "NAME <name>" (e.g. SIMBAD only knows
    "NAME SGR 1935+2154")."""
    from rag.sky_lookup import _clean

    def _batch(batch: list[str]) -> list:
        table = query_fn(batch)
        # object_number_id is the 1-based position in the submitted list.
        by_position = {int(r["object_number_id"]): r for r in table}
        rows = [by_position.get(i) for i in range(1, len(batch) + 1)]
        return [r if r is not None and _clean(r["main_id"]) else None for r in rows]

    rows = _batch(names)
    misses = [i for i, r in enumerate(rows) if r is None]
    if misses:
        for i, r in zip(misses, _batch([f"NAME {names[i]}" for i in misses])):
            rows[i] = r
    return rows


def resolve_pending(conn: sqlite3.Connection, query_fn=_query_simbad) -> dict:
    """Look up every 'pending' name in SIMBAD, in batches, caching both hits
    and misses. Objects SIMBAD identifies but gives no position (e.g.
    gravitational-wave events) are stored with null coordinates. *query_fn*
    is injectable for tests."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from rag.sky_lookup import _clean

    pending = conn.execute("SELECT name_key, raw_name FROM names WHERE status = 'pending'").fetchall()
    resolved = unresolved = 0
    for start in range(0, len(pending), SIMBAD_BATCH):
        batch = pending[start: start + SIMBAD_BATCH]
        # Curated fields never go to SIMBAD; everything else is looked up.
        curated = {i: _field(row["raw_name"])[1] for i, row in enumerate(batch)
                   if _field(row["raw_name"]) and _field(row["raw_name"])[1]}
        to_query = [i for i in range(len(batch)) if i not in curated]
        hits = [None] * len(batch)
        for i, hit in zip(to_query, _lookup([query_form(batch[i]["raw_name"]) for i in to_query], query_fn)
                          if to_query else []):
            hits[i] = hit
        for i, c in curated.items():
            hits[i] = {"main_id": f"{batch[i]['raw_name']} field", "ra": c["ra"], "dec": c["dec"],
                       "otype": "field", "rvz_redshift": None, "resolved_by": f"curated: {c['source']}"}
        for row, hit in zip(batch, hits):
            if hit is None:
                conn.execute(
                    "UPDATE names SET status = 'unresolved', checked_at = ? WHERE name_key = ?",
                    (_now(), row["name_key"]),
                )
                unresolved += 1
                continue
            main_id = _clean(hit["main_id"])
            ra, dec = _clean(hit["ra"]), _clean(hit["dec"])
            gal = SkyCoord(ra * u.deg, dec * u.deg).galactic if ra is not None else None
            conn.execute(
                "INSERT OR IGNORE INTO sources (main_id, ra_deg, dec_deg, gal_l_deg, gal_b_deg,"
                " object_type, redshift, resolved_by, resolved_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (main_id, ra, dec, gal.l.deg if gal else None, gal.b.deg if gal else None,
                 _clean(hit["otype"]), _clean(hit["rvz_redshift"]),
                 hit["resolved_by"] if isinstance(hit, dict) and "resolved_by" in hit else "SIMBAD", _now()),
            )
            source_id = conn.execute("SELECT id FROM sources WHERE main_id = ?", (main_id,)).fetchone()[0]
            conn.execute(
                "UPDATE names SET status = 'resolved', source_id = ?, checked_at = ? WHERE name_key = ?",
                (source_id, _now(), row["name_key"]),
            )
            resolved += 1
        conn.commit()
        logger.info(f"SIMBAD: {start + len(batch)}/{len(pending)} names checked")
        if start + SIMBAD_BATCH < len(pending):
            time.sleep(SIMBAD_PAUSE_S)
    return {"resolved": resolved, "unresolved": unresolved}


def build_source_db(path: Path = SOURCES_DB_PATH, retry_unresolved: bool = False) -> dict:
    """Scan all indexed chunks, then resolve names never seen before (plus,
    with *retry_unresolved*, names SIMBAD previously couldn't resolve)."""
    from rag.retrieval import get_all_chunks

    conn = connect(path)
    if retry_unresolved:
        conn.execute("UPDATE names SET status = 'pending' WHERE status = 'unresolved'")
    new_mentions = record_mentions(conn, get_all_chunks())
    looked_up = resolve_pending(conn)
    return {"new_mentions": new_mentions, **looked_up, **stats(conn)}


def stats(conn: sqlite3.Connection) -> dict:
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "sources": one("SELECT COUNT(*) FROM sources"),
        "names_resolved": one("SELECT COUNT(*) FROM names WHERE status = 'resolved'"),
        "names_unresolved": one("SELECT COUNT(*) FROM names WHERE status = 'unresolved'"),
        "mentions": one("SELECT COUNT(*) FROM mentions"),
    }


def query_sources(conn: sqlite3.Connection, object_type: str | None = None,
                  book: str | None = None, section: str | None = None,
                  paper: str | None = None, dec_min: float | None = None,
                  dec_max: float | None = None, min_papers: int = 1,
                  limit: int = 100) -> list[dict]:
    """Resolved sources, most-cited first, with the papers that mention them."""
    where, params = ["n.status = 'resolved'"], []
    for column, value in (("s.object_type", object_type), ("m.book", book),
                          ("m.section", section), ("m.paper", paper)):
        if value is not None:
            where.append(f"{column} = ?")
            params.append(value)
    if dec_min is not None:
        where.append("s.dec_deg >= ?")
        params.append(dec_min)
    if dec_max is not None:
        where.append("s.dec_deg <= ?")
        params.append(dec_max)
    rows = conn.execute(
        f"""SELECT s.*, COUNT(*) AS n_mentions, COUNT(DISTINCT m.paper) AS n_papers,
                   GROUP_CONCAT(DISTINCT m.paper) AS papers,
                   GROUP_CONCAT(DISTINCT n.raw_name) AS names_as_written
            FROM sources s JOIN names n ON n.source_id = s.id
            JOIN mentions m ON m.name_key = n.name_key
            WHERE {' AND '.join(where)}
            GROUP BY s.id HAVING n_papers >= ?
            ORDER BY n_papers DESC, n_mentions DESC LIMIT ?""",
        (*params, min_papers, limit),
    ).fetchall()
    return [
        {**dict(r), "papers": sorted(r["papers"].split(",")),
         "names_as_written": sorted(r["names_as_written"].split(","))}
        for r in rows
    ]


def source_mentions(conn: sqlite3.Connection, name: str, limit: int = 50) -> dict:
    """A source (by any name seen in the papers, or its SIMBAD main_id) and
    every passage that mentions it under any of its names."""
    row = conn.execute(
        "SELECT s.* FROM sources s JOIN names n ON n.source_id = s.id WHERE n.name_key = ?",
        (name_key(name),),
    ).fetchone() or conn.execute(
        # SIMBAD prefixes common names with "NAME " ("NAME Centaurus A").
        "SELECT * FROM sources WHERE REPLACE(UPPER(main_id), ' ', '') IN (?, ?)",
        (name_key(name), "NAME" + name_key(name)),
    ).fetchone()
    if row is None:
        status = conn.execute("SELECT status FROM names WHERE name_key = ?", (name_key(name),)).fetchone()
        return {"found": False, "name": name, "status": status[0] if status else "never seen in papers"}
    mentions = conn.execute(
        """SELECT n.raw_name, m.paper, m.book, m.section, m.page, m.context
           FROM names n JOIN mentions m ON m.name_key = n.name_key
           WHERE n.source_id = ? ORDER BY m.book, m.paper, m.page LIMIT ?""",
        (row["id"], limit),
    ).fetchall()
    return {"found": True, "source": dict(row), "mentions": [dict(m) for m in mentions]}


def _squash(text: str) -> str:
    """Lowercase with all whitespace removed: PDF extraction spaces text
    unpredictably ("10 . 3 h"), so quotes are compared in this form."""
    return re.sub(r"\s+", "", text).lower()


def find_quote(quote: str, paper: str, book: str | None = None) -> dict | None:
    """The chunk of *paper* containing *quote* verbatim (whitespace- and
    case-insensitive), or None."""
    from rag.corpus_tools import build_filter
    from rag.retrieval import get_all_chunks

    sources = set(build_filter(book=book, paper=paper)["source"])
    wanted = _squash(quote)
    for c in get_all_chunks():
        if c["meta"]["source"] in sources and wanted in _squash(c["text"]):
            return c["meta"]
    return None


def _check_quote(quote: str, paper: str, book: str | None) -> dict:
    meta = find_quote(quote, paper, book)
    if meta is None:
        raise ValueError(f"Quote not found verbatim in paper {paper!r}; entries must quote the paper")
    return meta


def _field_key(name: str) -> str:
    """Field names vary in hyphenation too: "ELAIS-N1" == "ELAIS N1" == "ELAISN1"."""
    return name_key(name).replace("-", "")


def _geometry(row) -> dict | None:
    """Placeable geometry from a fields/requests row: a centre, an RA/Dec box
    or a Galactic box (None if only an area or description is known)."""
    if row["ra_min"] is not None:
        return {"ra_range": [row["ra_min"], row["ra_max"]], "dec_range": [row["dec_min"], row["dec_max"]]}
    if row["gal_b_min"] is not None:
        return {"gal_l_range": [row["gal_l_min"], row["gal_l_max"]],
                "gal_b_range": [row["gal_b_min"], row["gal_b_max"]]}
    if row["ra_deg"] is not None:
        return {"ra": row["ra_deg"], "dec": row["dec_deg"]}
    return None


def _live_lookup(name: str) -> dict:
    """SIMBAD (then NED) lookup; module-level so tests can stub the network."""
    from rag.sky_lookup import resolve_object

    return resolve_object(name)


def _resolve_live(conn: sqlite3.Connection, name: str) -> dict | None:
    """Look up a target the papers' regex scan never saw (e.g. "OMC2", or a
    name whose minus sign the PDF text lost) and cache the answer in the
    sources/names tables, including "not found", so each name is asked once.
    Network failures are not cached."""
    key = name_key(name)
    row = conn.execute("SELECT status FROM names WHERE name_key = ?", (key,)).fetchone()
    if row and row["status"] == "unresolved":
        return None
    try:
        hit = _live_lookup(query_form(name))
    except Exception:  # network/service error: leave uncached, try again next time
        return None
    if not hit.get("found") or hit.get("ra_deg") is None:
        conn.execute("INSERT OR REPLACE INTO names (name_key, raw_name, status, checked_at)"
                     " VALUES (?, ?, 'unresolved', ?)", (key, name, _now()))
        conn.commit()
        return None
    conn.execute(
        "INSERT OR IGNORE INTO sources (main_id, ra_deg, dec_deg, gal_l_deg, gal_b_deg,"
        " object_type, redshift, resolved_by, resolved_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (hit["name"], hit["ra_deg"], hit["dec_deg"], hit.get("gal_l_deg"), hit.get("gal_b_deg"),
         hit.get("object_type"), hit.get("redshift"), f"{hit['service']} (live)", _now()),
    )
    source_id = conn.execute("SELECT id FROM sources WHERE main_id = ?", (hit["name"],)).fetchone()[0]
    conn.execute("INSERT OR REPLACE INTO names (name_key, raw_name, source_id, status, checked_at)"
                 " VALUES (?, ?, ?, 'resolved', ?)", (key, name, source_id, _now()))
    conn.commit()
    return {"ra": hit["ra_deg"], "dec": hit["dec_deg"], "from": f"{hit['service']} (live): {hit['name']}",
            "source_id": source_id}


def field_position(conn: sqlite3.Connection, name: str, live: bool = False) -> dict | None:
    """Best known geometry for a survey field or target: one stated in a
    paper (earliest recorded), else a position from the sources table
    (SIMBAD or curated), else, with *live*, a SIMBAD/NED lookup (cached).
    Adds "from" (where it came from)."""
    for row in conn.execute("SELECT * FROM fields WHERE field_key = ? ORDER BY created_at",
                            (_field_key(name),)):
        geom = _geometry(row)
        if geom:
            return {**geom, "from": f"paper: {row['paper']} p.{row['page']}"}
    row = conn.execute(
        "SELECT s.id, s.ra_deg, s.dec_deg, s.resolved_by FROM sources s JOIN names n ON n.source_id = s.id"
        " WHERE n.name_key = ? AND s.ra_deg IS NOT NULL", (name_key(name),)).fetchone()
    if row:
        return {"ra": row["ra_deg"], "dec": row["dec_deg"], "from": row["resolved_by"], "source_id": row["id"]}
    if live and " area: " not in name:  # unnamed-area labels are never object names
        return _resolve_live(conn, name)
    return None


def _pair(value, label: str) -> tuple:
    if value is None:
        return (None, None)
    if len(value) != 2:
        raise ValueError(f"{label} must be [min, max]")
    return tuple(float(v) for v in value)


def add_survey_field(conn: sqlite3.Connection, paper: str, quote: str, name: str | None = None,
                     description: str | None = None, book: str | None = None,
                     ra: float | None = None, dec: float | None = None,
                     ra_range: list | None = None, dec_range: list | None = None,
                     gal_l_range: list | None = None, gal_b_range: list | None = None,
                     area_deg2: float | None = None, extracted_by: str | None = None) -> dict:
    """Record a survey field or area as a paper states it: a named field
    (COSMOS, ELAIS-N1) and/or an area defined by its characteristics (sky
    coverage, Dec or Galactic-latitude limits, e.g. "20,000 deg2 of the
    southern sky" or "|b| < 5 deg"). Only stated values are recorded. Refused
    unless *quote* appears verbatim in the paper. Returns the label to use
    in requests and the best known geometry."""
    if not name and not description:
        raise ValueError("give the field's name, or a description of the area for an unnamed one")
    if (ra is None) != (dec is None) or (ra_range is None) != (dec_range is None):
        raise ValueError("give ra with dec, and ra_range with dec_range")
    if gal_b_range is not None and gal_l_range is None:
        gal_l_range = [0, 360]
    meta = _check_quote(quote, paper, book)
    stem = Path(meta["source"]).stem
    label = name or f"{stem} area: {description}"
    conn.execute(
        """INSERT OR REPLACE INTO fields (field_key, name, description, ra_deg, dec_deg,
               ra_min, ra_max, dec_min, dec_max, gal_l_min, gal_l_max, gal_b_min, gal_b_max,
               area_deg2, paper, book, page, quote, extracted_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (_field_key(label), label, description, ra, dec,
         *_pair(ra_range, "ra_range"), *_pair(dec_range, "dec_range"),
         *_pair(gal_l_range, "gal_l_range"), *_pair(gal_b_range, "gal_b_range"),
         area_deg2, stem, meta.get("doc_source"), meta.get("page_no", 0), quote, extracted_by, _now()),
    )
    conn.commit()
    return {"field": label, "paper": stem, "page": meta.get("page_no", 0),
            "geometry": field_position(conn, label, live=True)}


def list_fields(conn: sqlite3.Connection) -> list[dict]:
    """Recorded survey fields/areas: papers naming each, stated
    characteristics, and best known geometry (None = unplaced)."""
    out = []
    for r in conn.execute(
        """SELECT field_key, MIN(name) AS name, COUNT(*) AS n_papers, GROUP_CONCAT(paper) AS papers,
                  MAX(area_deg2) AS area_deg2, GROUP_CONCAT(description, ' | ') AS description
           FROM fields GROUP BY field_key ORDER BY n_papers DESC, name"""):
        out.append({"name": r["name"], "n_papers": r["n_papers"], "papers": sorted(r["papers"].split(",")),
                    "area_deg2": r["area_deg2"], "description": r["description"],
                    "geometry": field_position(conn, r["name"])})
    return out


def add_request(conn: sqlite3.Connection, name: str, hours: float, telescope: str,
                paper: str, quote: str, book: str | None = None,
                ra: float | None = None, dec: float | None = None,
                ra_range: list | None = None, dec_range: list | None = None,
                band: str | None = None, freq_mhz: float | None = None,
                sun: str = "any", min_elevation: float | None = None,
                commensal_group: str | None = None, extracted_by: str | None = None,
                position_note: str | None = None, field: str | None = None,
                gal_l_range: list | None = None, gal_b_range: list | None = None) -> dict:
    """Record an observing request stated in a paper.

    Refused unless *quote* appears verbatim in *paper*: requests must come
    from the text, not from an agent's inference. Position, in order: ra/dec,
    an RA/Dec box or a Galactic box as given; else the geometry of *field*
    (or *name*) as a recorded survey field/area or a source (e.g. "Cen A",
    "COSMOS"); else, only
    if *position_note* explains why (e.g. "unnamed deep field"), the request
    is stored unplaced and lst_pressure reports its hours separately.
    """
    from rag.scheduling import SUN_MODES

    if hours <= 0:
        raise ValueError("hours must be positive")
    telescope = telescope.lower().replace("ska-", "")
    if telescope not in ("low", "mid"):
        raise ValueError("telescope must be 'low' or 'mid'")
    if sun not in SUN_MODES:
        raise ValueError(f"sun must be one of {SUN_MODES}")
    if ra_range is not None and dec_range is None:
        raise ValueError("ra_range needs dec_range")
    meta = _check_quote(quote, paper, book)

    if gal_b_range is not None and gal_l_range is None:
        gal_l_range = [0, 360]
    source_id = None
    if ra is None and ra_range is None and gal_b_range is None:
        pos = field_position(conn, field or name, live=True)
        if pos:
            source_id = pos.get("source_id")
            ra, dec = pos.get("ra"), pos.get("dec")
            ra_range, dec_range = pos.get("ra_range"), pos.get("dec_range")
            gal_l_range, gal_b_range = pos.get("gal_l_range"), pos.get("gal_b_range")
        elif not position_note:
            raise ValueError(f"No position for {name!r}: give ra/dec or ra_range/dec_range, record the "
                             "field with add_survey_field, or pass position_note to store it unplaced")

    cur = conn.execute(
        """INSERT INTO requests (name, source_id, ra_deg, dec_deg, ra_min, ra_max, dec_min, dec_max,
               hours, telescope, band, freq_mhz, sun, min_elevation, commensal_group,
               paper, book, page, quote, extracted_by, created_at, position_note,
               gal_l_min, gal_l_max, gal_b_min, gal_b_max, field)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (name, source_id, ra, dec, *_pair(ra_range, "ra_range"), *_pair(dec_range, "dec_range"),
         hours, telescope, band, freq_mhz, sun, min_elevation, commensal_group,
         Path(meta["source"]).stem, meta.get("doc_source"), meta.get("page_no", 0),
         quote, extracted_by, _now(), position_note,
         *_pair(gal_l_range, "gal_l_range"), *_pair(gal_b_range, "gal_b_range"), field),
    )
    conn.commit()
    return {"id": cur.lastrowid, "paper": Path(meta["source"]).stem, "page": meta.get("page_no", 0),
            "placed": ra is not None or ra_range is not None or gal_b_range is not None}


def list_requests(conn: sqlite3.Connection, telescope: str | None = None,
                  extracted_by: str | None = None) -> list[dict]:
    """Stored requests, shaped for rag.scheduling.lst_pressure."""
    where, params = [], []
    if telescope:
        where.append("telescope = ?")
        params.append(telescope.lower().replace("ska-", ""))
    if extracted_by:
        where.append("extracted_by = ?")
        params.append(extracted_by)
    rows = conn.execute(
        "SELECT * FROM requests" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id",
        params,
    ).fetchall()
    out = []
    for r in rows:
        req = {k: r[k] for k in ("id", "name", "hours", "telescope", "band", "freq_mhz", "sun",
                                 "commensal_group", "paper", "book", "page", "quote", "field") if r[k] is not None}
        if r["min_elevation"] is not None:
            req["min_elevation"] = r["min_elevation"]
        geom = _geometry(r)
        if geom:
            req.update(geom)
        else:
            req["position_note"] = r["position_note"]
        out.append(req)
    return out


def unresolved_names(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    """Names SIMBAD couldn't resolve, most-mentioned first (candidates for a
    curated field list or a regex fix)."""
    return [dict(r) for r in conn.execute(
        """SELECT n.raw_name, COUNT(*) AS n_mentions, COUNT(DISTINCT m.paper) AS n_papers
           FROM names n JOIN mentions m ON m.name_key = n.name_key
           WHERE n.status = 'unresolved' GROUP BY n.name_key
           ORDER BY n_mentions DESC LIMIT ?""", (limit,))]


if __name__ == "__main__":
    import sys

    print(build_source_db(retry_unresolved="--retry-unresolved" in sys.argv))
