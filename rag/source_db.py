"""
Database of astronomical sources named in the indexed papers.

    python -m rag.source_db                      # build/update sources.db (incremental)
    python -m rag.source_db --retry-unresolved   # also re-query past SIMBAD misses

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
"""


def name_key(name: str) -> str:
    """Cache key: case- and spacing-insensitive, so "NGC5128" == "NGC 5128"."""
    return re.sub(r"\s+", "", name).upper()


def query_form(raw: str) -> str:
    """SIMBAD-friendly spelling of a name as written ("PSRs B1913+16" → "PSR B1913+16")."""
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
    return conn


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
        hits = _lookup([query_form(row["raw_name"]) for row in batch], query_fn)
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
                 _clean(hit["otype"]), _clean(hit["rvz_redshift"]), "SIMBAD", _now()),
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
