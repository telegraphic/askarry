"""Build the data layer of the AASKAII Atlas v2: one JSON per page under
aaskaii-atlas-v2/data/, computed from the corpus tools and databases so every
figure on the site can be regenerated. Prose is written separately.

    python scripts/build_atlas_data.py [--out aaskaii-atlas-v2/data]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import citation_db, corpus_tools, scheduling, source_db  # noqa: E402
from rag.bibliography import list_toc_entries  # noqa: E402
from rag.config import SCHEDULING  # noqa: E402

# Reviewed extraction run that feeds the time-demand page.
REQUESTS_RUN = "full-sonnet5-medium"

# Search terms per taxonomy entry, "|"-separated (corpus_tools._term_regexes
# syntax: all-caps alternatives match case-sensitively, the rest ignore case).
# Keys must match mcp_server.ATLAS_TAXONOMIES exactly.
MODE_TERMS = {
    "Deep HI intensity mapping": "intensity mapping",
    "Deep/wide HI galaxy survey": "HI galaxy survey|Hi galaxy survey|H i galaxy survey|HI survey|Hi survey|H i survey"
                                  "|untargeted HI|untargeted Hi|blind HI|blind Hi",
    "Wide-field continuum survey": "continuum survey|wide-area survey|all-sky survey",
    "Ultra-deep continuum imaging": "ultra-deep|ultra deep|deep field",
    "EoR / Cosmic Dawn 21-cm": "EoR|Epoch of Reionization|Epoch of Reionisation|Cosmic Dawn",
    "Spectral line / maser survey": "maser|radio recombination line|RRL|spectral line survey",
    "Polarimetry / RM grid": "RM grid|rotation measure grid|polarization survey|polarisation survey|polarimetric survey",
    "VLBI / astrometry": "VLBI|very long baseline interferometry|astrometry",
    "Pulsar timing array": "PTA|pulsar timing array",
    "Pulsar search survey": "pulsar search|pulsar survey|PSS",
    "Transient / commensal / ToO": "target of opportunity|target-of-opportunity|commensal|FRB|transient search",
    "Solar/heliospheric imaging": "heliosphere|heliospheric|solar radio|solar corona|solar imaging|coronal mass ejection",
    "Weak lensing": "weak lensing|cosmic shear",
    "Zoom-mode spectroscopy": "zoom mode|zoom-mode|zoom window",
    "SETI / technosignature": "SETI|technosignature|technosignatures",
}
TECHNIQUE_TERMS = {
    "Faraday rotation / RM": "Faraday rotation|rotation measure|Faraday depth|RM synthesis",
    "VLBI": "VLBI",
    "HI intensity mapping": "intensity mapping",
    "Machine learning / AI": "machine learning|deep learning|neural network|artificial intelligence",
    "Polarimetry": "polarimetry|polarimetric",
    "Pulsar timing": "pulsar timing",
    "Foreground removal": "foreground removal|foreground subtraction|foreground mitigation|foreground avoidance",
    "Ionospheric calibration": "ionospheric calibration|ionospheric",
    "Deconvolution / imaging algorithms": "deconvolution|CLEAN|imaging algorithm",
    "Spectral line / maser": "maser|spectral line",
    "Commensal / multi-messenger": "commensal|multi-messenger|multimessenger",
    "Source finding / classification": "source finding|source-finding|source finder|source classification",
    "VLBI astrometry": "VLBI astrometry|astrometric",
    "RFI mitigation": "RFI mitigation|RFI excision|RFI flagging",
    "Zeeman effect": "Zeeman",
    "Citizen science": "citizen science|Radio Galaxy Zoo",
}
# The v1 risk register's 14 named keywords, plus primary-beam and bandpass
# stability to make 16 (v1 said 16 but named 14).
RISK_TERMS = {
    "RFI": "RFI|radio frequency interference",
    "Ionospheric calibration": "ionospheric calibration|ionospheric phase|ionospheric distortion",
    "Direction-dependent calibration": "direction-dependent|direction dependent",
    "Dynamic-range limitation": "dynamic range",
    "Confusion limit": "confusion limit|confusion noise|confusion-limited",
    "Polarization leakage": "polarization leakage|polarisation leakage|instrumental polarization|instrumental polarisation",
    "Computational cost": "computational cost|computationally expensive|compute cost|HPC",
    "Data rate / volume": "data rate|data volume",
    "Survey-speed limitation": "survey speed",
    "Beam sidelobes": "sidelobe|sidelobes|side-lobe",
    "Timing / synchronisation": "clock synchronization|clock synchronisation|time synchronization|timing precision",
    "Commensal scheduling conflict": "scheduling conflict|scheduling constraints",
    "Telescope uptime": "uptime|downtime",
    "ML sim-to-real gap": "domain shift|domain adaptation|sim-to-real",
    "Primary-beam model": "primary beam model|beam model",
    "Bandpass / gain stability": "bandpass calibration|bandpass stability|gain stability",
}


def _meta(**args) -> dict:
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return {"git_sha": sha, "date": date.today().isoformat(), "args": args}


# A paper counts for a term only with >= 2 mentions, as in facility_network:
# drops passing mentions and affiliations (e.g. "SETI Institute").
MIN_HITS = 2


_CHAPTERS: dict[str, str] = {}


def book_chapters() -> dict[str, str]:
    """{paper id: section} for the book's chapters, i.e. indexed files with a
    bibliography entry. The index also holds re-downloaded copies (Curiel01b,
    Kurahara01_3, ...) and non-chapters (a GEP document, a contents page);
    counting only chapters keeps every figure a count of distinct chapters."""
    if not _CHAPTERS:
        for s, entries in list_toc_entries(corpus_tools.BOOKS["aaskaii"]).items():
            _CHAPTERS.update({Path(e["path"]).stem: s for e in entries})
    return _CHAPTERS


def _only_chapters(ids) -> list[str]:
    return sorted({i for i in ids if i in book_chapters()})


def _by_section(ids: list[str]) -> dict[str, list[str]]:
    out = defaultdict(list)
    for i in ids:
        out[book_chapters()[i]].append(i)
    return dict(sorted(out.items(), key=lambda kv: -len(kv[1])))


def _papers_per_section() -> dict[str, int]:
    return {s: len(v) for s, v in _by_section(list(book_chapters())).items()}


def _restrict_term(t: dict) -> dict:
    """A keyword_coverage term counted over chapters only."""
    ids = _only_chapters(p for sec in t["sections"].values() for p in sec["papers"])
    secs = _by_section(ids)
    return {**t, "n_papers": len(ids), "n_sections": len(secs),
            "sections": {s: {"n_papers": len(p), "papers": p} for s, p in secs.items()}}


def _coverage(labelled_terms: dict[str, str], min_hits: int = MIN_HITS) -> dict:
    """keyword_coverage over chapters, each result tagged by its taxonomy label."""
    cov = corpus_tools.keyword_coverage(list(labelled_terms.values()), book="aaskaii", min_hits_per_paper=min_hits)
    by_term = {t["term"]: _restrict_term(t) for t in cov["terms"]}
    return {"papers_per_section": _papers_per_section(),
            "items": [{"label": label, **by_term[term]} for label, term in labelled_terms.items()]}


def facilities() -> dict:
    """facility_network over chapters only: counts, sections and edges are
    recomputed from each facility's chapter list."""
    net = corpus_tools.facility_network(book="aaskaii")
    fac = []
    for f in net["facilities"]:
        ids = _only_chapters(f["papers"])
        if ids:
            fac.append({**f, "papers": ids, "n_papers": len(ids),
                        "sections": {s: len(p) for s, p in _by_section(ids).items()},
                        "n_sections": len(_by_section(ids))})
    sets = {f["facility"]: set(f["papers"]) for f in fac}
    names = [f["facility"] for f in fac]
    edges = sorted(({"a": a, "b": b, "shared_papers": len(sets[a] & sets[b]), "papers": sorted(sets[a] & sets[b])}
                    for i, a in enumerate(names) for b in names[i + 1:] if len(sets[a] & sets[b]) >= 2),
                   key=lambda e: -e["shared_papers"])
    return {**net, "facilities": fac, "edges": edges,
            "not_named": sorted(set(net["not_named"]) | {f["facility"] for f in net["facilities"]} - set(names))}


def chapters() -> dict:
    toc = list_toc_entries(corpus_tools.BOOKS["aaskaii"])
    missing = {m["paper"] for m in corpus_tools.missing_papers("aaskaii")["not_indexed"]}
    seen = set()  # one row per paper ID: a stray copy (e.g. a second Bianchi01.pdf) must not list twice

    def rows(entries):
        for e in entries:
            stem = Path(e["path"]).stem
            if stem not in seen:
                seen.add(stem)
                yield {"paper": stem, "title": e["title"], "authors": e["authors"],
                       "pdf_url": corpus_tools.AASKAII_PDF_URL.format(stem=stem), "indexed": stem not in missing}
    sections = {s: list(rows(entries)) for s, entries in toc.items()}
    # Chapters in the bibliography with no local PDF: listed (and linked online) but not searchable.
    for m in corpus_tools.missing_papers("aaskaii")["not_indexed"]:
        info = corpus_tools.paper_info(m["paper"])
        sections.setdefault(m["section"], []).append({
            "paper": m["paper"], "title": m["title"], "authors": info[0]["authors"] if info else [],
            # Never downloaded because skao.int serves no PDF for them (checked 2026-09-25: 10 are
            # absent from the AASKAII page, Kavanagh01 is listed but 404s), so link the page instead.
            "pdf_url": None, "indexed": False})
    return {"sections": [{"section": s, "papers": sorted(p, key=lambda r: r["title"].lower())} for s, p in sections.items()]}


# Topic terms for the 2015 book's science-working-group sections, used to see
# where each 2015 topic now sits among the 2026 sections. A 2026 chapter counts
# only if it treats the topic substantively (>= DIFF_MIN_HITS mentions).
TOPIC_TERMS_2015 = {
    "Cosmology": "cosmology|cosmological",
    "Epoch of Reionisation": "EoR|Epoch of Reionization|Epoch of Reionisation|Cosmic Dawn",
    "Fundamental Physics with Pulsars": "pulsar timing|PTA|binary pulsar|millisecond pulsar",
    "Magnetism": "magnetic field|magnetic fields|magnetism|rotation measure",
    "Synergies and Other Science": "multi-wavelength|multiwavelength|synergy|synergies",
    "The Continuum Universe": "radio continuum|continuum survey|star formation rate",
    "The Cradle of Life": "protoplanetary|exoplanet|exoplanets|astrochemistry|prebiotic",
    "The Hydrogen Universe": "neutral hydrogen|HI|21-cm|21 cm",
    "The Transient Universe": "transient|transients|FRB|fast radio burst",
}
DIFF_MIN_HITS = 5


def books_diff() -> dict:
    return {**{book: [{"section": s, "n_papers": len(e), "titles": [x["title"] for x in e]}
                      for s, e in list_toc_entries(corpus_tools.BOOKS[book]).items()]
               for book in ("aaska2015", "aaskaii")},
            "topic_spread": {"min_hits_per_paper": DIFF_MIN_HITS, **_coverage(TOPIC_TERMS_2015, DIFF_MIN_HITS)}}


def tools() -> dict:
    """MCP tools and resources as declared in mcp_server.py (name, first
    docstring line), read with ast so the list cannot drift from the code."""
    import ast
    tree = ast.parse((ROOT / "mcp_server.py").read_text())
    out = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for d in node.decorator_list:
            f = d.func if isinstance(d, ast.Call) else d
            if isinstance(f, ast.Attribute) and f.attr in ("tool", "resource"):
                doc = (ast.get_docstring(node) or "").strip().split("\n\n")[0].replace("\n", " ")
                out.append({"name": node.name, "kind": f.attr, "summary": doc,
                            "params": [a.arg for a in node.args.args]})
    return {"tools": out}


def citations() -> dict:
    conn = citation_db.connect()
    sections = [s for s in list_toc_entries(corpus_tools.BOOKS["aaskaii"])]
    def ranked(rows, limit):
        rows = [{**r, "papers": _only_chapters(r["papers"])} for r in rows]
        rows = [{**r, "n_papers": len(r["papers"])} for r in rows if r["papers"]]
        return sorted(rows, key=lambda r: -r["n_papers"])[:limit]
    # Over-fetch, then re-rank once duplicate copies stop counting as citers.
    return {"overall": ranked(citation_db.top_cited(conn, book="aaskaii", limit=80), 50),
            "by_section": {s: ranked(citation_db.top_cited(conn, book="aaskaii", section=s, limit=20), 10) for s in sections}}


def sources() -> dict:
    conn = source_db.connect()
    objs = [{**o, "papers": _only_chapters(o["papers"])} for o in source_db.query_sources(conn, book="aaskaii", limit=1000)]
    return {"sources": [{**o, "n_papers": len(o["papers"])} for o in objs if o["papers"]],
            "fields": [{**f, "papers": _only_chapters(f["papers"])} for f in source_db.list_fields(conn)]}


def time_demand(conn, extracted_by: str = REQUESTS_RUN) -> dict:
    """Reviewed requests per telescope, their LST pressure, and the survey
    tiers (accepted rows with a commensal group, plus their duplicates)."""
    every = source_db.list_requests(conn, None, extracted_by, include_rejected=True)
    status = defaultdict(lambda: {"n": 0, "hours": 0.0})
    for r in every:
        s = status[r.get("review_status") or "unreviewed"]
        s["n"] += 1
        s["hours"] += r["hours"]
    accepted = [r for r in every if r.get("review_status") == "accepted"]
    by_id = {r["id"]: r for r in every}
    tiers = []
    for r in accepted:
        if not r.get("commensal_group"):
            continue
        dups = [d for d in every if d.get("duplicate_of") == r["id"]]
        tiers.append({"group": r["commensal_group"], "telescope": r["telescope"], "defining_paper": r["paper"],
                      "hours": r["hours"], "placed": "position_note" not in r,
                      "field": r.get("reviewed_field") or r.get("field"),
                      "restated_by": [{"paper": d["paper"], "hours": d.get("extracted_hours", d["hours"])}
                                      for d in dups]})
    out = {"run": extracted_by, "status": dict(status), "tiers": tiers, "telescopes": {}}
    for tel in ("low", "mid"):
        reqs = [r for r in accepted if r["telescope"] == tel]
        placed = [r for r in reqs if "position_note" not in r]
        out["telescopes"][tel] = {
            "accepted_hours": sum(r["hours"] for r in reqs),
            "placed_hours": sum(r["hours"] for r in placed),
            "unplaced_hours": sum(r["hours"] for r in reqs if "position_note" in r),
            "requests": [{k: r.get(k) for k in ("id", "name", "paper", "hours", "commensal_group", "review_note")}
                         | {"placed": "position_note" not in r} for r in reqs],
            "lst_pressure": scheduling.lst_pressure(reqs, tel),
        }
    out["duplicates_resolved"] = sum(1 for r in every if r.get("review_status") == "duplicate"
                                     and r.get("duplicate_of") in by_id)
    return out


def build(out_dir: Path) -> dict[str, Path]:
    from mcp_server import ATLAS_TAXONOMIES  # the taxonomy is defined once, on the server
    assert list(MODE_TERMS) == ATLAS_TAXONOMIES["observing_modes"], "MODE_TERMS out of sync with the taxonomy"
    assert list(TECHNIQUE_TERMS) == ATLAS_TAXONOMIES["techniques"], "TECHNIQUE_TERMS out of sync with the taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = {
        "chapters": (chapters, {}),
        "tools": (tools, {"source": "mcp_server.py"}),
        "books_diff": (books_diff, {"topic_terms_2015": TOPIC_TERMS_2015, "min_hits_per_paper": DIFF_MIN_HITS}),
        "acronyms": (lambda: {"acronyms": corpus_tools.list_acronyms("aaskaii", limit=10**6)}, {"book": "aaskaii"}),
        "modes": (lambda: _coverage(MODE_TERMS), {"terms": MODE_TERMS, "min_hits_per_paper": MIN_HITS}),
        "methods": (lambda: _coverage(TECHNIQUE_TERMS), {"terms": TECHNIQUE_TERMS, "min_hits_per_paper": MIN_HITS}),
        "risks": (lambda: _coverage(RISK_TERMS), {"terms": RISK_TERMS, "min_hits_per_paper": MIN_HITS}),
        "facilities": (facilities, {"book": "aaskaii", "min_hits_per_paper": 2, "chapters_only": True}),
        "citations": (citations, {"book": "aaskaii"}),
        "sources": (sources, {"book": "aaskaii"}),
        "time_demand": (lambda: time_demand(source_db.connect()),
                        {"extracted_by": REQUESTS_RUN, "reviewed_only": True, "scheduling": SCHEDULING}),
    }
    written, results = {}, {}
    for name, (fn, args) in pages.items():
        path = out_dir / f"{name}.json"
        results[name] = fn()
        text = json.dumps({"generated": _meta(**args), **results[name]}, indent=1, default=str)
        path.write_text(text)
        # Pages load data with <script src>, since fetch() of a local file fails under file://.
        (out_dir / f"{name}.js").write_text(f"(window.ATLAS = window.ATLAS || {{}})[{name!r}] = {text};\n")
        written[name] = path
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} kB)")
    _workflow_args(out_dir, results)
    return written


def _workflow_args(out_dir: Path, results: dict) -> None:
    """Input for scripts/atlas_prose_workflow.js (passed as args.argsPath):
    per section, its indexed chapters and how many mention each mode/technique."""
    def slice_(cov, sec):
        return {it["label"]: it["sections"].get(sec, {}).get("n_papers", 0) for it in cov["items"]}
    out = [{"letter": "ABCDEFG"[i], "section": s["section"],
            "papers": [{"paper": p["paper"], "title": p["title"]} for p in s["papers"] if p["indexed"]],
            "mode_papers": slice_(results["modes"], s["section"]), "method_papers": slice_(results["methods"], s["section"])}
           for i, s in enumerate(x for x in results["chapters"]["sections"] if x["section"] in book_sections())]
    (out_dir / "agents").mkdir(exist_ok=True)
    (out_dir / "agents" / "workflow_args.json").write_text(json.dumps(out, indent=1))


def book_sections() -> list[str]:
    return list(list_toc_entries(corpus_tools.BOOKS["aaskaii"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "aaskaii-atlas-v2" / "data")
    build(ap.parse_args().out)
