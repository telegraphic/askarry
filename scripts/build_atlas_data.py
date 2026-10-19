"""Build the data layer of the AASKAII Atlas v2: one JSON per page under
aaskaii-atlas-v2/data/, computed from the corpus tools and databases so every
figure on the site can be regenerated. Prose is written separately.

    python scripts/build_atlas_data.py [--out aaskaii-atlas-v2/data]
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
            "by_section": {s: ranked(citation_db.top_cited(conn, book="aaskaii", section=s, limit=20), 10) for s in sections},
            "year_histogram": _year_histogram(conn)}


def _year_histogram(conn) -> dict:
    """Distinct cited works per publication year, over every chapter's reference list."""
    chapters = set(book_chapters())
    seen = {(r["ref_key"], r["year"]) for r in conn.execute(
        "SELECT DISTINCT r.ref_key, r.year, c.paper FROM refs r JOIN cites c ON c.ref_key = r.ref_key WHERE c.book = ?",
        ("aaskaii",)) if r["paper"] in chapters}
    years = defaultdict(int)
    for _, y in seen:
        years[y if y else "unknown"] += 1
    return {"n_works": len(seen), "by_year": {str(k): v for k, v in sorted(years.items(), key=lambda kv: (kv[0] == "unknown", str(kv[0])))}}


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
            "lst_by_group": _lst_by_group(placed, tel),
        }
    out["duplicates_resolved"] = sum(1 for r in every if r.get("review_status") == "duplicate"
                                     and r.get("duplicate_of") in by_id)
    return out


def _lst_by_group(placed: list[dict], tel: str) -> list[dict]:
    """Demand per LST bin for each commensal group (or lone request), so the
    page can show which surveys make up the pressure. Each group is run through
    lst_pressure on its own, so the groups sum to the telescope's total."""
    groups = defaultdict(list)
    for r in placed:
        groups[r.get("commensal_group") or r["name"]].append(r)
    out = []
    for name, rs in groups.items():
        p = scheduling.lst_pressure(rs, tel)
        if p["total_demand_h"]:
            out.append({"group": name, "papers": sorted({r["paper"] for r in rs}), "hours": p["total_demand_h"],
                        "demand_h": [b["demand_h"] for b in p["lst_bins"]]})
    return sorted(out, key=lambda g: -g["hours"])


# --- Modes & data products --------------------------------------------------
# The four broad categories in which the SKAO configures observations, and the
# fine-grained atlas modes that belong to each (None = outside the four).
MODE_CATEGORY = {
    "Wide-field continuum survey": "Continuum", "Ultra-deep continuum imaging": "Continuum",
    "Polarimetry / RM grid": "Continuum", "EoR / Cosmic Dawn 21-cm": "Continuum",
    "Weak lensing": "Continuum", "Solar/heliospheric imaging": "Continuum",
    "Deep/wide HI galaxy survey": "Zoom", "Deep HI intensity mapping": "Zoom",
    "Spectral line / maser survey": "Zoom", "Zoom-mode spectroscopy": "Zoom",
    "Pulsar search survey": "PSS", "Pulsar timing array": "PST",
    "VLBI / astrometry": None, "Transient / commensal / ToO": None, "SETI / technosignature": None,
}
CATEGORIES = ["Continuum", "Zoom", "PSS", "PST"]
OUTSIDE = "Outside the four"

# Request classification: first matching rule wins, except that Zoom also
# keeps Continuum when the request says "continuum".
_CLASS_RULES = [
    ("PST", re.compile(r"\bPSR J|\btiming\b|\bPTA\b", re.I)),
    ("PSS", re.compile(r"pulsar search|\bFRB|single.pulse", re.I)),
    (OUTSIDE, re.compile(r"\bVLBI\b")),
    ("Zoom", re.compile(r"\bH\s?I\b|\bHi\b|21.?cm|maser|\bRRL|recombination|absorption|\bline\b|astrochem|"
                        r"cyanopolyyne|iCOM|Zeeman|redshift.drift|intensity mapping|\bOH\b", re.I)),
]
# Corrections after reading every assignment once (request id -> categories).
REQUEST_CATEGORY_OVERRIDES: dict[int, list[str]] = {
    56: ["Zoom"],      # Galactic Centre: OH masers in 0.6 km/s channels
    65: ["PST"],       # RMs of known pulsars come from folded, polarised profiles
    85: [OUTSIDE],     # Cen A: an SKA-Mid + Parkes/HartRAO VLBI observation
    115: ["Continuum"],  # EoR power spectrum: full-band visibilities, as the EoR mode maps
    124: ["Zoom"],     # ALP decay: a line search in 10^4 spectral channels
    128: ["Zoom"],     # magnetar ALP conversion "with high spectral resolution"
}


def classify_request(r: dict) -> list[str]:
    """Categories of one reviewed request, from its name, quote and review note."""
    if r.get("id") in REQUEST_CATEGORY_OVERRIDES:
        return REQUEST_CATEGORY_OVERRIDES[r["id"]]
    text = " ".join(str(r.get(k) or "") for k in ("name", "quote", "review_note"))
    for cat, rx in _CLASS_RULES:
        if rx.search(r.get("name") or "") or (cat != OUTSIDE and rx.search(text)):
            if cat == "Zoom" and re.search(r"continuum", text, re.I):
                return ["Continuum", "Zoom"]
            return [cat]
    return ["Continuum"]


C_LIGHT = 299_792_458.0
OMEGA_EARTH = 7.2921e-5  # rad/s
MID_BANDS_MHZ = {"1": (350, 1050), "2": (950, 1760), "5a": (4600, 8500), "5b": (8300, 15400)}


def _odp_reference() -> dict:
    """Reference setup per telescope, each value tagged with where it comes from."""
    from ska_ost_array_config.constants import LOW_STATION_DIAMETER_M, MID_SKA_DISH_DIAMETER_M

    from rag.subarray_layout import get_subarray_layout
    import yaml
    fd = yaml.safe_load((ROOT / "rag/vendor/setup_validator/schema/frontend_defaults.yaml").read_text())
    ref = {}
    for tel, template, diam, dump, cw_khz in (("mid", "Mid_full_AAstar", MID_SKA_DISH_DIAMETER_M, 0.14, fd["skaMidContChannelWidth"]),
                                              ("low", "Low_full_AAstar", LOW_STATION_DIAMETER_M, 0.9, fd["skaLowContChannelWidth"])):
        lay = get_subarray_layout(template)
        ref[tel] = {
            "template": (template, "assumed: the full AA* array"),
            "n_station": (lay["n_stations"], "ska_ost_array_config"),
            "max_baseline_m": (round(lay["max_baseline_m"], 1), "ska_ost_array_config"),
            "diameter_m": (diam, "ska_ost_array_config"),
            "dump_time_s": (dump, "Dewdney et al. 2022, Table 1-2" if tel == "mid" else "Dewdney et al. 2022, Table 1-3"),
            "continuum_channel_kHz": (cw_khz, "setup-validator frontend defaults"),
            "continuum_bandwidth_MHz": ({b: fd["skaMidDefaults"][f"Band {b}"]["continuumBandwidth"] for b in MID_BANDS_MHZ}
                                        if tel == "mid" else fd["skaLowDefaults"]["continuumBandwidth"], "setup-validator frontend defaults"),
            "default_freq_MHz": (1355 if tel == "mid" else fd["skaLowDefaults"]["centralFrequency"],
                                 "assumed: Band 2 centre" if tel == "mid" else "setup-validator frontend defaults"),
            "zoom_channel_kHz": (13.44 if tel == "mid" else 1.808, "zoom_settings schema (widest Mid, widest Low)"),
            "zoom_n_chan": (4000, "image_cube schema: base maximum of n_channel_zoom"),
            "cube_n_chan": (10, "Arumugam et al. 2025: 'typically 10 channels'"),
            "continuum_stokes": (4, "Arumugam et al. 2025: IQUV unless specified"),
            "zoom_stokes": (1, "assumed: Stokes I line cube"),
            "image_bits": (32, "data_products.Image default"),
            "psf_oversampling": (3, "data_products.Image default"),
            "smearing_beams": (0.25, "assumed: smearing at most a quarter beam at the half-power radius"),
            "pst_subint_s": (10.0, "assumed: 10 s sub-integrations"),
        }
    return ref


def _band(freq_mhz: float) -> str:
    """The Mid band containing freq_mhz, or the nearest one."""
    for b, (lo, hi) in MID_BANDS_MHZ.items():
        if lo <= freq_mhz <= hi:
            return b
    return min(MID_BANDS_MHZ, key=lambda b: min(abs(freq_mhz - x) for x in MID_BANDS_MHZ[b]))


def _request_freq(r: dict, ref: dict) -> float:
    if r.get("freq_mhz"):
        return float(r["freq_mhz"])
    m = re.search(r"\bBand\s*(5a|5b|1|2)\b|^\W*(5a|5b|1|2)\W*$", r.get("band") or "", re.I)
    if r["telescope"] == "mid" and m:
        lo, hi = MID_BANDS_MHZ[(m.group(1) or m.group(2)).lower()]
        return (lo + hi) / 2
    return float(ref["default_freq_MHz"])


def _request_area(r: dict, field_area: dict) -> float | None:
    """Survey area in deg², from the linked field or the request's own text."""
    if field_area.get(r.get("field")):
        return field_area[r["field"]]
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*)?deg(?:2|²|rees? squared)", " ".join(
        str(r.get(k) or "") for k in ("name", "position_note", "review_note")))
    return float(m.group(1).replace(",", "")) if m else None


def request_odps(r: dict, cats: list[str], ref: dict, area: float | None) -> list[dict]:
    """ODPs one request needs, each with its parameters and estimated volume (TB)."""
    from astropy.units import Quantity

    from rag.vendor.setup_validator import data_products as dp
    R = {k: v[0] for k, v in ref.items()}
    hours = float(r["hours"])
    dur = Quantity(hours * 3600, "s")
    out = []

    def add(kind, cat, model, params, scale=1.0):
        out.append({"odp": kind, "category": cat, "params": params, "volume_TB": scale * float(model.data_volume.to("TB").value)})

    if "Continuum" in cats or "Zoom" in cats:
        f = _request_freq(r, R) * 1e6
        lam = C_LIGHT / f
        res, pb = lam / R["max_baseline_m"], 1.1 * lam / R["diameter_m"]
        fov = max(pb, math.radians(math.sqrt(area))) if area else pb
        # Average no further than a quarter-beam smear at the half-power radius.
        t_max = R["smearing_beams"] * res / (OMEGA_EARTH * pb / 2)
        n_t = int(min(1000, max(1, t_max // R["dump_time_s"])))
        img = dict(resolution=Quantity(math.degrees(res) * 3600, "arcsec"), fov=Quantity(math.degrees(fov), "deg"),
                   n_bit=R["image_bits"], n_psf_osamp=R["psf_oversampling"])
        common = {"freq_MHz": round(f / 1e6, 1), "resolution_arcsec": round(math.degrees(res) * 3600, 3),
                  "fov_deg": round(math.degrees(fov), 3), "t_avg_s": round(n_t * R["dump_time_s"], 2)}
    if "Continuum" in cats:
        bw = (R["continuum_bandwidth_MHz"][_band(f / 1e6)] if r["telescope"] == "mid" else R["continuum_bandwidth_MHz"]) * 1e6
        cw = R["continuum_channel_kHz"] * 1e3
        nu_max = R["smearing_beams"] * f * res / (pb / 2)
        n_f = int(min(1000, max(1, nu_max // cw)))
        vis = dp.CalibratedVisibilities(n_station=R["n_station"], t_int=n_t * R["dump_time_s"], bw=bw, cbw=n_f * cw,
                                        n_stokes=4, duration=dur, storage_model="msv2")
        add("calibrated_visibilities", "Continuum", vis, {**common, "bandwidth_MHz": bw / 1e6,
            "channel_kHz": round(n_f * cw / 1e3, 1), "n_chan": vis.n_chan, "n_station": R["n_station"]})
        add("mfs_image", "Continuum", dp.Image(**img, n_chan=1, n_stokes=R["continuum_stokes"]),
            {**common, "n_stokes": R["continuum_stokes"]})
        cube = dp.Image(**img, n_chan=R["cube_n_chan"], n_stokes=R["continuum_stokes"])
        add("image_cube", "Continuum", cube, {**common, "n_chan": R["cube_n_chan"], "n_stokes": R["continuum_stokes"], "n_pix": cube.n_pix})
    if "Zoom" in cats:
        cw = R["zoom_channel_kHz"] * 1e3
        vis = dp.CalibratedVisibilities(n_station=R["n_station"], t_int=n_t * R["dump_time_s"], bw=R["zoom_n_chan"] * cw, cbw=cw,
                                        n_stokes=4, duration=dur, storage_model="msv2")
        add("calibrated_visibilities", "Zoom", vis, {**common, "bandwidth_MHz": round(R["zoom_n_chan"] * cw / 1e6, 2),
            "channel_kHz": R["zoom_channel_kHz"], "n_chan": vis.n_chan, "n_station": R["n_station"]})
        cube = dp.ZoomImage(**img, n_chan=R["zoom_n_chan"], n_stokes=R["zoom_stokes"])
        add("zoom_image_cube", "Zoom", cube, {**common, "n_chan": R["zoom_n_chan"], "n_stokes": R["zoom_stokes"], "n_pix": cube.n_pix})
    if "PSS" in cats:
        add("pss_candidates_folded", "PSS", dp.PulsarSearchProduct(mode=dp.PssModes.FOLDED, duration=dur),
            {"mode": "folded", "rate_MB_s": round(0.0839 / 1024 * 1e6, 1)})
    if "PST" in cats:
        # One archive per hour on source; the class caps n_subint at 100,000.
        n_sub = int(3600 // R["pst_subint_s"])
        arc = dp.FoldedPulsarArchive(n_subint=n_sub)
        add("pst_timing", "PST", arc, {"n_subint_per_hour": n_sub, "n_chan": int(arc.n_chan), "n_phase_bin": arc.n_phase_bin,
                                       "n_stokes": arc.n_stokes, "n_bit": arc.n_bit}, scale=hours)
    return out


def mode_odps(conn=None) -> dict:
    """Reviewed requests grouped into Continuum, Zoom, PSS and PST, with the ODPs
    each needs and their estimated volumes."""
    conn = conn or source_db.connect()
    reqs = [r for r in source_db.list_requests(conn, None, REQUESTS_RUN, include_rejected=True)
            if r.get("review_status") == "accepted"]
    field_area = {n: a for n, a in conn.execute("SELECT name, MAX(area_deg2) FROM fields GROUP BY name") if a}
    band = dict(conn.execute("SELECT id, band FROM requests"))  # list_requests leaves the band out
    ref = _odp_reference()
    rows = []
    for r in reqs:
        r = {**r, "band": band.get(r["id"])}
        cats = classify_request(r)
        area = _request_area(r, field_area)
        odps = request_odps(r, cats, ref[r["telescope"]], area) if r["telescope"] in ref else []
        rows.append({k: r.get(k) for k in ("id", "name", "paper", "telescope", "hours")}
                    | {"group": r.get("commensal_group"), "categories": cats, "area_deg2": area, "odps": odps,
                       "total_TB": sum(o["volume_TB"] for o in odps)})
    # Commensal requests share one observation: count each ODP once per group, at its largest member.
    best = {}
    for r in rows:
        for o in r["odps"]:
            key = (r["group"] or f"id{r['id']}", o["odp"], o["category"])
            if key not in best or o["volume_TB"] > best[key][1]["volume_TB"]:
                best[key] = (r, o)
    totals = defaultdict(lambda: defaultdict(float))
    for r, o in best.values():
        totals[f"{r['telescope']}|{o['category']}"][o["odp"]] += o["volume_TB"]
    hours = defaultdict(float)
    for r in rows:  # requested hours split evenly over a request's categories
        for c in r["categories"]:
            hours[f"{r['telescope']}|{c}"] += r["hours"] / len(r["categories"])
    by_cat = {c: [m for m, k in MODE_CATEGORY.items() if k == c] for c in CATEGORIES}
    cov = _coverage({c: "|".join(MODE_TERMS[m] for m in ms) for c, ms in by_cat.items()})
    return {"run": REQUESTS_RUN, "mode_category": MODE_CATEGORY, "coverage": cov,
            "reference": {t: {k: {"value": v[0], "source": v[1]} for k, v in d.items()} for t, d in ref.items()},
            "requests": rows, "hours": dict(hours),
            "volume_TB": {k: dict(v) for k, v in totals.items()}}


# Frequency coverage of each SKA receiver band, GHz (Band 6 is proposed, not built).
BAND_GHZ = {"Low": (0.05, 0.35), "1": (0.35, 1.05), "2": (0.95, 1.76), "3": (1.65, 3.05), "4": (2.8, 5.18),
            "5a": (4.6, 8.5), "5b": (8.3, 15.4), "5": (4.6, 15.4), "6": (15.4, 24.0)}
_BANDS = re.compile(r"\bBands?\s+((?:\d[ab]?(?![\d.])|and|or|to|Band|[,+\s])+)")
_BANDWIDTH = re.compile(r"[~≈>=\s]*\d+(?:\.\d+)?\s*(?:MHz|kHz|GHz)\s*(?:contiguous\s*)?(?:bandwidth|subbands?|sub-bands?)"
                        r"|\(\d+(?:\.\d+)?\s*MHz contiguous\)")
_NUM = r"[~≈<>]?=?\s*(\d+(?:\.\d+)?)"
_UNIT = {"MHz": 1e-3, "GHz": 1.0}
_CROSS = re.compile(rf"{_NUM}\s*(MHz|GHz)\s*(?:-|–|to)\s*{_NUM}\s*(MHz|GHz)")
_RANGE = re.compile(rf"{_NUM}\s*(?:-|–|to)\s*{_NUM}\s*(MHz|GHz)\b")
_LIST = re.compile(rf"{_NUM}((?:\s*(?:,|and|or)\s*[~≈<>]?=?\s*\d+(?:\.\d+)?)*)\s*(MHz|GHz|cm)\b")


def freq_range(text: str | None) -> list[list[float]] | None:
    """GHz intervals [[lo, hi], ...] named by a requirement's free-text
    frequency field: stated ranges, single frequencies (lo == hi),
    wavelengths and band names. None when it names none. Bandwidths are
    not frequencies and are ignored; a list of bands stays a list, so
    "Bands 1 and 5a" does not claim the gap between them."""
    if not text:
        return None
    text = _BANDWIDTH.sub(" ", text)
    out = [list(BAND_GHZ[b]) for m in _BANDS.finditer(text) for b in re.findall(r"\d[ab]?", m.group(1)) if b in BAND_GHZ]
    if re.search(r"SKA-Low|\bLow\b", text) and not re.search(r"\d\s*MHz", text):
        out.append(list(BAND_GHZ["Low"]))
    text = _BANDS.sub(" ", text)
    for rx, pick in ((_CROSS, lambda m: [float(m[1]) * _UNIT[m[2]], float(m[3]) * _UNIT[m[4]]]),
                     (_RANGE, lambda m: [float(m[1]) * _UNIT[m[3]], float(m[2]) * _UNIT[m[3]]])):
        out += [pick(m) for m in rx.finditer(text)]
        text = rx.sub(" ", text)
    for m in _LIST.finditer(text):
        for x in [m[1], *re.findall(r"\d+(?:\.\d+)?", m[2])]:
            v = float(x)
            if v:
                f = 30 / v if m[3] == "cm" else v * _UNIT[m[3]]
                out.append([f, f])
    return sorted([round(min(i), 4), round(max(i), 4)] for i in out) or None


def requirement_freqs() -> dict:
    """Frequency span of every register row (from data/prose.json, written by
    verify_atlas_quotes.py) for the coverage strip on the requirements page."""
    path = ROOT / "aaskaii-atlas-v2" / "data" / "prose.json"
    rows = [r for s in json.loads(path.read_text())["sections"] for r in s["requirements"]] if path.exists() else []
    spans = {r["id"]: freq_range(r.get("frequency")) for r in rows}
    return {"bands": BAND_GHZ, "freq_ghz": {k: v for k, v in spans.items() if v},
            "n_rows": len(rows), "n_with_text": sum(1 for r in rows if r.get("frequency"))}


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
        "requirement_freqs": (requirement_freqs, {"source": "data/prose.json"}),
        "mode_odps": (mode_odps, {"extracted_by": REQUESTS_RUN, "reviewed_only": True}),
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
