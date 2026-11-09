"""Run the atlas-critique regression cases against the live index and tools.

    uv run python scripts/regression_cases.py --out docs/regression/before.json
    uv run python scripts/regression_cases.py --table docs/regression/before.json docs/regression/after.json

Each case records what the tools report today, so the same script run before
and after the tool changes gives the before/after table in
docs/atlas-tool-changes.md. Keyword cases run twice: with the settings
scripts/build_atlas_data.py uses (min_hits_per_paper=2, chapters only) and
with the tool's own defaults. Cases that need a parameter the current code
doesn't have record "unsupported" rather than failing, so the script runs on
both sides of every change. Calculator and SIMBAD cases make live calls.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import corpus_tools, scheduling, source_db  # noqa: E402
from rag.bibliography import list_toc_entries  # noqa: E402

KEYWORD_CASES = {
    "K1": "ApJ|MNRAS|A&A",
    "K2": "Chandra",
    "K2b": "Chandra X-ray",
    "K3": "Planck",
    "K4": "pulsar survey",
    "K5": "data rate|data volume",
    "K6": "zoom mode|zoom-mode|zoom window",
    "K7": "VLBI",
}
COPIES = ["Curiel01b", "Curiel01c", "Kurahara01b", "Kurahara01c", "Kurahara01_3", "Oswald01b", "contents_webpage"]
REQUESTS_RUN = "full-sonnet5-medium"  # the extraction run the atlas used
CALC_V12 = "https://sensitivity-calculator.skao.int/api/v12"
POINTING = "00:00:00 -30:00:00"


def _has_param(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


def _papers(term_result: dict) -> list[str]:
    return [p for s in term_result["sections"].values() for p in s["papers"]]


def keyword_cases() -> dict:
    chapters = {Path(e["path"]).stem for es in list_toc_entries(corpus_tools.BOOKS["aaskaii"]).values() for e in es}
    out = {}
    for label, kwargs in (("atlas", {"min_hits_per_paper": 2}), ("defaults", {})):
        cov = {t["term"]: t for t in corpus_tools.keyword_coverage(list(KEYWORD_CASES.values()), **kwargs)["terms"]}
        for cid, term in KEYWORD_CASES.items():
            papers = _papers(cov[term])
            # Atlas settings count distinct chapters only, as build_atlas_data._restrict_term does.
            n = len({p for p in papers if p in chapters}) if label == "atlas" else cov[term]["n_papers"]
            out.setdefault(cid, {"term": term})[label] = n
            if cid == "K4":
                out[cid][f"{label}_has_Joshi01"] = "Joshi01" in papers
        if "_index" in cov or "index" in cov:
            out.setdefault("I3", {})["fingerprint"] = True
    # Plural variants counted literally today ("data rates"), and bare "zoom", for comparison.
    lit = {t["term"]: t["n_papers"] for t in corpus_tools.keyword_coverage(
        ["data rates|data volumes", "zoom", "pulsar surveys"])["terms"]}
    out["K5"]["defaults_literal_plural_only"] = lit["data rates|data volumes"]
    out["K6"]["defaults_bare_zoom"] = lit["zoom"]
    return out


def index_cases() -> dict:
    cov = corpus_tools.keyword_coverage(["ApJ|MNRAS|A&A"])
    papers = _papers(cov["terms"][0])
    toc = [Path(e["path"]).stem for es in list_toc_entries(corpus_tools.BOOKS["aaskaii"]).values() for e in es]
    return {
        "I1": {"what": "Bianchi01 listed in list_documents / counted in keyword_coverage",
               "list_documents": toc.count("Bianchi01"), "keyword_coverage": papers.count("Bianchi01")},
        "I2": {"what": "copies counted by keyword_coverage by default",
               "copies_counted": sorted(c for c in COPIES if c in papers)},
        "I3": {"what": "index fingerprint returned",
               "manifest": (ROOT / "chroma_db" / "manifest.json").exists(),
               "fingerprint_in_response": "_index" in cov},
    }


def _compare(cid: str, **kw) -> dict:
    """One compare_to_calculator call, keeping only the fields the table needs."""
    from rag import sensitivity_calculator as sc
    fn = sc.compare_to_calculator
    unsupported = [k for k in kw if not _has_param(fn, k)]
    if unsupported:
        return {"unsupported": unsupported}
    try:
        r = fn(**kw)
    except Exception as exc:  # live API: record, don't abort the run
        return {"error": str(exc)[:200]}
    best = r.get("closest") or {}
    strict = r.get("strict") or {}
    return {
        "closest_pct_diff": best.get("pct_diff"), "closest_variant": {k: best.get(k) for k in
                                                                      ("weighting_mode", "robustness", "subarray_configuration", "freq_centre_hz", "freq_centre_mhz") if best.get(k) is not None},
        "strict_pct_diff": (strict.get("result") or {}).get("pct_diff") if strict else None,
        "strict_error": strict.get("error") if strict else None,
        "defaults_applied": strict.get("defaults_applied") if strict else None,
        "array_substituted": r.get("array_substituted") or strict.get("array_substituted"),
        "n_free_parameters": r.get("n_free_parameters"),
        "variant_errors": sorted({_detail(v["error"]) for v in r.get("variants", []) if "error" in v})[:2],
    }


def _detail(err: str) -> str:
    """The calculator's own 'detail' message from an HTTPError string."""
    i = err.find("'detail': ")
    return err[i + 11:i + 120].split("'")[0] if i >= 0 else err[-100:]


def calculator_cases() -> dict:
    from rag import sensitivity_calculator as sc
    if "/v11" in sc.BASE_URL:  # v11 was retired upstream; run the old logic on the live v12 API
        sc.BASE_URL = CALC_V12
    b03 = dict(rx_band="Band 2", freq_centre_hz=1.4e9, bandwidth_hz=810e6, integration_time_s=3600,
               pointing_centre=POINTING, subarray_configuration="AA*")
    e03 = dict(freq_centre_mhz=150, bandwidth_mhz=200, integration_time_h=8, pointing_centre=POINTING,
               subarray_configuration="LOW_AA4_all", weighting_mode="robust", robustness=-1)
    c02 = dict(rx_band="Band 5b", freq_centre_hz=11.85e9, bandwidth_hz=1e9, integration_time_s=300 * 3600,
               pointing_centre=POINTING, subarray_configuration="AA*")
    f33 = dict(rx_band="Band 2", freq_centre_hz=1.4e9, bandwidth_hz=808e6, integration_time_s=4 * 3600,
               pointing_centre=POINTING, subarray_configuration="AA4")
    zoom = dict(rx_band="Band 2", freq_centres_hz=[1.4e9], total_bandwidths_hz=[3.2e6], spectral_resolutions_hz=[13440],
                integration_time_s=3600, pointing_centre=POINTING, weighting_mode="robust", robustness=0)
    out = {
        "S1": {"case": "B-03 (15.8 uJy/beam, AA*, 1 h, 810 MHz)", "best_fit": _compare(
            "S1", telescope="mid", params=b03, claimed_sensitivity=15.8),
               # What the atlas agent did: move the centre to 1.355 GHz to fit 810 MHz in Band 2.
               "best_fit_sweep": _compare("S1", telescope="mid", params=b03, claimed_sensitivity=15.8,
                                          sweep={"freq_centre_hz": [1.4e9, 1.355e9]})},
        "S2": {"case": "E-03 (5 uJy/beam, AA4, 8 h, robust -1; 150 vs 200 MHz)",
               "strict": _compare("S2", telescope="low", params=e03, claimed_sensitivity=5, mode="strict"),
               "best_fit_sweep": _compare("S2", telescope="low", params=e03, claimed_sensitivity=5,
                                          sweep={"freq_centre_mhz": [150, 200]})},
        "S3": {"case": "C-02 (0.02 K, 8.17 arcsec beam, Band 5b, AA*)",
               "result": _compare("S3", telescope="mid", params=c02, claimed_sensitivity=0.02, unit="K",
                                  quantity="spectral", claimed_beam_arcsec=8.17, claimed_channel_width_hz=7.2e3,
                                  mode="strict")},
        "S4": {"case": "F-33 (2 uJy as 5 sigma, AA4, 4 h)",
               "as_1sigma": _compare("S4", telescope="mid", params=f33, claimed_sensitivity=2, mode="strict"),
               "as_5sigma": _compare("S4", telescope="mid", params=f33, claimed_sensitivity=2, mode="strict",
                                     sigma_level=5)},
    }
    zoom_ok = {}
    for sub in ("AA* (MID_inner_r2km)", "MID_inner_r2km_aastar"):
        try:
            r = sc.query_sensitivity_calculator("mid", "zoom/calculate", {**zoom, "subarray_configuration": sub})
            zoom_ok[sub] = r.get("subarray_used", "ok")
        except Exception as exc:
            zoom_ok[sub] = "error: " + _detail(str(exc))
    out["S5"] = {"case": "zoom accepts subarray name and label", **zoom_ok}
    out["S6"] = {"case": "robust 0.5", "result": _compare(
        "S6", telescope="mid", claimed_sensitivity=2,
        params=dict(rx_band="Band 2", freq_centre_hz=1.31e9, bandwidth_hz=700e6, integration_time_s=3600,
                    pointing_centre=POINTING, subarray_configuration="AA*", weighting_mode="robust", robustness=0.5),
        mode="strict")}
    return out


def capability_cases() -> dict:
    from rag import observing_setup_validator as v
    fn = v.validate_across_contexts
    out = {"V1": {"case": "default contexts", "contexts": list(getattr(v, "DEFAULT_CONTEXTS", v.CONTEXTS))}}
    if _has_param(fn, "requirement"):
        vlbi = fn(None, requirement={"telescope": "mid", "mode": {"value": "vlbi"}}, check_sensitivity=False)
        out["V2"] = {"case": "VLBI requirement", "verdict": vlbi["verdict"]}
        hard = fn(None, requirement={"telescope": "mid", "band": {"value": "5b"},
                                     "sensitivity": {"value": 1, "unit": "uJy/beam"},
                                     "angular_resolution": {"value": 0.1, "unit": "arcsec"},
                                     "hours": {"value": 10, "unit": "h"}}, check_sensitivity=True)
        out["V3"] = {"case": "Band 5b, 1 uJy, 0.1 arcsec", "verdict": hard["verdict"],
                     "failing": hard["failing_dimensions"],
                     "SV-AA*": {d: x["verdict"] for d, x in hard["by_context"]["SV-AA*"]["dimensions"].items()}}
    else:
        out["V2"] = out["V3"] = {"unsupported": ["requirement"]}
    return out


def quote_cases() -> dict:
    quote = ("In this example we show how we can detect and resolve the emission at SKA-Mid Band 5a and 5b, "
             "with 10 hours integration time, reaching a rms of about 1.1 and 1.3 \U0001d707 Jy/beam and "
             "achieving a peak SNR of about 4-8 and 8-13 (depending from the disk inclination), respectively.")
    try:
        from rag.requirements import check_fields
    except ImportError:
        return {"Q1": {"case": "C-19 weighting stated or inferred", "unsupported": ["rag.requirements"]}}
    fields = {"weighting": {"value": "robust -2", "source_text": None, "inferred": False},
              "sensitivity": {"value": 1.1, "unit": "uJy/beam", "source_text": "1.1", "inferred": False},
              "hours": {"value": 10, "unit": "h", "source_text": "10 hours", "inferred": False}}
    checks = check_fields(quote, fields)
    return {"Q1": {"case": "C-19 weighting stated or inferred", **{k: v["verdict"] for k, v in checks.items()}}}


def time_cases() -> dict:
    conn = source_db.connect()
    every = source_db.list_requests(conn, None, REQUESTS_RUN, include_rejected=True)
    accepted = [r for r in every if r.get("review_status") == "accepted"]
    mid = [r for r in accepted if r["telescope"] == "mid"]
    kw = {"min_elevation_deg": [15, 20, 30, 45]} if _has_param(scheduling.lst_pressure, "min_elevation_deg") else {}
    p = scheduling.lst_pressure(mid, "mid")
    placed = p["total_demand_h"]
    total = sum(r["hours"] for r in mid)
    out = {
        "T1": {"case": "Mid demand/supply ratio returned",
               "returned": p.get("total_demand_supply_ratio"),
               "accepted_hours": round(total), "supply_h": p["total_supply_h"],
               "computed_ratio": round(total / p["total_supply_h"], 2)},
        "T2": {"case": "share of Mid hours placed on the sky", "placed_h": placed,
               "placed_fraction": round(placed / total, 3),
               "all_sky_hi_placed": not any("All-sky HI" in u["name"] for u in p["unplaced"])},
        "T4": {"case": "sun constraint applied by default", "assumption": p["assumptions"].get("sun"),
               "apply_sun_constraints": p["assumptions"].get("apply_sun_constraints")},
        "T5": {"case": "conflicting duplicate hours reported",
               "conflicts": len(p["conflicts"]) if "conflicts" in p else "unsupported"},
    }
    if kw:
        multi = scheduling.lst_pressure(mid, "mid", **kw)
        out["T2"]["placed_h_by_min_el"] = {k: v["total_demand_h"] for k, v in multi["by_min_elevation"].items()}
    vis = {}
    for name, ra, dec in (("Cygnus A", 299.868, 40.734), ("M31", 10.685, 41.269)):
        for el in (15, 20, 45):
            w = scheduling.visibility_windows(ra, dec, "mid", min_elevation=el)
            vis[f"{name} @{el}"] = {"max_el": w["max_elevation_deg"], "usable_h_per_year": w["usable_hours_per_year"]}
    out["T3"] = {"case": "Cygnus A and M31 from Mid", **vis,
                 "never_visible_at_default": p["never_visible"]}
    return out


def data_product_cases() -> dict:
    from rag import data_product_estimator as dpe
    out = {}
    types = getattr(dpe, "PRODUCT_TYPES", None)
    out["D1"] = {"case": "single-dish / wide-band product types",
                 "types": sorted(types) if types else "unsupported (zoom-window sizing in build_atlas_data)"}
    if types and "Spectral Cube (wide-band)" in types:
        hi = dpe.estimate_data_product_size("Spectral Cube (wide-band)", {
            "telescope": "mid", "freq_min_mhz": 950, "freq_max_mhz": 1420, "channel_width_khz": 13.44,
            "resolution_arcsec": 7.0, "fov_deg": 1.0, "n_stokes": 1, "n_beam": 1, "n_station": 144,
            "duration_h": 10000, "t_int_s": 0.14})
        im = dpe.estimate_data_product_size("Single-dish Autocorrelation", {
            "telescope": "mid", "freq_min_mhz": 350, "freq_max_mhz": 1050, "channel_width_khz": 13.44,
            "n_dish": 144, "n_pol": 2, "duration_h": 10000, "t_int_s": 2.0})
        out["D2"] = {"case": "all-sky HI cube", "bandwidth_MHz": hi["bandwidth_MHz"], "n_chan": hi["n_chan"],
                     "tiers_TB": hi["tiers_TB"]}
        out["D3"] = {"case": "Band 1 intensity mapping", "bandwidth_MHz": im["bandwidth_MHz"], "n_chan": im["n_chan"],
                     "tiers_TB": im["tiers_TB"]}
    else:
        odps = json.loads((ROOT / "aaskaii-atlas-v2/data/mode_odps.json").read_text())["requests"]
        hi = next(r for r in odps if r["name"] == "All-sky HI survey")
        zoom_vis = next(o for o in hi["odps"] if o["category"] == "Zoom" and o["odp"] == "calibrated_visibilities")
        out["D2"] = {"case": "all-sky HI cube (atlas v2 sizing)", "bandwidth_MHz": zoom_vis["params"]["bandwidth_MHz"],
                     "n_chan": zoom_vis["params"]["n_chan"], "tiers_TB": "unsupported (one total)"}
        out["D3"] = {"case": "Band 1 intensity mapping", "result": "unsupported (sized as zoom window)"}
    return out


def citation_cases() -> dict:
    from rag import citation_db
    conn = sqlite3.connect(ROOT / "citations.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(refs)")}
    journals = tuple(sorted(citation_db.JOURNAL_ABBREVIATIONS)) if hasattr(citation_db, "JOURNAL_ABBREVIATIONS") else (
        "apj", "mnras", "araa", "aa", "aj", "pasa", "nature", "science", "d", "physics", "apjl", "apjs", "pasp")
    n_bad = conn.execute(f"SELECT COUNT(*) FROM refs WHERE first_author IN ({','.join('?' * len(journals))})",
                         journals).fetchone()[0]
    out = {"C1": {"case": "references whose first author parses as a journal", "n": n_bad},
           "C2": {"case": "refs grouped by bibcode", "bibcode_column": "bibcode" in cols,
                  "n_with_bibcode": conn.execute("SELECT COUNT(*) FROM refs WHERE bibcode IS NOT NULL").fetchone()[0]
                  if "bibcode" in cols else 0}}
    try:
        from rag.sky_lookup import resolve_object
        r = resolve_object("Cygnus A")
        out["C3"] = {"case": "Cygnus A display name", "name": r.get("name"),
                     "display_name": r.get("display_name", "unsupported"), "confidence": r.get("confidence")}
    except Exception as exc:
        out["C3"] = {"case": "Cygnus A display name", "error": str(exc)[:120]}
    return out


def run() -> dict:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"generated": {"date": date.today().isoformat(), "git_sha": sha}}
    for name, fn in (("keyword", keyword_cases), ("index", index_cases), ("calculator", calculator_cases),
                     ("capability", capability_cases), ("quote", quote_cases), ("time_demand", time_cases),
                     ("data_products", data_product_cases), ("citations", citation_cases)):
        print(f"... {name}", file=sys.stderr)
        out[name] = fn()
    return out


def _cell(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))[:220].replace("|", "\\|")


def table(before: dict, after: dict) -> str:
    """Markdown before/after table, one row per case."""
    rows = ["| Case | What | Before | After |", "|---|---|---|---|"]
    for group in before:
        if group == "generated":
            continue
        for cid, b in before[group].items():
            a = after.get(group, {}).get(cid, {})
            what = b.get("case") or b.get("term") or b.get("what") or cid
            strip = lambda d: {k: v for k, v in d.items() if k not in ("case", "term", "what")}
            rows.append(f"| {cid} | {what} | {_cell(strip(b))} | {_cell(strip(a))} |")
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--table", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.table:
        print(table(*(json.loads(p.read_text()) for p in args.table)))
    else:
        result = run()
        text = json.dumps(result, indent=1, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        else:
            print(text)
