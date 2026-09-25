"""Check every quote the atlas section agents returned against the chapter it
is attributed to, and write the surviving rows for the pages.

    python scripts/verify_atlas_quotes.py

Reads aaskaii-atlas-v2/data/agents/sections.json (the workflow's output:
{sections, sensitivity, capability}) and writes data/prose.json + prose.js.
A quote is kept when it appears verbatim in the paper (source_db.find_quote,
the same check add_observing_request applies), with its page corrected to
the chunk it was found in, or when each of its sentences is (a quote that
crosses a chunk boundary); otherwise when verify_quote judges it
supported_by_claimed_paper (marked "near-verbatim"). Everything else is
dropped and listed under "rejected" so the method page can report it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import corpus_tools, source_db  # noqa: E402

DATA = ROOT / "aaskaii-atlas-v2" / "data"


def check(row: dict) -> dict | None:
    """The row with a 'verification' field, or None if the quote fails."""
    meta = source_db.find_quote(row["quote"], row["paper"], "aaskaii")
    if meta:
        return {**row, "page": meta.get("page_no", row.get("page")), "verification": "verbatim"}
    # A quote that crosses a chunk boundary is found sentence by sentence.
    parts = [p for p in re.split(r"(?<=[.;])\s+", row["quote"]) if len(p) >= 40]
    metas = [source_db.find_quote(p, row["paper"], "aaskaii") for p in parts]
    if len(parts) > 1 and all(metas):
        return {**row, "page": metas[0].get("page_no", row.get("page")), "verification": "verbatim"}
    v = corpus_tools.verify_quote(row["quote"][:400], row["paper"], 3, None, "aaskaii")
    if v["verdict"] == "supported_by_claimed_paper":
        return {**row, "verification": "near-verbatim"}
    return None


def main() -> None:
    src = json.loads((DATA / "agents" / "sections.json").read_text())
    rejected, sections = [], []
    for s in src["sections"]:
        kept = {}
        for kind in ("requirements", "open_questions"):
            kept[kind] = []
            for row in s[kind]:
                ok = check(row)
                if ok:
                    kept[kind].append(ok)
                else:
                    rejected.append({"section": s["section"], "kind": kind, **row})
        sections.append({**s, **kept})
    kept_ids = {r["id"] for s in sections for r in s["requirements"]}
    out = {"sections": sections, "rejected": rejected,
           "sensitivity": [c for c in src.get("sensitivity") or [] if c["id"] in kept_ids],
           "capability": [c for c in src.get("capability") or [] if c["id"] in kept_ids],
           "counts": {"quotes": sum(len(s[k]) for s in src["sections"] for k in ("requirements", "open_questions")),
                      "rejected": len(rejected)}}
    text = json.dumps(out, indent=1)
    (DATA / "prose.json").write_text(text)
    (DATA / "prose.js").write_text(f"(window.ATLAS = window.ATLAS || {{}})['prose'] = {text};\n")
    print(json.dumps(out["counts"]), f"verbatim={sum(r['verification'] == 'verbatim' for s in sections for k in ('requirements', 'open_questions') for r in s[k])}")


if __name__ == "__main__":
    main()
