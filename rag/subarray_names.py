"""
Resolves subarray-template-like names across the three SKA naming
vocabularies touched elsewhere in this repo:

1. The live sensitivity calculator (rag/sensitivity_calculator.py) — entries
   like {"name": "LOW_AAstar_all", "label": "AA*", "n_stations": 307}.
2. ska_ost_array_config's canonical template catalog (rag/subarray_layout.py)
   — case-insensitive names like "LOW_FULL_AASTAR", "LOW_INNER_R10KM_AASTAR".
3. The vendored setup-validator schema's per-context allowed `template`
   values (rag/observing_setup_capabilities.py) — e.g. "Low_full_AA2".

What actually querying all three showed (SKA-Low and SKA-Mid, 2026-09-05):

- "Region" sensitivity-calculator entries (e.g. "LOW_inner_r10km_aastar",
  "MID_inner_r125m_aa4") match a ska_ost_array_config canonical template
  name 1:1, case-insensitively.
- "Whole array" sensitivity-calculator entries use a different pattern,
  "<TEL>_<RELEASE>_all" (e.g. "LOW_AA2_all", "MID_AA4_all") rather than
  ska_ost_array_config's "<TEL>_FULL_<RELEASE>". Confirmed by matching
  n_stations (LOW_AA2_all=68 == LOW_FULL_AA2=68, etc.) that "_all" means
  "the FULL template for that release". AA0.5 and AA1 have "_all"
  sensitivity-calculator entries but NO ska_ost_array_config template at
  all — those early releases aren't modelled there.
- A few sensitivity-calculator aliases (*_core_only, *_AAstar_SKA_only,
  *_MeerKAT_only) are calculator-specific antenna-subset filters of the
  full array with no standalone ska_ost_array_config template name; they
  resolve to no ska_ost_array_config match and are reported as such
  rather than guessed at.
- setup-validator schema `template` values match ska_ost_array_config
  names case-insensitively 1:1 (they're already in "<TEL>_FULL_<RELEASE>"
  / "<TEL>_<region>_<RELEASE>" form), and so resolve to sensitivity-
  calculator entries via the same two rules above. One observed
  exception: schema context "sv_aastar"/"cycle_1" for ska_low lists
  "Low_substation_18m_r1km_AAstar", missing the "_inner_" that "cycle_0"'s
  "Low_substation_18m_inner_r1km_AAstar" and ska_ost_array_config's
  "LOW_SUBSTATION_18M_INNER_R1KM_AASTAR" both have — this one template
  legitimately resolves to no ska_ost_array_config/sensitivity-calculator
  match (probably a schema typo upstream; not something to silently paper
  over here).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from rag.observing_setup_capabilities import describe_schema
from rag.sensitivity_calculator import query_sensitivity_calculator

_CALC_TELESCOPES = ("low", "mid")
_SCHEMA_TELESCOPE_IDS = ("ska_low", "ska_mid")
_SCHEMA_ROOT = Path(__file__).resolve().parent / "vendor" / "setup_validator" / "schema"

_ALL_SUFFIX_RE = re.compile(r"^(low|mid)_([a-z0-9.]+)_all$", re.IGNORECASE)


def _normalize_telescope(telescope: str | None) -> str | None:
    """Accept "low"/"mid" or "ska_low"/"ska_mid" and return "low"/"mid" (or None)."""
    if telescope is None:
        return None
    t = telescope.lower().replace("ska_", "").replace("ska-", "")
    if t not in _CALC_TELESCOPES:
        raise ValueError(f"Unknown telescope {telescope!r}; expected one of low/mid/ska_low/ska_mid")
    return t


@lru_cache(maxsize=1)
def _sensitivity_subarrays() -> dict[str, list[dict]]:
    """Live sensitivity-calculator subarrays lists, cached for this process.

    ponytail: module-level cache with no TTL/refresh — this is a single
    long-lived MCP server process (per the task), so "restart to refresh"
    is an acceptable ceiling; add TTL-based invalidation if that changes.
    """
    return {t: query_sensitivity_calculator(t, "subarrays", {}) for t in _CALC_TELESCOPES}


@lru_cache(maxsize=1)
def _ska_ost_templates() -> frozenset[str]:
    from ska_ost_array_config import get_supported_templates

    return frozenset(get_supported_templates())


@lru_cache(maxsize=1)
def _schema_contexts() -> tuple[str, ...]:
    return tuple(sorted(p.name for p in _SCHEMA_ROOT.iterdir() if p.is_dir()))


@lru_cache(maxsize=1)
def _all_schema_templates() -> tuple[tuple[str, str, str], ...]:
    """(context, schema_telescope_id, template) for every allowed subarray
    template across every context/telescope that defines one."""
    rows = []
    for ctx in _schema_contexts():
        for tel_id in _SCHEMA_TELESCOPE_IDS:
            try:
                spec = describe_schema("subarray_config", context=ctx, telescope=tel_id)
            except Exception:
                continue
            template_spec = spec.get("template") or {}
            for tpl in template_spec.get("allowed_values", []):
                rows.append((ctx, tel_id, tpl))
    return tuple(rows)


def _all_alias_to_full(name: str) -> str | None:
    """"<TEL>_<RELEASE>_all" -> "<TEL>_FULL_<RELEASE>" (case-insensitive), else None."""
    m = _ALL_SUFFIX_RE.match(name)
    if not m:
        return None
    tel, release = m.groups()
    return f"{tel}_FULL_{release}".upper()


def _to_canonical(name: str) -> str:
    """Best-effort canonical ska_ost_array_config name for `name`.

    Returns the uppercased ska_ost_array_config template name when `name`
    is (or aliases to, via the "_all" convention) a known template; falls
    back to the plain uppercased input when there's no known translation
    (e.g. AA0.5/AA1 "_all" aliases, or the *_core_only/*_SKA_only/
    *_MeerKAT_only sensitivity-calculator aliases), so unrelated inputs
    still compare equal/unequal sensibly rather than crashing.
    """
    templates = _ska_ost_templates()
    up = name.upper()
    if up in templates:
        return up
    alias = _all_alias_to_full(name)
    if alias is not None and alias in templates:
        return alias
    return up


def resolve_subarray_name(name: str, telescope: str | None = None) -> dict:
    """Look up a subarray-template-like name across all three SKA naming
    vocabularies this repo touches, and report what matches.

    Args:
        name: A subarray-template-like name in any of the three
            vocabularies, e.g. "LOW_AAstar_all" (sensitivity calculator),
            "LOW_FULL_AASTAR" (ska_ost_array_config canonical), or
            "Low_full_AA2" (setup-validator schema `template` value).
        telescope: Restrict to "low"/"mid" (or "ska_low"/"ska_mid"). None
            checks both.

    Returns:
        dict with:
            "input": the name as given.
            "canonical_ska_ost_array_config_name": the resolved canonical
                ska_ost_array_config template name, or None if `name`
                doesn't resolve to one (e.g. it's a calculator-only alias
                like "*_core_only" with no standalone template).
            "sensitivity_calculator_matches": {telescope: [entry, ...]} for
                each telescope with a matching live sensitivity-calculator
                subarrays entry.
            "setup_validator_matches": [{"context", "telescope", "template"},
                ...] for each context/telescope where this resolves to an
                allowed subarray_config `template` value.
    """
    tel = _normalize_telescope(telescope)
    telescopes = (tel,) if tel else _CALC_TELESCOPES
    canonical = _to_canonical(name)
    is_known = canonical in _ska_ost_templates()

    sensitivity_matches: dict[str, list[dict]] = {}
    for t in telescopes:
        matches = [e for e in _sensitivity_subarrays()[t] if _to_canonical(e["name"]) == canonical]
        if matches:
            sensitivity_matches[t] = matches

    schema_telescope_ids = (
        (f"ska_{tel}",) if tel else _SCHEMA_TELESCOPE_IDS
    )
    setup_validator_matches = [
        {"context": ctx, "telescope": tel_id, "template": tpl}
        for ctx, tel_id, tpl in _all_schema_templates()
        if tel_id in schema_telescope_ids and _to_canonical(tpl) == canonical
    ]

    return {
        "input": name,
        "canonical_ska_ost_array_config_name": canonical if is_known else None,
        "sensitivity_calculator_matches": sensitivity_matches,
        "setup_validator_matches": setup_validator_matches,
    }


def resolve_context_subarrays(context: str, telescope: str) -> list[dict]:
    """For a setup-validator context, resolve each allowed subarray
    `template` value to its live sensitivity-calculator entry, so an agent
    can chain describe_schema("subarray_config", context, telescope) ->
    a sensitivity-calculator call without guessing a name translation.

    Args:
        context: Observing context, e.g. "sv_aa2", "cycle_0" (see
            rag.observing_setup_capabilities.describe_schema).
        telescope: "low"/"mid" or "ska_low"/"ska_mid".

    Returns:
        list of {"template": str, "sensitivity_calculator_entry": dict | None}
        — one entry per allowed template, in schema order; entry is None
        when no live sensitivity-calculator subarray matches (e.g. the
        schema typo noted in this module's docstring).

    Raises:
        ValueError: telescope not recognised, or the context/telescope has
            no subarray_config schema (see describe_schema).
    """
    tel = _normalize_telescope(telescope)
    schema_tel_id = f"ska_{tel}"
    spec = describe_schema("subarray_config", context=context, telescope=schema_tel_id)
    templates = (spec.get("template") or {}).get("allowed_values", [])

    calc_entries = _sensitivity_subarrays()[tel]
    results = []
    for tpl in templates:
        canonical = _to_canonical(tpl)
        entry = next((e for e in calc_entries if _to_canonical(e["name"]) == canonical), None)
        results.append({"template": tpl, "sensitivity_calculator_entry": entry})
    return results
