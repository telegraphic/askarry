# Changelog

Changes to the MCP tools that alter what they return. Each entry says what a
caller has to change and which existing counts move. See
`docs/atlas-tool-changes.md` for what each change means for the AASKAII Atlas.

## Unreleased

### Data products, citations and objects (WS7, WS8)

- **`estimate_data_product_size_tool` gains two whole-observation modes.**
  `"Spectral Cube (wide-band)"` sizes a line cube across a whole band (an
  HI survey) rather than one 4,000 × 13.44 kHz zoom window (54 MHz).
  `"Single-dish Autocorrelation"` sizes intensity mapping from
  autocorrelation spectra (no cross-correlations). Both return `tiers_TB`:
  `visibilities_TB`, `archived_TB` and `delivered_TB`.
- Every data-product result now says whether the archive keeps the product
  and gives `policy_source`. The archive policy is an unverified
  assumption in `config.DATA_PRODUCT_POLICY` (science products archived;
  visibilities kept only for processing).
- **Citations are grouped by DOI or ADS bibcode.** References now record
  `journal` and an ADS `bibcode` built offline from journal, year, volume,
  page and first initial (about 20 journals; none for five-digit pages).
  Refs sharing a DOI, bibcode, arXiv id, or first author + year + volume +
  page are merged (the last catches citations whose journal the PDF text
  lost), so the two books' citation styles meet; the surviving key prefers
  the bibcode. Existing databases gain the new columns on connect. `ref_key` values change (`bib:1998AJ....115.1693C`
  instead of `condon|1998|115|1693`). Rebuild with `build_citation_db_tool`.
- Fixed: a single-author reference in the 2026 style
  (`S. P. Reynolds. ARA&A , 46:89`) took the journal as the first author.
  New `first_author_suspect` flags any first author that still looks like a
  journal abbreviation.
- **`resolve_object_tool` returns `display_name`, `main_id` and
  `confidence`.** The display name is a familiar one (Cygnus A, not
  MCG+07-41-003): the query if it is one of SIMBAD's common names, else a
  Messier number, else SIMBAD's common name, else a well-known catalogue
  name. Confidence: 1.0 when the query is one of SIMBAD's identifiers, 0.8
  when SIMBAD resolved it otherwise, 0.6 from the NED fallback. `name` is
  kept. `query_sources_tool` adds `display_name` (the name the papers use
  most).

### Time demand: lst_pressure_tool and visibility_windows_tool (WS6)

- **Demand-to-supply ratio.** `lst_pressure_tool` returns
  `total_demand_supply_ratio` (every request, placed or not, commensal
  groups counted once) and `demand_supply_ratio` (placed requests only),
  with `total_demand_h_all` and `placed_h`. `total_demand_h` is unchanged
  (placed demand).
- **Default change: elevation.** A request below its `min_elevation`
  (default 45°) but above the telescope's mechanical limit (Mid 15°; Low 0°,
  in `config.SCHEDULING`, unverified) is placed using the mechanical limit
  and listed in `elevation_relaxed` with its maximum elevation.
  `never_visible` now means never above the mechanical limit.
  `visibility_windows_tool` reports `above_mechanical_limit`.
- **`min_elevation_deg`** (number or list) sets the limit for requests
  without their own; a list returns `by_min_elevation` and a `summary`.
- **Default change: Sun constraints off.** Each request's `sun` setting
  (night, avoid twilight) is ignored unless `apply_sun_constraints=true`.
- **Declination bands and areas are placed.** A request or survey field
  with `dec_range` alone is a band over all RA; a field with only
  `area_deg2` is placed on an assumed band of that area centred on the
  site's zenith declination, listed under `footprint_assumed`.
  `add_survey_field_tool` accepts `dec_range` without `ra_range`, and
  `review_observing_request_tool(field=...)` accepts area-only fields. A
  request with no position whose position note states an area ("unnamed
  1000 deg2 field") is placed the same way, marked
  `area_from: "position_note"`.
- **`conflicts`.** With `from_db`, requests marked as duplicates whose
  hours differ from the request they repeat are listed with both papers'
  hours and quotes.
- **Repeated extraction.** `extracted_by` accepts a list of runs; the
  result's `agreement` says, per request, which runs found it and whether
  their hours agree.

### Requirement records, field checks and capability verdicts (WS5, WS4)

- **New `extract_requirements_tool(quote)`** returns a draft requirement
  record and its schema. Each field (frequency, bandwidth, channel width,
  sensitivity, sigma level, weighting, array stage, band, hours, area, FoV,
  angular resolution, time resolution, mode, telescope) is
  `{value, unit, source_text, inferred}`; the parser fills what it finds in
  the quote. Rules: `rag/requirements.py`.
- **`verify_quote_tool(fields=...)`** checks each field of a record against
  the quote: `stated` (the value appears, in any unit spelling) or
  `inferred`, with a flag when a field marked stated isn't in the quote.
- **Default change: `validate_across_contexts_tool` adds the AA4 design
  baseline** to SV-AA2, SV-AA*, Cycle 0 and Cycle 1. With an `obs_config`,
  AA4 is reported as `not_validated` (the validator has no AA4 context).
- **`validate_across_contexts_tool(requirement=...)`** gives a verdict per
  context for each stated dimension: observing mode, band, frequency,
  bandwidth, channel width, angular resolution and sensitivity (the strict
  calculator; its beam judges resolution at the same weighting). Overall
  verdicts: `available_from <context>`, `aa4_only`, `not_met`, and
  `mode_not_offered` (VLBI, transient buffer, solar: no schema in any
  context, AA4 included), kept separate from `aa4_only`. `obs_config` is now
  optional.

### Sensitivity calculator: strict and best-fit modes (WS3)

- **Fixed: the calculator API moved to v12.** v11 now returns 404, so every
  `query_sensitivity_calc` and `compare_to_calculator_tool` call was failing.
  v12 takes the same parameters and returns the same shape.
- **Subarray names and labels are interchangeable.** The calculator lists
  each subarray with a name (`MID_inner_r2km_aastar`) and a label
  (`AA* (MID_inner_r2km)`), but the Mid API accepts only labels and the Low
  API only names. Either is now translated, everywhere, and the response's
  `subarray_used` echoes the name, the label and what was sent.
- **`compare_to_calculator_tool(mode="strict")`** uses only the stated
  params plus declared defaults (robust 0 when no weighting is given, the
  AA* array, band centre and default continuum bandwidth for the stated
  band, a −30° pointing) and lists them in `defaults_applied`. It refuses to
  guess a Mid band.
- **`mode="best_fit"` (default, as before)** now returns
  `n_free_parameters`, `free_parameters` and the `strict` result alongside
  the variants. New `sweep` parameter for explicit sweeps (e.g. centre
  frequency). A stated weighting is no longer swept.
- **`array_substituted`.** When the calculator refuses a mixed array for a
  band or window (Band 5 on the 13.5-m dishes; more than 720 MHz in Band 2),
  the 15-m-only subarray at the same stage is used and the swap is reported
  with the calculator's reason. It is never silent.
- New: claims in K (`unit="K"`, converted with the frequency and the stated
  or calculated beam), `sigma_level` (5 for a 5σ limit), `stokes="V"`
  (thermal noise only), `claimed_channel_width_hz` (rescales a per-channel
  figure), and non-integer robust values (interpolated between neighbouring
  integers and labelled `interpolated`).

### keyword_coverage_tool and facility_network_tool (WS2)

- **Default change: abstract and body only.** New `zones` parameter
  (default `["abstract", "body"]`; `null` for every zone). Reference lists,
  affiliations, author lists and acknowledgements no longer count.
- **Default change: plurals and variants match.** Each word matches in
  singular or plural (`survey`/`surveys`, `frequency`/`frequencies`,
  `FRB`/`FRBs`), and words may be joined by a space or a hyphen
  (`zoom mode`/`zoom-mode`). `strict=true` restores literal whole-word
  matching.
- **Default change: curated exclusions.** Other meanings of facility names
  are not counted: Max-Planck, the Planck constant/function/law, Planck
  Collaboration citations; "Chandra et al." and "Chandra (2019)"; the Fermi
  level/energy/acceleration; Gaia-Enceladus; "Euclid's". Turn off with
  `use_curated_exclusions=false` (or `strict=true`); add your own with
  `exclude`.
- New `return_contexts` (k matched snippets per paper, with zone, heading
  and page) and `usage_stats` (mentions per 1,000 body words; whether the
  term is in a heading or the abstract).
- Responses now state the `zones`, the `matching` mode and the
  `exclusions` applied. Same options on `facility_network_tool`.
- To reproduce counts from before this change, pass `zones=null`,
  `strict=true`, `include_noncanonical=true`.

### Index: zones, duplicates, fingerprint (WS1)

- **New chunk metadata.** Every chunk now records its document `zone`
  (`front_matter`, `abstract`, `body`, `acknowledgements`, `references` or
  `appendix`), its document's `doc_status` (`canonical`, `duplicate` or
  `unlisted`) and, for copies, `duplicate_of`. Rules: `rag/zones.py`.
  Existing indexes get them with `python ingest.py --retag` (metadata only,
  no re-parse or re-embedding, about 15 s); new ingests retag automatically.
- **Index manifest.** `chroma_db/manifest.json` records the index
  fingerprint (sha256 over every chunk's path, index and text), document and
  chunk counts, zone and duplicate rule versions, Docling versions, the
  embedding model, chunk size and git commit.
- **Fingerprint on every response.** Every tool result now carries the
  index fingerprint: in the MCP result `_meta.index`, as an `_index` key in
  JSON-object results, and as a closing `[index …]` line in text results.
  JSON-list results (`search_astronomy_docs(as_json=True)`) are unchanged.
- **New `index_health_tool`.** Duplicate paper IDs, copies, unlisted book
  documents, unclassified documents, chapters with no PDF, documents whose
  reference list was not found, and whether the manifest still matches.
- **Default change: copies no longer count.** `keyword_coverage_tool` and
  `facility_network_tool` count each paper once (by file, not by stem) and
  leave out re-downloaded copies (`Curiel01b`, `Curiel01c`, `Oswald01b`, the
  misfiled `Bianchi01` copy) and book documents with no bibliography entry
  (`contents_webpage`, the GEP PDF). Pass `include_noncanonical=true` for
  the old behaviour.
- `list_documents` lists each bibliography entry once (`Bianchi01` was
  listed twice).
- Fixed: `get_document_chunks` raised `NameError` for any document without
  a bibliography entry.
- Fixed: the BM25/corpus caches now also refresh when the manifest changes,
  so a retag (same chunk count, new metadata) is picked up without a restart.
