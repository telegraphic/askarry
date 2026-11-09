# Changelog

Changes to the MCP tools that alter what they return. Each entry says what a
caller has to change and which existing counts move. See
`docs/atlas-tool-changes.md` for what each change means for the AASKAII Atlas.

## Unreleased

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
