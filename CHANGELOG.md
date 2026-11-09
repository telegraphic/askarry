# Changelog

Changes to the MCP tools that alter what they return. Each entry says what a
caller has to change and which existing counts move. See
`docs/atlas-tool-changes.md` for what each change means for the AASKAII Atlas.

## Unreleased

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
