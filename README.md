# ASKARRY — SKA documentation search for Claude (MCP server)

ASKARRY is an [MCP](https://modelcontextprotocol.io) server that gives Claude searchable access to the [Advancing Astrophysics with the SKA II](https://www.skao.int/en/science-users/aaskaii) (AASKAII) book, its 2015 predecessor and the public SKA Key Capabilities documents. It also bundles SKA observing-setup tools: sensitivity calculator, setup validator, data-volume estimator, SIMBAD/NED lookup, visibility and LST-pressure analysis.

Retrieval runs locally (hybrid semantic + BM25 search, cross-encoder re-ranking) and Claude writes the answers from the passages it gets back. No local LLM is needed.

> This is a research prototype, shared as is, with no support or maintenance guarantee.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10+ and access to the SKAO package index (`ska-ost-array-config` is only on `artefact.skao.int`, see `pyproject.toml`).

**1. Install**

```bash
git clone <repo-url> astronomy-rag && cd astronomy-rag
uv sync
```

**2. Get the data.** The search indexes are too large for git and are shared separately. Unpack them into the repository root so you end up with:

```
astronomy-rag/
├── chroma_db/            main index (AASKAII, AASKA 2015, SKA Key Capabilities)
├── chroma_textbooks/     textbook index
├── sources.db            astronomical sources named in the papers
└── citations.db          references cited by the papers
```

To keep them elsewhere, set `ASKARRY_DATA_DIR` to the folder that holds those four items. The server checks for them on start-up and tells you what is missing.

**3. Connect Claude.** Pick one.

*Claude Code*

```bash
claude mcp add astronomy-rag -- uv run --directory /path/to/astronomy-rag python mcp_server.py
```

*Claude Desktop*: add this to `~/Library/Application Support/Claude/claude_desktop_config.json` and restart the app. Use the absolute path from `which uv`, because Claude Desktop does not see your shell `PATH`.

```json
{
  "mcpServers": {
    "astronomy-rag": {
      "command": "/absolute/path/to/uv",
      "args": ["run", "--directory", "/path/to/astronomy-rag", "python", "mcp_server.py"],
      "env": { "ASKARRY_DATA_DIR": "/path/to/data (optional)" }
    }
  }
}
```

The first search downloads two small models from Hugging Face (`BAAI/bge-large-en-v1.5`, about 1.3 GB, and a cross-encoder re-ranker), so allow a few minutes and an internet connection the first time.

**4. Try it.** Ask Claude: "Use astronomy-rag to find what AASKAII says about HI intensity mapping with SKA-Mid."

## Sharing one server with a team

The server speaks stdio by default. To serve it on the network instead:

```bash
uv run python mcp_server.py --transport http --host 0.0.0.0 --port 8000
```

Clients then connect to `http://<host>:8000/mcp`. For Claude Desktop use [`mcp-remote`](https://www.npmjs.com/package/mcp-remote):

```json
{ "mcpServers": { "askarry": { "command": "npx", "args": ["mcp-remote", "http://<host>:8000/mcp", "--allow-http"] } } }
```

There is no authentication. The server makes outbound requests (Google Scholar, SIMBAD, the SKAO sensitivity calculator) and can write to `sources.db`, so keep it on a trusted network or behind a proxy that adds a token.

## What the server offers

Ask Claude to list the tools, or see `mcp_server.py`. By group:

| Group | Tools |
|-------|-------|
| Document search | `search_astronomy_docs`, `search_ska_capabilities`, `search_textbooks`, `list_documents`, `list_ska_capability_docs`, `list_textbooks`, `get_document_chunks`, `verify_quote_tool`, `verify_citation_tool` |
| Literature | `search_google_scholar_*`, `get_author_info_tool`, `get_paper_info_tool`, `citing_papers_tool`, `top_cited_papers_tool`, `list_missing_papers_tool` |
| Sources and sky | `resolve_object_tool`, `ned_lookup_tool`, `cone_search_tool`, `query_sources_tool`, `get_source_mentions_tool`, `visibility_windows_tool`, `lst_pressure_tool`, `facility_network_tool` |
| Observing setup | `query_sensitivity_calc`, `compare_to_calculator_tool`, `validate_observing_setup_tool`, `validate_across_contexts_tool`, `describe_schema_tool`, `describe_rules_tool`, `get_context_defaults_tool`, `get_subarray_layout_tool`, `estimate_*_tool`, `list_odps_tool` |
| Observing requests | `add_observing_request_tool`, `list_observing_requests_tool`, `review_observing_request_tool`, `add_survey_field_tool`, `list_survey_fields_tool` |
| Acronyms | `list_acronyms_tool`, `lookup_acronym_tool` |

Static documents describe the full-design telescope, not what a given stage supports today. For "what can I configure now" questions the server's own instructions tell Claude to cross-check the live schema and validator tools, so ask for the context you care about (for example `sv_aa2`).

## Building your own index (optional)

Only needed if you are adding documents. Colleagues who received the databases can skip this.

```bash
uv sync --extra ingest              # adds docling
mkdir -p pdfs && cp ~/Downloads/paper.pdf pdfs/
uv run python ingest.py             # index pdfs/ into chroma_db/
uv run python ingest.py --reindex   # wipe and rebuild
uv run python ingest.py --textbooks # textbooks go into chroma_textbooks/
```

The `pdfs/` folder is not in git (it is about 1 GB); `pdfs/download.py` fetches the AASKAII chapters.

## Deprecated: Gradio web UI

`legacy/gradio_app.py` is an older local-only chat UI that answers with an [Ollama](https://ollama.com) model instead of Claude. It is unmaintained and kept for reference. Its dependencies are the optional `gui` extra, so a normal install does not pull them in:

```bash
ollama pull gemma4
uv run --extra gui python -m legacy.gradio_app   # http://localhost:7860
```

<video src="https://github.com/user-attachments/assets/8d68da0e-47ca-4c52-a8a9-8dca620dd92a" controls width="100%"></video>

## Tests

```bash
uv run pytest
```

## Technology

| Component | Technology |
|-----------|------------|
| PDF parsing & chunking | [docling](https://github.com/DS4SD/docling) + `HybridChunker` (ingest extra) |
| Embeddings | `BAAI/bge-large-en-v1.5` (sentence-transformers) |
| Keyword search | BM25 (`rank-bm25`) |
| Vector store | ChromaDB (local, persistent) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Server | FastMCP |

---

## Theory of operation

### Ingestion (`ingest.py` → `rag/ingestion.py`)

```
PDF / HTML files
      │
      ▼
 docling DocumentConverter
 (preserves tables, headings, multi-column layouts)
      │
      ▼
 HybridChunker  ──► chunks ≤ MAX_CHUNK_TOKENS tokens each
 (structure-aware: never splits mid-sentence or mid-table;
  each chunk carries metadata: page number, section heading, caption)
      │
      ▼
 SentenceTransformer.encode()  →  dense vector per chunk
      │
      ▼
 ChromaDB upsert  (idempotent — skips files already indexed)
```

Docling understands PDF document structure (headings, tables, figures, footnotes) and produces a rich intermediate representation. `HybridChunker` then divides that into semantically coherent passages that respect heading boundaries, making each chunk more meaningful than naive character-count splitting.

Each chunk stored in ChromaDB carries:
- The chunk text
- `source` — absolute file path
- `chunk_index` — position within the document
- `page_no` — page number in the source PDF
- `heading` — outermost section heading above this chunk
- `caption` — figure/table caption if the chunk is a figure or table

---

### Retrieval (`rag/retrieval.py`)

Every query passes through a four-stage pipeline:

```
User query
      │
      ├─── (optional) HyDE  ──► LLM generates a short hypothetical answer
      │                          that answer is embedded instead of the question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1a — Semantic search                         │
│  query → BGE embedding → ChromaDB cosine search     │
│  → top RETRIEVAL_CANDIDATES results                 │
├─────────────────────────────────────────────────────┤
│  Stage 1b — BM25 keyword search                     │
│  query tokenised → BM25Okapi.get_scores()           │
│  → top RETRIEVAL_CANDIDATES results                 │
└─────────────────────────────────────────────────────┘
      │
      ▼
  Stage 2 — Reciprocal Rank Fusion (RRF)
  merged_score = 1/(60 + rank_semantic)
               + BM25_WEIGHT × 1/(60 + rank_bm25)
  → deduplicated, fused ranking
      │
      ▼
  Stage 3 — Cross-encoder re-ranking
  Each (query, chunk) pair scored by a cross-encoder model;
  heavier model, sees both texts together → more accurate relevance
  → top TOP_K chunks
      │
      ▼
  Stage 4 — Context window expansion
  For each top chunk, fetch the immediately preceding and
  following chunks (±1) from the same document and join them.
  Gives the LLM a wider passage, reducing mid-sentence truncation.
      │
      ▼
  TOP_K expanded passages
```

**Why two search legs?**
Semantic (vector) search captures meaning — it finds passages about "antenna sensitivity" even if the query says "dish performance". BM25 captures exact token matches — it reliably finds passages containing "SKA-Low", "CSP.LMC", or a specific part number. Technical documentation is full of jargon and identifiers that semantic search can miss; BM25 catches them. RRF combines both lists without needing to normalise their scores onto a common scale.

**Why a cross-encoder re-ranker?**
The first-stage models (BGE + BM25) score each chunk independently of the query. A cross-encoder sees the full (query, passage) pair at once, which allows much more accurate relevance judgement — at the cost of speed. Running it only on `RETRIEVAL_CANDIDATES` results (not the whole corpus) keeps it practical.

**Why HyDE?**
Technical questions are phrased very differently from the passages that answer them ("How do I configure a subarray?" vs. "Subarrays are configured by…"). HyDE bridges that gap by generating a passage in the *answer* style and embedding that instead, shifting the query vector closer to where the answer lives in embedding space.

---

### Generation (`rag/generation.py`)

The top `TOP_K` expanded passages are assembled into a context block and sent to Ollama with a system prompt that instructs the model to:
- Answer only from the provided passages
- Cite which document(s) the answer draws from
- Use correct SKA terminology

The LLM never searches the internet — every answer is grounded in the indexed documents.

---

### Bibliography & Table of Contents (`rag/bibliography.py`)

`pdfs/bibliography.json` is a static lookup of `{chapter_id: {title, authors, section}}` for the AASKAII chapter PDFs, generated offline by `pdfs/build_bibliography.py` (scrapes the live <https://www.skao.int/en/aaskaii> page — re-run manually if new chapters are published). `rag/bibliography.py` matches chunk `source` paths to bibliography entries by filename stem and provides:

- `lookup_citation()` / `format_citation()` — used in the **Source Passages** panel to show "Surname et al (2026) Title" instead of a raw file path.
- `list_toc_entries()` — powers the **Table of Contents** tab in the web UI, grouping chapters by section (in book order) with author lists and direct links to the local PDFs.

Local PDFs with no matching bibliography entry are silently excluded from the Table of Contents by design.

---

## Configuration reference (`rag/config.py`)

### Directories

| Setting | Default | Purpose |
|---------|---------|---------|
| `PDF_DIR` | `pdfs/` | Local folder scanned recursively for PDFs and HTML files |
| `PDF_DIRS` | `[PDF_DIR]` | List of all directories to ingest; edit to add/remove sources |
| `DATA_DIR` | repo root | Where the databases live; override with the `ASKARRY_DATA_DIR` environment variable |
| `CHROMA_DIR` | `chroma_db/` (under `DATA_DIR`) | Where ChromaDB persists the vector index on disk |
| `TEXTBOOKS_DIR` / `TEXTBOOKS_CHROMA_DIR` | `pdfs/textbooks/` / `chroma_textbooks/` | Textbooks (one PDF, or a folder of chapter PDFs, per book) indexed separately via `python ingest.py --textbooks`; never added to the main index |
| `COLLECTION_NAME` | `"astronomy"` | ChromaDB collection name; change this if you want separate indexes for different document sets |
| `BIBLIOGRAPHY_PATH` | `pdfs/bibliography.json` | Title/author/section lookup for AASKAII chapters, used for citations and the Table of Contents tab (see `rag/bibliography.py`) |
| `AASKAII_YEAR` | `2026` | Publication year shown in formatted citations for all AASKAII chapters |

### Embedding model

| Setting | Default | Purpose |
|---------|---------|---------|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Sentence-transformers model used to encode chunks at ingest time and queries at retrieval time. BGE-large outperforms the older MiniLM on technical retrieval benchmarks. **If you change this you must wipe `chroma_db/` and re-ingest** — stored vectors must match the model that queries them. |
| `EMBEDDING_QUERY_PROMPT` | `"Represent this sentence for searching relevant passages: "` | BGE models are trained with an asymmetric scheme: index-time documents are encoded as-is, but queries must be prefixed with this instruction to pull the query vector into the right region of embedding space. Omitting it measurably degrades recall. |

### Re-ranker

| Setting | Default | Purpose |
|---------|---------|---------|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model that re-scores (query, passage) pairs in Stage 3. A cross-encoder is slower than a bi-encoder but significantly more accurate because it attends jointly to query and passage tokens. MS-MARCO was trained on web search relevance; `BAAI/bge-reranker-base` (trained on the same data as the embedding model) is an alternative worth trying. |

### Ollama (LLM, deprecated GUI only)

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_MODEL` | `gemma4` | The local LLM used for answer generation and (when enabled) HyDE hypothesis generation. Must be pulled first: `ollama pull gemma4`. Any Ollama model works — larger models give better answers but are slower. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address. Change this if Ollama is running on a different machine or port. |

### Retrieval tuning

| Setting | Default | Purpose |
|---------|---------|---------|
| `TOP_K` | `5` | How many chunks are ultimately passed to the LLM. Higher values give the LLM more context but slow generation and can dilute the answer with less relevant material. 3–7 is a typical range. |
| `RETRIEVAL_CANDIDATES` | `20` | How many results each of the two first-stage search legs (semantic + BM25) fetches before fusion. A larger pool gives RRF and the cross-encoder more to work with, improving recall at the cost of re-ranking time. |
| `MAX_CHUNK_TOKENS` | `480` | Maximum tokens per chunk at ingest time. BGE-large has a hard 512-token limit (including `[CLS]`/`[SEP]` special tokens); staying at 480 gives headroom for the tokenizer and prevents the model from silently truncating. **Changing this requires a full re-ingest.** |
| `BM25_WEIGHT` | `0.5` | Controls how much the BM25 keyword leg contributes to the RRF score. `0.0` = pure semantic search (ignores BM25); `1.0` = semantic and keyword legs weighted equally. SKA documentation is rich in acronyms and part numbers that benefit from keyword matching, so 0.5 is a reasonable default. Raise toward 1.0 if exact-term queries perform poorly; lower toward 0.0 if unrelated documents with matching keywords keep appearing. |
| `USE_HYDE` | `False` | When `True`, Ollama generates a short hypothetical answer before retrieval; that answer text is embedded instead of the raw question. This shifts the query vector into "answer space" and improves recall for highly technical questions. Adds roughly the latency of one LLM call (~1–3 s) per query. Off by default; enable it if semantic retrieval results feel topically off. |
| `CONFIDENCE_THRESHOLD` | `0.3` | Sigmoid-normalized cross-encoder relevance (0-1) below which the top retrieved chunk is treated as a weak match. Below this, the UI shows a low-confidence warning while the answer streams, and the LLM is given an explicit hedging instruction instead of being left to guess from marginal passages. |

### MCP server (Claude Desktop)

| Setting | Default | Purpose |
|---------|---------|--------|
| `MCP_TOP_K` | `20` | Chunks returned to Claude per `search_astronomy_docs` call. Frontier models have large context windows and can synthesise more passages than a local model, so this is set higher than `TOP_K`. |
| `MCP_RETRIEVAL_CANDIDATES` | `50` | First-stage candidate pool for the MCP path. A larger pool keeps the re-ranker funnel meaningful when `MCP_TOP_K` is high. |

---

## Project structure

```
astronomy-rag/
├── pdfs/                     ← Add your PDFs here
│   ├── bibliography.json     ← Title/author/section lookup (see build_bibliography.py)
│   └── build_bibliography.py ← One-time offline scraper for bibliography.json
├── chroma_db/                ← Auto-created by ingest.py
├── chroma_textbooks/         ← Auto-created by ingest.py --textbooks
├── rag/
│   ├── config.py             ← All tuneable settings
│   ├── ingestion.py          ← docling → HybridChunker → ChromaDB
│   ├── retrieval.py          ← Hybrid BM25 + semantic → RRF → rerank → expand
│   ├── generation.py         ← Ollama prompt & response (Gradio UI path)
│   └── bibliography.py       ← Citation lookup + Table of Contents data
├── ingest.py                 ← Build the index from PDFs (needs the `ingest` extra)
├── mcp_server.py             ← The MCP server: run this
├── legacy/gradio_app.py      ← Deprecated Gradio UI (needs the `gui` extra)
├── pyproject.toml            ← Dependencies (uv)
└── uv.lock
```
