# SKARRY — SKA RAG Documentation Search

SKARRY is a local Retrieval-Augmented Generation (RAG) system designed to answer questions from the openly-accessible Advancing Astrophysics with the SKA II book. Rather than training a model on the documents, it builds a searchable knowledge base from them and retrieves the most relevant passages at question time.
The workflow has two major phases:

1) Ingestion (run once when documents are added)
2) Querying (run every time a user asks a question)

All computation runs on-device (so it does not transfer data to the cloud).

## Demo

<video src="docs/askarry-screengrab.mp4" controls width="100%"></video>

| Component | Technology |
|-----------|------------|
| PDF parsing & chunking | [docling](https://github.com/DS4SD/docling) + `HybridChunker` |
| Embeddings | `BAAI/bge-large-en-v1.5` (sentence-transformers) |
| Keyword search | BM25 (`rank-bm25`) |
| Vector store | ChromaDB (local, persistent) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | [Ollama](https://ollama.com) (`gemma4` by default) |
| Web UI | Gradio |


## High-level overview

### Ingestion

- **Document discovery and ingestion** The system scans the configured document directories and identifies all PDF and HTML files that should be added to the searchable corpus.

- **Document structure extraction with Docling** Each document is parsed by Docling's `DocumentConverter`, which preserves structural elements such as headings, tables, figures, captions, and multi-column layouts.

- **Structure-aware chunking** The parsed document is passed through Docling's `HybridChunker`, which divides the content into manageable passages while respecting document structure.

- **Metadata enrichment** Each chunk is annotated with metadata including its source document, page number, chunk index, section heading, and any associated figure or table caption.

- **Dense embedding generation** Every chunk is converted into a semantic vector using the `BAAI/bge-large-en-v1.5` embedding model so similar concepts can be matched during retrieval.

- **Persistent vector indexing** The chunk text, metadata, and embeddings are stored in a local ChromaDB collection to create a searchable vector index.

### Querying (Retrieval)

- **Optional HyDE query expansion** When enabled, the local LLM generates a hypothetical answer to the user's question and the system embeds that answer instead of the original query.

- **Semantic vector retrieval** The query embedding is compared against stored chunk embeddings in ChromaDB to retrieve semantically similar passages.

- **Keyword retrieval using BM25** The same query is searched using BM25 to find passages containing important exact matches such as acronyms, identifiers, and technical terminology.

- **Hybrid ranking via Reciprocal Rank Fusion** The semantic and BM25 result lists are combined using Reciprocal Rank Fusion to produce a single ranked set of candidates.

- **Cross-encoder re-ranking** A cross-encoder model jointly evaluates each query and candidate passage pair to produce a more accurate relevance score.

- **Confidence assessment** The top re-ranking score is compared against a confidence threshold to determine whether retrieval quality is strong or weak.

- **Context window expansion** The system retrieves neighbouring chunks around each highly ranked passage to provide additional surrounding context.

- **Context assembly** The highest-ranked expanded passages are combined into a single context block that will be supplied to the language model.

### Answer Generation

- **Grounded answer generation** The context and user query are sent to the local Ollama-hosted LLM, which is instructed to answer only using the retrieved passages and cite its sources.

### Presentation and Citation

- **Citation and bibliography enhancement** Source documents are matched against a local bibliography database to display formatted citations rather than raw file paths.

- **Presentation through the Gradio interface** The generated answer, supporting source passages, citations, and confidence information are displayed in the Gradio web application.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — install from <https://ollama.com/download> then pull a model:
   ```bash
   ollama pull gemma4
   ```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add PDFs to the pdfs/ folder
mkdir -p pdfs
cp ~/Downloads/my_paper.pdf pdfs/
```

> All commands below (`python ingest.py`, `python app.py`, etc.) assume the
> `.venv` is activated — remember to `source .venv/bin/activate` first in
> every new shell.

---

## Usage

### Step 1 — Ingest PDFs

```bash
python ingest.py

# To wipe the existing ChromaDB store and reindex everything from scratch:
python ingest.py --reindex
```

### Step 2 — Launch the web UI

```bash
python app.py
```

Open your browser at <http://localhost:7860>.

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
| `CHROMA_DIR` | `chroma_db/` | Where ChromaDB persists the vector index on disk |
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

### Ollama (LLM)

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

---

## Project structure

```
astronomy-rag/
├── pdfs/                     ← Add your PDFs here
│   ├── bibliography.json     ← Title/author/section lookup (see build_bibliography.py)
│   └── build_bibliography.py ← One-time offline scraper for bibliography.json
├── chroma_db/                ← Auto-created by ingest.py
├── rag/
│   ├── config.py             ← All tuneable settings
│   ├── ingestion.py          ← docling → HybridChunker → ChromaDB
│   ├── retrieval.py          ← Hybrid BM25 + semantic → RRF → rerank → expand
│   ├── generation.py         ← Ollama prompt & response
│   └── bibliography.py       ← Citation lookup + Table of Contents data
├── ingest.py                 ← Run this first
├── app.py                    ← Gradio web UI (Ask a Question / Table of Contents tabs)
└── requirements.txt
```
