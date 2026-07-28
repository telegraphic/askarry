"""
File ingestion pipeline.

Steps:
  1. Discover PDF and HTML files across all configured directories (recursive).
  2. Skip files whose source path is already present in ChromaDB.
  3. Parse + chunk new files in parallel worker processes (docling is the bottleneck).
  4. Embed all chunks in the main process with sentence-transformers.
  5. Upsert into ChromaDB (idempotent — safe to re-run).
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from loguru import logger

from . import store
from .config import (
    ACRONYM_HEADINGS,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    INGEST_WORKERS,
    PDF_DIRS,
    SUPPORTED_SUFFIXES,
)

# HybridChunker/tokenizers count tokens on the *full* section text before
# splitting it down to MAX_CHUNK_TOKENS, so transformers' fast tokenizer logs
# a "Token indices sequence length is longer than the specified maximum..."
# warning for every long section. This is expected and harmless here (the
# chunker still splits correctly) — silence just that logger. Set at module
# level so it also applies inside spawned ProcessPoolExecutor workers, which
# re-import this module fresh.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Worker-process state (populated once per worker by _worker_init)
# ---------------------------------------------------------------------------
_worker_converter: DocumentConverter | None = None
_worker_chunker: HybridChunker | None = None


def _worker_init(chunk_size: int, embedding_model_name: str) -> None:
    """Load heavy resources once per worker process to avoid per-file overhead."""
    global _worker_converter, _worker_chunker
    # Use AutoTokenizer (lighter than full SentenceTransformer) for chunking only
    from transformers import AutoTokenizer  # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    _worker_converter = DocumentConverter()
    _worker_chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=chunk_size)
    )


def _worker_parse_chunk(
    source: str,
) -> tuple[str, list[str], list[dict]] | None:
    """Parse and chunk one file. Called in a worker process."""
    name = Path(source).name
    try:
        conv_result = _worker_converter.convert(source)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not parse {name} — {exc}")
        return None

    chunks = list(_worker_chunker.chunk(dl_doc=conv_result.document))
    if not chunks:
        logger.warning(f"No chunks extracted from {name}")
        return None

    texts = [c.text for c in chunks]
    metadatas = [_chunk_metadata(source, i, c) for i, c in enumerate(chunks)]
    logger.info(f"Parsed {len(texts)} chunks from {name}")
    return source, texts, metadatas


def _chunk_metadata(source: str, index: int, chunk) -> dict:
    """Extract rich metadata from a docling DocChunk for storage in ChromaDB.

    ChromaDB only accepts str/int/float/bool values — no lists or None.
    """
    meta = chunk.meta

    # Page number: take the first page from the first doc_item's provenance
    page_no: int = 0
    for item in meta.doc_items:
        if item.prov:
            page_no = item.prov[0].page_no
            break

    # Section heading: outermost heading above this chunk
    heading = ""
    if meta.headings:
        heading = meta.headings[0]

    # Caption (for figures/tables)
    caption = ""
    if meta.captions:
        caption = meta.captions[0]

    # Tag acronym/glossary sections so retrieval can use them for query expansion
    heading_lower = heading.lower()
    chunk_type = "text"
    for marker in ACRONYM_HEADINGS:
        if marker in heading_lower:
            chunk_type = "acronyms"
            break

    return {
        "source": source,
        "chunk_index": index,
        "page_no": page_no,
        "heading": heading,
        "caption": caption,
        "chunk_type": chunk_type,
    }


def ingest_files(dirs: list[Path] | None = None, reset: bool = False) -> dict[str, int]:
    """
    Ingest all supported files found (recursively) in *dirs* into ChromaDB.

    Files whose resolved path is already recorded as a source in ChromaDB are
    skipped, so re-running only processes new files.

    If *reset* is True, the existing ChromaDB store at CHROMA_DIR is deleted
    first, so every file is parsed and indexed from scratch.

    Returns a dict mapping file path string → number of chunks indexed.
    """
    dirs = dirs if dirs is not None else PDF_DIRS

    if reset and CHROMA_DIR.exists():
        logger.info(f"Reindex: removing existing ChromaDB store at '{CHROMA_DIR}'")
        shutil.rmtree(CHROMA_DIR)

    # Embedding model lives in the main process only
    model = store.get_embedding_model()
    chunk_size = model.max_seq_length

    collection = store.get_chroma_collection()

    existing_metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    indexed_sources: set[str] = {m["source"] for m in existing_metadatas}

    all_files = store.discover_files(dirs, SUPPORTED_SUFFIXES)
    if not all_files:
        return {}

    new_sources: list[str] = []
    for fp in all_files:
        source = str(fp.resolve())
        if source in indexed_sources:
            logger.info(f"Skipping {fp.name} (already indexed)")
        else:
            new_sources.append(source)

    if not new_sources:
        return {}

    # --- Parse + chunk in parallel worker processes ---
    n_workers = min(INGEST_WORKERS, len(new_sources))
    logger.info(f"Parsing {len(new_sources)} file(s) with {n_workers} worker(s)…")

    parsed: list[tuple[str, list[str], list[dict]]] = []

    if n_workers <= 1:
        # Avoid subprocess overhead when only one worker is needed
        _worker_init(chunk_size, EMBEDDING_MODEL)
        for source in new_sources:
            result = _worker_parse_chunk(source)
            if result:
                parsed.append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(chunk_size, EMBEDDING_MODEL),
        ) as executor:
            futures = {
                executor.submit(_worker_parse_chunk, source): source
                for source in new_sources
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    parsed.append(result)

    # --- Embed + upsert in the main process ---
    results: dict[str, int] = {}
    for source, texts, metadatas in parsed:
        name = Path(source).name
        logger.info(f"Embedding {len(texts)} chunks from {name}…")
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        ids = [store.chunk_id(source, m["chunk_index"]) for m in metadatas]
        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(texts)} chunks from {name}")
        results[source] = len(texts)

    return results
