"""
Shared, process-cached access to heavy resources used across the pipeline:
the ChromaDB collection, the sentence-transformers embedding model, and the
Ollama client. Also holds small stateless helpers (`chunk_id`,
`discover_files`) and a tiny cache utility (`CountCache`) for in-memory
indexes that must be rebuilt whenever the underlying collection changes size.

Centralising these here means `rag/ingestion.py`, `rag/retrieval.py`, and
`rag/generation.py` all share one construction path per resource instead of
each redefining it slightly differently.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CHUNK_TOKENS,
    OLLAMA_BASE_URL,
)

_collections: dict[str, chromadb.Collection] = {}  # keyed by str(db_dir)
_embedding_model: SentenceTransformer | None = None
_ollama_client: ollama.Client | None = None

_collection_lock = threading.Lock()
_embedding_model_lock = threading.Lock()
_ollama_client_lock = threading.Lock()


def get_chroma_collection(db_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    """Return the process-cached ChromaDB collection stored under *db_dir*
    (default: the main CHROMA_DIR index), creating it if needed. Each store
    directory (e.g. config.TEXTBOOKS_CHROMA_DIR) gets its own cached handle."""
    key = str(db_dir)
    with _collection_lock:
        if key not in _collections:
            client = chromadb.PersistentClient(path=key)
            _collections[key] = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return _collections[key]


def reset_collection(db_dir: Path = CHROMA_DIR) -> None:
    """Clear the cached ChromaDB collection handle for *db_dir*.

    Must be called after the on-disk store is deleted (e.g. during --reindex)
    so the next call to get_chroma_collection() opens a fresh client instead
    of returning a stale handle pointing at the deleted database.
    """
    with _collection_lock:
        _collections.pop(str(db_dir), None)


def get_embedding_model() -> SentenceTransformer:
    """Return the process-cached embedding model, with its max sequence
    length clamped to MAX_CHUNK_TOKENS."""
    global _embedding_model
    with _embedding_model_lock:
        if _embedding_model is None:
            model = SentenceTransformer(EMBEDDING_MODEL)
            model.max_seq_length = min(MAX_CHUNK_TOKENS, model.max_seq_length)
            _embedding_model = model
        return _embedding_model


def get_ollama_client() -> ollama.Client:
    """Return the process-cached Ollama client, created lazily on first use."""
    global _ollama_client
    with _ollama_client_lock:
        if _ollama_client is None:
            _ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        return _ollama_client


def chunk_id(source: str, index: int) -> str:
    """Stable, collision-resistant chunk ID derived from the full file path."""
    raw = f"{source}::{index}"
    return hashlib.sha1(raw.encode()).hexdigest()


def discover_files(dirs: list[Path], suffixes: set[str]) -> list[Path]:
    """Recursively find all files with any of *suffixes* across *dirs*.

    Non-existent directories are silently skipped (callers that need to
    report missing directories to the user do so themselves).
    """
    found: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for suffix in suffixes:
            found.extend(sorted(d.rglob(f"*{suffix}")))
    return found


_T = TypeVar("_T")


class CountCache:
    """Caches an arbitrary value, invalidated whenever `collection.count()`
    changes since the last build.

    Used for in-memory indexes (BM25, acronym lookup) that are expensive to
    construct but must be rebuilt after re-ingestion changes the collection.
    """

    def __init__(self) -> None:
        self._value: _T | None = None
        self._count: int | None = None
        self._lock = threading.Lock()

    def get(self, collection: chromadb.Collection, build_fn: Callable[[], _T]) -> _T:
        current_count = collection.count()
        with self._lock:
            if self._value is None or current_count != self._count:
                self._value = build_fn()
                self._count = current_count
            return self._value
