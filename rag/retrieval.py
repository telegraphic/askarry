"""
Hybrid retrieval from ChromaDB: semantic vector search + BM25 keyword search,
fused with Reciprocal Rank Fusion (RRF), then cross-encoder re-ranked.

Optionally uses HyDE (Hypothetical Document Embeddings) to improve recall on
technical queries by embedding a generated answer rather than the raw question.

Module-level caches mean models load only once per process.
"""

from __future__ import annotations

import hashlib
import logging
import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (
    BM25_WEIGHT,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATES,
    TOP_K,
    USE_HYDE,
)

log = logging.getLogger(__name__)

_ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

# Module-level cache — populated on first call to retrieve()
_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_collection: chromadb.Collection | None = None
# BM25 index rebuilt whenever the collection changes size
_bm25: BM25Okapi | None = None
_bm25_docs: list[dict] | None = None   # parallel list of {text, meta} dicts
_bm25_count: int = 0                   # collection.count() at last BM25 build
# Acronym lookup: {"CSP": "Central Signal Processor", ...} built from tagged chunks
_acronyms: dict[str, str] | None = None
_acronyms_count: int = 0

# Matches lines like "CSP   Central Signal Processor" or "CSP: Central Signal Processor"
_ACRONYM_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9\-\.]{1,15})\s*[:\-–]?\s{1,}([A-Za-z][^\n]{4,80})\s*$",
    re.MULTILINE,
)


def _get_resources() -> tuple[SentenceTransformer, CrossEncoder, chromadb.Collection]:
    global _model, _reranker, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _model, _reranker, _collection


def warmup() -> None:
    """Pre-load all models and the ChromaDB collection into memory.

    Call this once at application startup (e.g. on a background thread) so
    that the first user query does not pay the cold-start penalty.
    """
    _get_resources()


def _get_acronyms(collection: chromadb.Collection) -> dict[str, str]:
    """Build (or return cached) acronym→expansion dict from tagged chunks."""
    global _acronyms, _acronyms_count
    current_count = collection.count()
    if _acronyms is not None and current_count == _acronyms_count:
        return _acronyms

    _acronyms = {}
    _acronyms_count = current_count
    try:
        results = collection.get(
            where={"chunk_type": "acronyms"},
            include=["documents"],
        )
    except Exception:
        # ChromaDB raises if no metadata filter matches — that's fine
        return _acronyms

    for text in results.get("documents") or []:
        for match in _ACRONYM_LINE_RE.finditer(text):
            acronym = match.group(1).strip()
            expansion = match.group(2).strip()
            if acronym not in _acronyms:   # first definition wins
                _acronyms[acronym] = expansion

    log.info("Loaded %d acronyms from indexed documents.", len(_acronyms))
    return _acronyms


def _expand_query(query: str, acronyms: dict[str, str]) -> str:
    """Append full forms of recognised abbreviations to the BM25 query.

    "What does the CSP do?" → "What does the CSP do? CSP Central Signal Processor"

    The original query is preserved so semantic search is unaffected.
    """
    if not acronyms:
        return query
    found = re.findall(r"\b([A-Z][A-Z0-9\-\.]{1,15})\b", query)
    expansions = [
        f"{token} {acronyms[token]}"
        for token in dict.fromkeys(found)   # deduplicate, preserve order
        if token in acronyms
    ]
    if not expansions:
        return query
    expanded = query + " " + " ".join(expansions)
    log.info("Query expanded: %s", expanded)
    return expanded


def _get_bm25(collection: chromadb.Collection) -> tuple[BM25Okapi, list[dict]]:
    """Return a BM25 index over all documents in the collection.

    The index is rebuilt lazily when the collection size changes (e.g. after
    ingest). For large collections this is slightly slow on first call but
    subsequent queries are fast.
    """
    global _bm25, _bm25_docs, _bm25_count
    current_count = collection.count()
    if _bm25 is None or current_count != _bm25_count:
        log.info("Building BM25 index over %d documents…", current_count)
        # Fetch everything — ChromaDB stores all text so this is in-memory.
        all_results = collection.get(include=["documents", "metadatas"])
        _bm25_docs = [
            {"text": doc, "meta": meta}
            for doc, meta in zip(all_results["documents"], all_results["metadatas"])
        ]
        tokenised = [doc["text"].lower().split() for doc in _bm25_docs]
        _bm25 = BM25Okapi(tokenised)
        _bm25_count = current_count
    return _bm25, _bm25_docs


def _hyde_query(query: str) -> str:
    """Generate a short hypothetical document passage for the query.

    HyDE (Hypothetical Document Embeddings): embed the *answer* space instead
    of the question space to reduce the query–document distribution gap for
    technical retrieval.
    """
    prompt = (
        "Write a concise technical paragraph (3-5 sentences) that would "
        "directly answer the following question as if it appeared in SKA "
        "documentation. Do not include the question itself.\n\n"
        f"Question: {query}"
    )
    try:
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 150,
                "temperature": 0.2,
                "top_p": OLLAMA_TOP_P,
                "top_k": OLLAMA_TOP_K,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        log.warning("HyDE generation failed (%s); falling back to raw query.", exc)
        return query


def _rrf_fuse(
    semantic_hits: list[dict],
    bm25_hits: list[dict],
    bm25_weight: float,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion of semantic and BM25 result lists.

    RRF score = Σ weight_i / (k + rank_i)
    The semantic leg always has weight 1.0; the BM25 leg has *bm25_weight*.
    Returns a deduplicated list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    # Map chunk id → chunk dict so we can reconstruct the result list
    chunks_by_id: dict[str, dict] = {}

    def _id(chunk: dict) -> str:
        return _chunk_id(chunk["source"], chunk["chunk_index"])

    # Semantic leg (weight = 1.0)
    for rank, chunk in enumerate(semantic_hits, start=1):
        cid = _id(chunk)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunks_by_id[cid] = chunk

    # BM25 leg (weight = bm25_weight)
    for rank, chunk in enumerate(bm25_hits, start=1):
        cid = _id(chunk)
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + rank)
        if cid not in chunks_by_id:
            chunks_by_id[cid] = chunk

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in ranked_ids:
        chunk = dict(chunks_by_id[cid])
        chunk["score"] = round(scores[cid], 6)
        result.append(chunk)
    return result


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the *top_k* most relevant chunks for *query*.

    Pipeline:
      1. Optional HyDE: generate a hypothetical answer and embed that instead.
      2. Semantic search: embed query → ChromaDB vector search.
      3. BM25 keyword search over the same corpus.
      4. Reciprocal Rank Fusion to merge both ranked lists.
      5. Cross-encoder re-rank the fused candidates.
      6. Context window expansion: fetch neighbouring chunks for richer context.

    Each returned dict has at minimum:
        text        : str   — expanded passage text
        source      : str   — originating file path
        chunk_index : int
        page_no     : int
        heading     : str
        caption     : str
        score       : float — RRF score
    """
    model, reranker, collection = _get_resources()

    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python ingest.py` first."
        )

    # --- Optional HyDE ---
    embed_text = _hyde_query(query) if USE_HYDE else query
    if USE_HYDE:
        log.info("HyDE hypothesis: %s…", embed_text[:120])

    # --- Stage 1a: Semantic vector search ---
    n_candidates = min(RETRIEVAL_CANDIDATES, collection.count())
    query_embedding = model.encode(
        [embed_text], prompt=EMBEDDING_QUERY_PROMPT
    ).tolist()
    vec_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits = []
    for doc, meta, dist in zip(
        vec_results["documents"][0],
        vec_results["metadatas"][0],
        vec_results["distances"][0],
    ):
        score = max(0.0, 1.0 - dist / 2.0)
        semantic_hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "page_no": meta.get("page_no", 0),
            "heading": meta.get("heading", ""),
            "caption": meta.get("caption", ""),
            "score": round(score, 4),
        })

    # --- Stage 1b: BM25 keyword search (with acronym expansion) ---
    bm25, bm25_docs = _get_bm25(collection)
    acronyms = _get_acronyms(collection)
    expanded_query = _expand_query(query, acronyms)
    tokenised_query = expanded_query.lower().split()
    bm25_scores = bm25.get_scores(tokenised_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:n_candidates]
    bm25_hits = []
    for i in top_bm25_indices:
        meta = bm25_docs[i]["meta"]
        bm25_hits.append({
            "text": bm25_docs[i]["text"],
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "page_no": meta.get("page_no", 0),
            "heading": meta.get("heading", ""),
            "caption": meta.get("caption", ""),
            "score": float(bm25_scores[i]),
        })

    # --- Stage 2: Reciprocal Rank Fusion ---
    fused = _rrf_fuse(semantic_hits, bm25_hits, bm25_weight=BM25_WEIGHT)

    if not fused:
        return []

    # --- Stage 3: Cross-encoder re-rank ---
    # Re-rank only as many candidates as make sense (cap to keep latency low)
    rerank_pool = fused[: min(RETRIEVAL_CANDIDATES, len(fused))]
    pairs = [[query, c["text"]] for c in rerank_pool]
    ce_scores = reranker.predict(pairs)
    ranked = sorted(zip(ce_scores, rerank_pool), key=lambda x: x[0], reverse=True)
    top = [chunk for _, chunk in ranked[:top_k]]

    # --- Stage 4: Context window expansion ---
    return _expand_context(top, collection)


def _expand_context(chunks: list[dict], collection: chromadb.Collection) -> list[dict]:
    """Fetch the immediately neighbouring chunks (±1) from the same source and
    join them with the matched chunk so the LLM receives a wider passage."""
    expanded = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        source = chunk["source"]
        idx = chunk["chunk_index"]

        neighbour_ids = [
            _chunk_id(source, i)
            for i in [idx - 1, idx, idx + 1]
            if i >= 0
        ]

        try:
            result = collection.get(
                ids=neighbour_ids,
                include=["documents", "metadatas"],
            )
        except Exception:
            expanded.append(chunk)
            seen_ids.add(_chunk_id(source, idx))
            continue

        neighbour_pairs = sorted(
            zip(result["documents"], result["metadatas"]),
            key=lambda x: x[1].get("chunk_index", 0),
        )
        combined_text = "\n\n".join(doc for doc, _ in neighbour_pairs if doc)

        cid = _chunk_id(source, idx)
        if cid not in seen_ids:
            seen_ids.add(cid)
            expanded.append({**chunk, "text": combined_text})

    return expanded


def _chunk_id(source: str, index: int) -> str:
    """Mirror of ingestion._chunk_id — must stay in sync."""
    raw = f"{source}::{index}"
    return hashlib.sha1(raw.encode()).hexdigest()
