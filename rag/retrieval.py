"""
Hybrid retrieval from ChromaDB: semantic vector search + BM25 keyword search,
fused with Reciprocal Rank Fusion (RRF), then cross-encoder re-ranked.

Optionally uses HyDE (Hypothetical Document Embeddings) to improve recall on
technical queries by embedding a generated answer rather than the raw question.

Module-level caches mean models load only once per process.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import chromadb
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import store
from .config import (
    BM25_WEIGHT,
    CHROMA_DIR,
    CONTEXT_EXPANSION_WINDOW,
    DOC_SOURCE_AASKAII,
    DOC_SOURCE_PRIORITY,
    EMBEDDING_QUERY_PROMPT,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATES,
    SEED_ACRONYMS,
    TOP_K,
    USE_HYDE,
)

# Module-level cache — populated on first call to retrieve()
_reranker: CrossEncoder | None = None
# BM25 index and acronym lookup, both rebuilt whenever the collection size
# changes (e.g. after ingest); see store.CountCache. One cache per ChromaDB
# collection (keyed by collection.id) so the main and textbook indexes never share.
_bm25_caches: defaultdict[str, store.CountCache] = defaultdict(store.CountCache)
_acronyms_caches: defaultdict[str, store.CountCache] = defaultdict(store.CountCache)

# Matches lines like "CSP   Central Signal Processor" or "CSP: Central Signal Processor"
_ACRONYM_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9\-\.]{1,15})\s*[:\-–]?\s{1,}([A-Za-z][^\n]{4,80})\s*$",
    re.MULTILINE,
)


def _get_resources(
    db_dir: Path = CHROMA_DIR,
) -> tuple[SentenceTransformer, CrossEncoder, chromadb.Collection]:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return store.get_embedding_model(), _reranker, store.get_chroma_collection(db_dir)


def warmup() -> None:
    """Pre-load all models and the ChromaDB collection into memory.

    Call this once at application startup (e.g. on a background thread) so
    that the first user query does not pay the cold-start penalty.
    """
    _get_resources()


def _get_acronyms(collection: chromadb.Collection) -> dict[str, str]:
    """Build (or return cached) acronym→expansion dict from tagged chunks."""

    def _build() -> dict[str, str]:
        acronyms: dict[str, str] = {}
        try:
            results = collection.get(
                where={"chunk_type": "acronyms"},
                include=["documents"],
            )
        except Exception:
            # ChromaDB raises if no metadata filter matches — that's fine
            results = None

        for text in (results or {}).get("documents") or []:
            for match in _ACRONYM_LINE_RE.finditer(text):
                acronym = match.group(1).strip()
                expansion = match.group(2).strip()
                if acronym not in acronyms:   # first definition wins
                    acronyms[acronym] = expansion

        n_from_docs = len(acronyms)
        for acronym, expansion in SEED_ACRONYMS.items():
            acronyms.setdefault(acronym, expansion)   # documents win on conflicts

        logger.info(
            f"Loaded {len(acronyms)} acronyms "
            f"({n_from_docs} from indexed documents, "
            f"{len(acronyms) - n_from_docs} from seed list)."
        )
        return acronyms

    return _acronyms_caches[str(collection.id)].get(collection, _build)


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
    logger.info(f"Query expanded: {expanded}")
    return expanded


def _get_bm25(collection: chromadb.Collection) -> tuple[BM25Okapi, list[dict]]:
    """Return a BM25 index over all documents in the collection.

    The index is rebuilt lazily when the collection size changes (e.g. after
    ingest). For large collections this is slightly slow on first call but
    subsequent queries are fast.
    """

    def _build() -> tuple[BM25Okapi, list[dict]]:
        logger.info(f"Building BM25 index over {collection.count()} documents…")
        # Fetch everything — ChromaDB stores all text so this is in-memory.
        all_results = collection.get(include=["documents", "metadatas"])
        bm25_docs = [
            {"text": doc, "meta": meta}
            for doc, meta in zip(all_results["documents"], all_results["metadatas"])
        ]
        tokenised = [doc["text"].lower().split() for doc in bm25_docs]
        return BM25Okapi(tokenised), bm25_docs

    return _bm25_caches[str(collection.id)].get(collection, _build)


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
        response = store.get_ollama_client().chat(
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
        logger.warning(f"HyDE generation failed ({exc}); falling back to raw query.")
        return query


def _sigmoid(x: float) -> float:
    """Squash an unbounded cross-encoder logit to a 0-1 confidence score."""
    return 1.0 / (1.0 + math.exp(-x))


def _meta_to_chunk(meta: dict, text: str, score: float) -> dict:
    """Build the common chunk dict shape from a ChromaDB metadata record.

    Shared by the semantic-search and BM25-search hit loops (and implicitly
    matched by `_expand_context`) so both legs of hybrid retrieval produce
    identically-shaped dicts before RRF fusion merges them.
    """
    return {
        "text": text,
        "source": meta.get("source", "unknown"),
        "chunk_index": meta.get("chunk_index", 0),
        "page_no": meta.get("page_no", 0),
        "heading": meta.get("heading", ""),
        "section_path": meta.get("section_path", ""),
        "caption": meta.get("caption", ""),
        "doc_title": meta.get("doc_title", ""),
        "doc_source": meta.get("doc_source", DOC_SOURCE_AASKAII),
        "score": score,
    }


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
        return store.chunk_id(chunk["source"], chunk["chunk_index"])

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


def retrieve(
    query: str,
    top_k: int | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    use_hyde: bool | None = None,
    expansion_window: int | None = None,
    where: dict | None = None,
    db_dir: Path = CHROMA_DIR,
) -> list[dict]:
    """Return the *top_k* most relevant chunks for *query*.

    Pipeline:
      1. Optional HyDE: generate a hypothetical answer and embed that instead.
      2. Semantic search: embed query → ChromaDB vector search.
      3. BM25 keyword search over the same corpus.
      4. Reciprocal Rank Fusion to merge both ranked lists.
      5. Cross-encoder re-rank the fused candidates.
      6. Context window expansion: fetch neighbouring chunks for richer context.

    If *on_progress* is given, it is called as `on_progress(fraction, desc)`
    (fraction in 0-1) at each stage boundary so a caller (e.g. the Gradio UI)
    can drive a real progress bar instead of an opaque "searching" spinner.

    *use_hyde* overrides the ``USE_HYDE`` config flag for this call. Pass
    ``False`` from the MCP server to avoid requiring Ollama to be running.
    *None* means use the config default.

    *expansion_window* overrides ``CONTEXT_EXPANSION_WINDOW`` for this call.
    The MCP server passes ``2`` to give frontier models a wider passage context.

    *where* is an optional ChromaDB metadata filter (e.g.
    ``{"doc_source": "ska_capabilities"}``) scoping both the semantic and
    BM25 search legs to matching chunks only. ``None`` searches everything.

    *db_dir* selects the ChromaDB store to search (default: the main index;
    pass ``config.TEXTBOOKS_CHROMA_DIR`` for the separate textbook index).

    Each returned dict has at minimum:
        text         : str   — expanded passage text
        source       : str   — originating file path
        chunk_index  : int
        page_no      : int
        heading      : str   — outermost section heading
        section_path : str   — full heading hierarchy (" > "-joined)
        caption      : str
        doc_title    : str   — paper/document title from bibliography
        score        : float — cross-encoder relevance, sigmoid-normalised to
                       0-1 (higher = more relevant)
    """
    top_k = top_k if top_k is not None else TOP_K
    _use_hyde = USE_HYDE if use_hyde is None else use_hyde
    _expansion_window = CONTEXT_EXPANSION_WINDOW if expansion_window is None else expansion_window
    model, reranker, collection = _get_resources(db_dir)

    if collection.count() == 0:
        flag = "" if db_dir == CHROMA_DIR else " --textbooks"
        raise RuntimeError(
            f"The vector store at {db_dir} is empty. Run `python ingest.py{flag}` first."
        )

    # --- Optional HyDE ---
    if _use_hyde and on_progress:
        on_progress(0.05, "Generating hypothetical answer (HyDE)…")
    embed_text = _hyde_query(query) if _use_hyde else query
    if _use_hyde:
        logger.info(f"HyDE hypothesis: {embed_text[:120]}…")

    # --- Stage 1a: Semantic vector search ---
    if on_progress:
        on_progress(0.3, "Embedding query & searching vector index…")
    n_candidates = min(RETRIEVAL_CANDIDATES, collection.count())
    query_embedding = model.encode(
        [embed_text], prompt=EMBEDDING_QUERY_PROMPT
    ).tolist()
    vec_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    semantic_hits = []
    for doc, meta, dist in zip(
        vec_results["documents"][0],
        vec_results["metadatas"][0],
        vec_results["distances"][0],
    ):
        score = max(0.0, 1.0 - dist / 2.0)
        semantic_hits.append(_meta_to_chunk(meta, doc, round(score, 4)))

    # --- Stage 1b: BM25 keyword search (with acronym expansion) ---
    if on_progress:
        on_progress(0.5, "Running keyword (BM25) search…")
    bm25, bm25_docs = _get_bm25(collection)
    acronyms = _get_acronyms(collection)
    expanded_query = _expand_query(query, acronyms)
    tokenised_query = expanded_query.lower().split()
    bm25_scores = bm25.get_scores(tokenised_query)
    candidate_indices = range(len(bm25_scores))
    if where:
        candidate_indices = [
            i
            for i in candidate_indices
            if all(bm25_docs[i]["meta"].get(k) == v for k, v in where.items())
        ]
    top_bm25_indices = sorted(
        candidate_indices, key=lambda i: bm25_scores[i], reverse=True
    )[:n_candidates]
    bm25_hits = []
    for i in top_bm25_indices:
        meta = bm25_docs[i]["meta"]
        bm25_hits.append(
            _meta_to_chunk(meta, bm25_docs[i]["text"], float(bm25_scores[i]))
        )

    # --- Stage 2: Reciprocal Rank Fusion ---
    if on_progress:
        on_progress(0.6, "Fusing search results…")
    fused = _rrf_fuse(semantic_hits, bm25_hits, bm25_weight=BM25_WEIGHT)

    if not fused:
        return []

    # --- Stage 3: Cross-encoder re-rank ---
    if on_progress:
        on_progress(0.85, "Re-ranking candidates…")
    # Re-rank only as many candidates as make sense (cap to keep latency low)
    rerank_pool = fused[: min(RETRIEVAL_CANDIDATES, len(fused))]
    pairs = [[query, c["text"]] for c in rerank_pool]
    ce_scores = reranker.predict(pairs)
    # The raw cross-encoder logit isn't a meaningful score to show a user (it's
    # unbounded and can be negative), and it previously wasn't even used as the
    # displayed "relevance" — the stale RRF fusion score was shown instead,
    # which doesn't track the final (cross-encoder) ranking order at all.
    # Squash to a 0-1 confidence via sigmoid so the UI and the low-confidence
    # check below both reflect what actually determined the ranking.
    # Apply the per-doc_source priority multiplier (DOC_SOURCE_PRIORITY) here,
    # after relevance is established but before the final ranking, so an
    # older/superseded source needs a genuinely stronger match to outrank an
    # equally-relevant newer one, rather than always winning or losing.
    confidences = [
        _sigmoid(float(s)) * DOC_SOURCE_PRIORITY.get(chunk["doc_source"], 1.0)
        for s, chunk in zip(ce_scores, rerank_pool)
    ]
    ranked = sorted(
        zip(confidences, rerank_pool), key=lambda x: x[0], reverse=True
    )
    top = [
        {**chunk, "score": round(confidence, 4)}
        for confidence, chunk in ranked[:top_k]
    ]

    # --- Stage 4: Context window expansion ---
    if on_progress:
        on_progress(1.0, "Expanding context…")
    return _expand_context(top, collection, window=_expansion_window)


def _expand_context(
    chunks: list[dict],
    collection: chromadb.Collection,
    window: int = 1,
) -> list[dict]:
    """Fetch neighbouring chunks (±window) from the same source and join them
    with the matched chunk so the LLM receives a wider passage.
    window=1 fetches ±1 (three raw chunks); window=2 fetches ±2 (five), etc."""
    expanded = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        source = chunk["source"]
        idx = chunk["chunk_index"]

        neighbour_ids = [
            store.chunk_id(source, i)
            for i in range(idx - window, idx + window + 1)
            if i >= 0
        ]

        try:
            result = collection.get(
                ids=neighbour_ids,
                include=["documents", "metadatas"],
            )
        except Exception:
            expanded.append(chunk)
            seen_ids.add(store.chunk_id(source, idx))
            continue

        neighbour_pairs = sorted(
            zip(result["documents"], result["metadatas"]),
            key=lambda x: x[1].get("chunk_index", 0),
        )
        combined_text = "\n\n".join(doc for doc, _ in neighbour_pairs if doc)

        cid = store.chunk_id(source, idx)
        if cid not in seen_ids:
            seen_ids.add(cid)
            expanded.append({**chunk, "text": combined_text})

    return expanded
