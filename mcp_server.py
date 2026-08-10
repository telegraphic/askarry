"""
MCP server exposing the astronomy-RAG retrieval pipeline to Claude Desktop
(and any other MCP-capable frontier LLM client).

Tools
-----
search_astronomy_docs   Semantic + BM25 hybrid search, re-ranked; returns
                        passages with title, section hierarchy, page, relevance.
list_documents          Table of contents grouped by section.
get_document_chunks     All indexed chunks from a named paper in reading order.

Usage
-----
Run directly for local testing:

    source .venv/bin/activate
    python mcp_server.py

Register in Claude Desktop's config (~/Library/Application Support/Claude/claude_desktop_config.json):

    {
      "mcpServers": {
        "astronomy-rag": {
          "command": "/Users/daniel.price/Data/astronomy-rag/.venv/bin/python",
          "args":    ["/Users/daniel.price/Data/astronomy-rag/mcp_server.py"]
        }
      }
    }

Notes
-----
- Models are loaded lazily on first tool call; no Ollama connection is required
  (use_hyde=False is always passed so HyDE is skipped even if USE_HYDE=True
  in config.py).
- Re-ingest with `python ingest.py --reindex` after pulling new PDFs so that
  doc_title and section_path metadata are populated for all chunks.
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from rag import store
from rag.bibliography import format_citation, list_toc_entries, lookup_citation
from rag.config import MCP_RETRIEVAL_CANDIDATES, MCP_TOP_K
from rag.retrieval import retrieve

mcp = FastMCP(
    "astronomy-rag",
    instructions=(
        "This server provides access to an indexed collection of SKA (Square "
        "Kilometre Array) astronomy papers from the AASKAII compendium. Use "
        "search_astronomy_docs to find relevant passages for a question, "
        "list_documents to discover what papers are available, and "
        "get_document_chunks to read a specific paper in full."
    ),
)


def _format_passage(i: int, chunk: dict) -> str:
    """Render one retrieved chunk as a readable block for the LLM."""
    # Prefer stored doc_title; fall back to bibliography lookup; then stem.
    doc_title = chunk.get("doc_title", "")
    if not doc_title:
        entry = lookup_citation(chunk["source"])
        if entry:
            doc_title = entry["title"]
        else:
            doc_title = Path(chunk["source"]).stem

    # Prefer full section hierarchy; fall back to outermost heading.
    section = chunk.get("section_path") or chunk.get("heading", "")
    page = chunk.get("page_no", 0)
    score = chunk.get("score", 0.0)
    caption = chunk.get("caption", "")

    lines = [f"[{i}] {doc_title}"]
    if section:
        lines.append(f"    Section: {section}")
    if page:
        lines.append(f"    Page: {page}")
    lines.append(f"    Relevance: {score:.0%}")
    if caption:
        lines.append(f"    Caption: {caption}")
    lines.append(f"    {chunk['text']}")
    return "\n".join(lines)


@mcp.tool()
def search_astronomy_docs(query: str, top_k: int = MCP_TOP_K) -> str:
    """Search the SKA astronomy document database for passages relevant to a query.

    Uses a hybrid pipeline: semantic vector search + BM25 keyword search,
    fused with Reciprocal Rank Fusion, then cross-encoder re-ranked.
    Returns up to *top_k* passages (default 20) with source title, section
    hierarchy, page number, and relevance score.

    Use this to answer questions about SKA science, engineering, and methods.
    Cite passages inline with [n] markers matching the returned passage numbers.

    Args:
        query:  The question or topic to search for.
        top_k:  Maximum number of passages to return (default 20).
    """
    chunks = retrieve(
        query,
        top_k=top_k,
        use_hyde=False,          # never require Ollama in the MCP path
        expansion_window=2,      # wider context per passage for frontier LLMs
    )
    if not chunks:
        return "No relevant passages found for that query."

    parts = [_format_passage(i, c) for i, c in enumerate(chunks, 1)]
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_documents() -> str:
    """List all astronomy papers in the database, grouped by AASKAII section.

    Use this before searching to discover what documents are available,
    or to answer questions like "what papers cover gravitational waves?".
    """
    toc = list_toc_entries()
    if not toc:
        return (
            "No documents found. Run `python ingest.py` in the project directory "
            "to index the PDF collection first."
        )

    lines = []
    for section, entries in toc.items():
        lines.append(f"## {section} ({len(entries)} papers)")
        for entry in entries:
            authors = entry.get("authors", [])
            if authors:
                surname = authors[0].split()[-1]
                author_str = f" — {surname} et al." if len(authors) > 1 else f" — {surname}"
            else:
                author_str = ""
            lines.append(f"- {entry['title']}{author_str}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_document_chunks(filename: str, max_chunks: int = 60) -> str:
    """Retrieve all indexed chunks from a specific document in reading order.

    Useful for summarising or deeply analysing a single paper. The document
    is identified by filename stem (e.g. 'ska_mid_baseline_design') or full
    filename including extension. Use list_documents() first to find names.

    Returns up to *max_chunks* consecutive passages (default 60). If the
    document has more chunks they are noted at the end.

    Args:
        filename:   Filename stem or full filename of the document.
        max_chunks: Maximum number of chunks to return (default 60).
    """
    collection = store.get_chroma_collection()
    if collection.count() == 0:
        return "The vector store is empty. Run `python ingest.py` first."

    stem = Path(filename).stem  # normalise — strip extension if present

    # Step 1: fetch only metadatas to find the canonical source path (fast).
    all_meta = collection.get(include=["metadatas"])
    matching_source: str | None = None
    for meta in all_meta["metadatas"]:
        if Path(meta.get("source", "")).stem == stem:
            matching_source = meta["source"]
            break

    if matching_source is None:
        return (
            f"No chunks found for '{filename}'. "
            "Use list_documents() to see available document names, "
            "or check that ingestion has been run."
        )

    # Step 2: fetch all chunks for that source, ordered for reading.
    results = collection.get(
        where={"source": matching_source},
        include=["documents", "metadatas"],
    )
    pairs = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda x: x[1].get("chunk_index", 0),
    )
    total = len(pairs)

    # Build header from bibliography.
    entry = lookup_citation(matching_source)
    if entry:
        doc_title = entry["title"]
        authors = entry.get("authors", [])
        if authors:
            surname = authors[0].split()[-1]
            author_str = f" — {surname} et al." if len(authors) > 1 else f" — {surname}"
        else:
            author_str = ""
        citation = format_citation(entry)
    else:
        doc_title = stem
        author_str = ""
        citation = stem

    lines = [
        f"# {doc_title}{author_str}",
        f"Citation: {citation}",
        f"Total indexed chunks: {total}",
        "",
    ]

    for doc, meta in pairs[:max_chunks]:
        section = meta.get("section_path") or meta.get("heading", "")
        page = meta.get("page_no", 0)
        caption = meta.get("caption", "")

        header_parts = []
        if section:
            header_parts.append(f"§ {section}")
        if page:
            header_parts.append(f"p.{page}")
        if caption:
            header_parts.append(f"[{caption}]")

        if header_parts:
            lines.append(f"[{' | '.join(header_parts)}]")
        lines.append(doc)
        lines.append("")

    if total > max_chunks:
        lines.append(
            f"… {total - max_chunks} more chunks not shown "
            f"(call again with max_chunks={total} to retrieve all)."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
