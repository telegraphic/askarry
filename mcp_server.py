"""
MCP server exposing the astronomy-RAG retrieval pipeline to Claude Desktop
(and any other MCP-capable frontier LLM client).

Tools
-----
search_astronomy_docs   Semantic + BM25 hybrid search, re-ranked; returns
                        passages with title, section hierarchy, page, relevance.
search_textbooks        Same hybrid search over a separate radio-astronomy
                        textbook index (interferometry, receivers, pulsars) —
                        technical background, not SKA-specific facts.
list_textbooks          Textbooks available to search_textbooks.
list_documents          Table of contents grouped by section.
get_document_chunks     All indexed chunks from a named paper in reading order.
query_sensitivity_calculator
                        Live query to the SKAO sensitivity calculator REST API.
validate_observing_setup
                        Validate an SKA telescope observing configuration against
                        the vendored ska-sci-ops-setup-validator rule engine.
estimate_data_product_size
                        Estimate the data volume/rate of an SKA data product using
                        the vendored odp-data-size-tool formulas.
list_capability_schemas, describe_schema
                        Introspect the setup-validator's schema directly (allowed
                        subarray templates, min/max ranges, fixed values, per
                        context/telescope) — more authoritative than static
                        capability docs since it's the schema real configs are
                        validated against.
describe_rules          Cross-field validation rules (why a combination of
                        otherwise-valid values can still fail) for a schema.
get_context_defaults    Default/max bandwidth, beam-count, and channel-count
                        values per telescope/band for an observing context —
                        same data the setup-validator frontend pre-fills forms with.
get_subarray_layout     Max baseline and station count for a subarray template
                        (via ska_ost_array_config).
list_odps, estimate_setup_data_volume
                        Bridge a validated observing setup to real data-volume
                        numbers for every output data product it defines.
resolve_subarray_name, resolve_context_subarrays
                        Translate a subarray-template name across the
                        sensitivity-calculator / ska_ost_array_config /
                        setup-validator-schema naming vocabularies.
search_google_scholar_key_words, search_google_scholar_advanced
                        Search Google Scholar by keyword, or filtered by
                        author/year range.
get_author_info_tool   Look up an author's affiliation, interests, citation
                        count, and top publications on Google Scholar.
keyword_coverage_tool   Exact per-section/per-paper mention counts for terms.
verify_quote_tool       Check a quoted figure/sentence against its claimed paper.
get_paper_info_tool, list_missing_papers_tool
                        Bibliography details; chapters missing from the index.
lookup_acronym_tool, list_acronyms_tool
                        Acronym senses, variants, categories and papers.
compare_to_calculator_tool
                        Quoted sensitivity vs. live calculator, all weightings.
validate_across_contexts_tool
                        One obs_config validated under every observing context.
resolve_object_tool, ned_lookup_tool, cone_search_tool
                        SIMBAD/NED object lookup via astroquery (read-only).
build_source_db_tool, query_sources_tool, get_source_mentions_tool,
list_unresolved_sources_tool
                        Database of sources named in the papers (SIMBAD-resolved).
build_citation_db_tool, top_cited_papers_tool, citing_papers_tool
                        Database of references cited by the papers (most-cited works).

Resources
---------
askarry://atlas/taxonomies  Fixed observing-mode / technique lists for analyses.

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

import json
from pathlib import Path

import requests
from fastmcp import FastMCP

from rag import store
from rag import citation_db, corpus_tools, source_db
from rag.bibliography import bib_key, format_citation, list_toc_entries, lookup_citation, section_of
from rag.config import (
    DOC_SOURCE_AASKAII,
    DOC_SOURCE_LABELS,
    DOC_SOURCE_SKA_CAPABILITIES,
    MCP_BACKGROUND_TOP_K,
    MCP_RETRIEVAL_CANDIDATES,
    MCP_TEXTBOOK_TOP_K,
    MCP_TOP_K,
    SKA_CAPABILITIES_DIR,
    SUPPORTED_SUFFIXES,
    TEXTBOOKS_CHROMA_DIR,
    TEXTBOOKS_DIR,
    textbook_title,
)
from rag.data_product_estimator import estimate_data_product_size
from rag.google_scholar import get_author_info
from rag.google_scholar import search_google_scholar as _search_google_scholar
from rag.google_scholar import search_google_scholar_advanced as search_google_scholar_advanced_impl
from rag.observing_setup_capabilities import (
    describe_rules,
    describe_schema,
    get_context_defaults,
    list_capability_schemas,
)
from rag.observing_setup_data_volume import estimate_setup_data_volume, list_odps
from rag.observing_setup_validator import validate_across_contexts, validate_observing_setup
from rag.retrieval import retrieve
from rag.sensitivity_calculator import compare_to_calculator, query_sensitivity_calculator
from rag.sky_lookup import cone_search, ned_lookup, resolve_object
from rag.subarray_layout import get_subarray_layout
from rag.subarray_names import resolve_context_subarrays, resolve_subarray_name

mcp = FastMCP(
    "astronomy-rag",
    instructions=(
        "This server provides access to an indexed collection of SKA (Square "
        "Kilometre Array) astronomy papers, mainly from the AASKAII "
        "compendium but also including chapters from its 10-years-older "
        "predecessor, the 2015 'Advancing Astrophysics with the SKA' "
        "proceedings. Each passage returned by search_astronomy_docs is "
        "labeled with its source book and year — when passages from the two "
        "books conflict, prefer the newer AASKAII information. Use "
        "search_astronomy_docs to find relevant passages for a question, "
        "list_documents to discover what papers are available, and "
        "get_document_chunks to read a specific paper in full.\n"
        "\n"
        "Document sets: search_astronomy_docs covers AASKAII science papers "
        "(methods/results); search_ska_capabilities covers a separate SKA Key "
        "Capabilities technical/engineering document set. Choose based on "
        "whether the question is about science outcomes or technical/"
        "observational capability; search both if unsure. Tool and document "
        "inventory changes over time — call list_documents / "
        "list_ska_capability_docs / list_capability_schemas_tool to check "
        "what currently exists rather than assuming a tool or doc is present "
        "or absent.\n"
        "\n"
        "MOST IMPORTANT RULE: static documents describe the aspirational full "
        "design, not current stage-specific limits. Example: asked for the "
        "max subarrays for SKA-Low AA2, search_ska_capabilities surfaces "
        "'up to 16 concurrent subarrays' — the full-design figure — but "
        "describe_schema_tool('obs_config_container', context='sv_aa2', "
        "telescope='ska_low') shows the real enforced limit is n_subarray "
        "max=1 for that specific context (max=4 for sv_aastar; 16 only at "
        "the full AA4 design baseline). For ANY 'what can I currently "
        "configure/do' question, always cross-check describe_schema_tool / "
        "describe_rules_tool / validate_observing_setup_tool against the "
        "matching context (e.g. 'sv_aa2', 'sv_aastar', 'cycle_0') rather than "
        "relying on static capability documents alone. Use the documents for "
        "background, rationale, and science; use the live schema/validator "
        "for current numeric limits. When a document and a live tool "
        "disagree, trust the live tool.\n"
        "\n"
        "Schema-first, not guess-first: validate_observing_setup_tool's "
        "obs_config schema is strict, and its error messages for a missing "
        "nested field are often generic (they may name only the containing "
        "block, not the specific missing field) — blind trial-and-error "
        "against it burns many calls without converging. Always call "
        "list_capability_schemas_tool + describe_schema_tool (and "
        "describe_rules_tool for cross-field constraints) first to get exact "
        "field names, allowed values, and min/max/units before constructing "
        "an obs_config. Same applies to estimate_data_product_size_tool / "
        "estimate_setup_data_volume_tool — call list_odps_tool or check the "
        "matching odps/<product_type> schema first, and note that some "
        "required fields (observation duration, raw correlator dump time, "
        "image resolution/field-of-view) are never in the schema and must be "
        "supplied directly — see list_odps_tool's missing_params.\n"
        "\n"
        "Context sensitivity: whenever a question names or implies a "
        "specific array assembly (AA0.5/AA1/AA2/AA*/AA4) or observing cycle, "
        "pass that as the context argument explicitly — answers differ "
        "substantially by context and a default may not match what's asked.\n"
        "\n"
        "Live vs. static data: query_sensitivity_calc, get_subarray_layout_"
        "tool, resolve_subarray_name_tool/resolve_context_subarrays_tool, and "
        "the Google Scholar tools, and the SIMBAD/NED lookup tools all call real external services/packages "
        "over the network — results are live, not cached, and can change. "
        "Call query_sensitivity_calc's 'subarrays' endpoint (or "
        "resolve_context_subarrays_tool, which does the name translation for "
        "you) before calling continuum/zoom/pss calculate endpoints, to get "
        "a valid subarray_configuration name.\n"
        "\n"
        "Technical background: search_textbooks covers a separate index of "
        "standard radio-astronomy textbooks (Thompson/Moran/Swenson "
        "interferometry & synthesis imaging, Tools of Radio Astronomy, "
        "Essential Radio Astronomy, pulsar handbooks) — established "
        "fundamentals such as interferometry, imaging/calibration theory, "
        "receivers and noise, radiation processes, and pulsar physics/"
        "timing. Use it (or pass include_background=True to "
        "search_astronomy_docs / search_ska_capabilities) for derivations, "
        "definitions, 'how does X work' questions, or when an SKA paper "
        "assumes background the user may lack. Textbooks predate SKA and "
        "never override SKA-specific numbers from the capability docs or "
        "live tools. list_textbooks shows what's indexed.\n"
        "\n"
        "Google Scholar tools (search_google_scholar_key_words, "
        "search_google_scholar_advanced, get_author_info_tool): good for "
        "science-justification/literature-review content, citation tracking, "
        "and sanity-checking whether a topic has already been published on. "
        "Treat results as noisy — preprint and published versions of the "
        "same paper often appear as separate entries (de-duplicate before "
        "citing), citation counts can be inflated by self-citation, and "
        "common author names are ambiguous (use get_author_info_tool to "
        "disambiguate when precision matters). Use Scholar for external/"
        "citation context; use search_astronomy_docs for SKA-specific "
        "background already curated in this dataset — never rely on Scholar "
        "for SKA capability facts.\n"
        "\n"
        "Scope limit for proposal-drafting use cases: this server covers the "
        "technical-feasibility side of an SKA observing/project proposal "
        "well (setup validation, sensitivity, data volume, science "
        "background, literature search). It has NO coverage of the "
        "administrative/policy side — no proposal template, no submission "
        "deadlines/portal, no finalized TAC review criteria, no finalized "
        "data rights/proprietary period, no cost model. State this "
        "explicitly rather than inventing such details, and use web search "
        "for current SKAO timeline/policy pages, since those sit outside "
        "this dataset and change over time.\n"
        "\n"
        "Corpus-wide analysis (counting, ranking, tabulating across papers): "
        "use keyword_coverage_tool for 'how many sections/papers mention X' "
        "— exact counts, not a tally of search hits. Scope searches with "
        "search_astronomy_docs' book/section/paper filters instead of "
        "prompt instructions, and use the real book sections from "
        "list_documents(book=...) rather than inventing themes. Before "
        "publishing any sourced figure, run verify_quote_tool on it with its "
        "claimed paper; use compare_to_calculator_tool to check quoted "
        "sensitivities and validate_across_contexts_tool to check stage "
        "feasibility. Check list_missing_papers_tool before concluding a "
        "topic is absent. The askarry://atlas/taxonomies resource holds the "
        "fixed category lists used by earlier analyses.\n"
        "\n"
        "Housekeeping: large search results can exceed output size limits — "
        "prefer specific queries and a modest top_k over broad, high-top_k "
        "ones."
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

    book_label = DOC_SOURCE_LABELS.get(chunk.get("doc_source", DOC_SOURCE_AASKAII), "")
    lines = [f"[{i}] {doc_title} ({book_label})" if book_label else f"[{i}] {doc_title}"]
    lines.append(f"    Paper ID: {bib_key(chunk['source'])} (chunk {chunk.get('chunk_index', 0)})")
    book_section = section_of(chunk["source"])
    if book_section:
        lines.append(f"    Book section: {book_section}")
    if section:
        lines.append(f"    Section: {section}")
    if page:
        lines.append(f"    Page: {page}")
    lines.append(f"    Relevance: {score:.0%}")
    if caption:
        lines.append(f"    Caption: {caption}")
    lines.append(f"    {chunk['text']}")
    return "\n".join(lines)


def _search(
    query: str,
    top_k: int,
    where: dict | None = None,
    include_background: bool = False,
    as_json: bool = False,
) -> str:
    """Run the main-index search and render passages; optionally append the
    top textbook passages for the same query, numbered after the main ones."""
    chunks = retrieve(
        query,
        top_k=top_k,
        use_hyde=False,          # never require Ollama in the MCP path
        expansion_window=2,      # wider context per passage for frontier LLMs
        where=where,
    )
    if as_json:
        return json.dumps([
            {
                "n": i,
                "paper": bib_key(c["source"]),
                "chunk_index": c["chunk_index"],
                "title": c.get("doc_title") or (lookup_citation(c["source"]) or {}).get("title", ""),
                "book": DOC_SOURCE_LABELS.get(c["doc_source"], c["doc_source"]),
                "book_section": section_of(c["source"]),
                "heading": c.get("section_path") or c.get("heading", ""),
                "page": c.get("page_no", 0),
                "score": c.get("score", 0.0),
                "text": c["text"],
            }
            for i, c in enumerate(chunks, 1)
        ])
    parts = [_format_passage(i, c) for i, c in enumerate(chunks, 1)]
    out = "\n\n---\n\n".join(parts) if parts else "No relevant passages found for that query."
    if include_background:
        try:
            background = retrieve(
                query,
                top_k=MCP_BACKGROUND_TOP_K,
                use_hyde=False,
                expansion_window=1,
                db_dir=TEXTBOOKS_CHROMA_DIR,
            )
        except RuntimeError as exc:  # textbook store not ingested yet
            return f"{out}\n\n## Technical background (textbooks)\n{exc}"
        if background:
            bg_parts = [
                _format_passage(i, c) for i, c in enumerate(background, len(chunks) + 1)
            ]
            out += "\n\n## Technical background (textbooks)\n\n" + "\n\n---\n\n".join(bg_parts)
    return out


@mcp.tool()
def search_astronomy_docs(
    query: str,
    top_k: int = MCP_TOP_K,
    include_background: bool = False,
    book: str | None = None,
    section: str | None = None,
    paper: str | None = None,
    as_json: bool = False,
) -> str:
    """Search the SKA astronomy document database for passages relevant to a query.

    Uses a hybrid pipeline: semantic vector search + BM25 keyword search,
    fused with Reciprocal Rank Fusion, then cross-encoder re-ranked.
    Returns up to *top_k* passages with source title, section hierarchy, page
    number, and relevance score. The default value of *top_k* is set by
    MCP_TOP_K in config.py.

    Use this to answer questions about SKA science, engineering, and methods.
    Cite passages inline with [n] markers matching the returned passage numbers.

    Args:
        query:  The question or topic to search for.
        top_k:  Maximum number of passages to return (default: MCP_TOP_K from config.py).
        include_background: Also append a few textbook passages (see
                search_textbooks) for underlying technical background.
        book:   Restrict to one document set: "aaskaii" (2026 book),
                "aaska2015" (2015 book) or "ska_capabilities". Default: all.
        section: Restrict to one book section, exactly as list_documents
                names it (e.g. "The Cosmos", "Methods and Techniques").
                Combine with book to avoid mixing the two books.
        paper:  Restrict to one paper, by Paper ID/filename stem (e.g.
                "Vacca01") or a title substring.
        as_json: Return a JSON list of passages (paper, book, book_section,
                page, score, text) instead of formatted text.
    """
    try:
        where = corpus_tools.build_filter(book=book, section=section, paper=paper)
    except ValueError as exc:
        return f"Invalid request: {exc}"
    return _search(query, top_k, where=where, include_background=include_background, as_json=as_json)


@mcp.tool()
def search_ska_capabilities(
    query: str,
    top_k: int = MCP_TOP_K,
    include_background: bool = False,
    as_json: bool = False,
) -> str:
    """Search the SKA key capabilities technical documents for passages relevant to a query.

    This is a separate, smaller document set from the AASKAII science papers
    (see search_astronomy_docs) — technical material describing SKA's
    engineering capabilities rather than science results. Same hybrid
    search/rerank pipeline, scoped to just this document set.

    Args:
        query:  The question or topic to search for.
        top_k:  Maximum number of passages to return (default: MCP_TOP_K from config.py).
        include_background: Also append a few textbook passages (see
                search_textbooks) for underlying technical background.
        as_json: Return a JSON list of passages instead of formatted text.
    """
    return _search(
        query,
        top_k,
        where={"doc_source": DOC_SOURCE_SKA_CAPABILITIES},
        include_background=include_background,
        as_json=as_json,
    )


@mcp.tool()
def search_textbooks(query: str, top_k: int = MCP_TEXTBOOK_TOP_K) -> str:
    """Search radio-astronomy textbooks for technical background on a topic.

    A separate index from the SKA papers: standard references on
    interferometry and synthesis imaging, calibration, receivers and noise,
    radiation processes, and pulsar physics/timing. Use for fundamentals,
    derivations and definitions — not for SKA-specific numbers. Same hybrid
    search/rerank pipeline as search_astronomy_docs.

    Args:
        query:  The question or topic to search for.
        top_k:  Maximum number of passages to return (default: MCP_TEXTBOOK_TOP_K).
    """
    try:
        chunks = retrieve(
            query,
            top_k=top_k,
            use_hyde=False,
            expansion_window=1,  # textbook chunks are dense; ±1 is plenty
            db_dir=TEXTBOOKS_CHROMA_DIR,
        )
    except RuntimeError as exc:
        return str(exc)
    if not chunks:
        return "No relevant passages found for that query."
    return "\n\n---\n\n".join(_format_passage(i, c) for i, c in enumerate(chunks, 1))


@mcp.tool()
def list_textbooks() -> str:
    """List the textbooks available to search_textbooks (one line per book,
    with its number of PDF files when split into chapters)."""
    files = store.discover_files([TEXTBOOKS_DIR], SUPPORTED_SUFFIXES)
    if not files:
        return f"No textbooks found. Add PDFs to {TEXTBOOKS_DIR} and run `python ingest.py --textbooks`."
    books: dict[str, int] = {}
    for f in files:
        title = textbook_title(f)
        books[title] = books.get(title, 0) + 1
    return "\n".join(
        f"- {title}" + (f" ({n} chapter files)" if n > 1 else "")
        for title, n in sorted(books.items())
    )


@mcp.tool()
def list_ska_capability_docs() -> str:
    """List all SKA key capabilities technical documents available to search.

    Use this before search_ska_capabilities to discover what documents exist
    in this separate (non-AASKAII) document set.
    """
    files = store.discover_files([SKA_CAPABILITIES_DIR], SUPPORTED_SUFFIXES)
    if not files:
        return (
            "No SKA key capabilities documents found. Add PDF/HTML files to "
            f"{SKA_CAPABILITIES_DIR} and run `python ingest.py`."
        )
    return "\n".join(f"- {f.name}" for f in files)


@mcp.tool()
def list_documents(book: str | None = None) -> str:
    """List all astronomy papers in the database, grouped by book section.

    Use this before searching to discover what documents are available,
    or to answer questions like "what papers cover gravitational waves?".
    Each line ends with the paper's ID (filename stem), which the
    search_astronomy_docs `paper` filter and other tools accept.

    Args:
        book: "aaskaii" (2026 book) or "aaska2015" (2015 book). Default: both,
            with the two books' section names mixed together.
    """
    try:
        toc = list_toc_entries(corpus_tools.BOOKS[book] if book else None)
    except KeyError:
        return f"Invalid request: unknown book {book!r}; expected 'aaskaii' or 'aaska2015'."
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
            lines.append(f"- {entry['title']}{author_str} [{bib_key(entry['path'])}]")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_document_chunks(filename: str, max_chunks: int = 60) -> str:
    """Retrieve all indexed chunks from a specific document in reading order.

    Useful for summarising or deeply analysing a single paper. The document
    is identified by its paper ID from list_documents() (e.g. 'Vacca01', or
    'AASKA2015/Vacca01' for the 2015 chapter with the same filename), filename
    stem (e.g. 'ska_mid_baseline_design') or full filename including extension.

    Returns up to *max_chunks* consecutive passages (default 60). If the
    document has more chunks they are noted at the end.

    Args:
        filename:   Filename stem or full filename of the document.
        max_chunks: Maximum number of chunks to return (default 60).
    """
    collection = store.get_chroma_collection()
    if collection.count() == 0:
        return "The vector store is empty. Run `python ingest.py` first."

    wanted = Path(filename).with_suffix("").as_posix()  # strip extension if present

    # Step 1: fetch only metadatas to find the canonical source path (fast).
    # Exact paper ID first (e.g. "AASKA2015/Vacca01" vs AASKAII "Vacca01"),
    # then bare filename stem.
    sources = {m.get("source", "") for m in collection.get(include=["metadatas"])["metadatas"]}
    matching_source = next((s for s in sources if bib_key(s) == wanted), None) or next(
        (s for s in sorted(sources) if Path(s).stem == Path(wanted).name), None)

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


@mcp.tool()
def query_sensitivity_calc(telescope: str, endpoint: str, params: dict) -> str:
    """Query the live SKAO sensitivity calculator REST API (sensitivity-calculator.skao.int).

    This calls the real backend service over the network — it is not a
    document search. Call "subarrays" first if unsure which subarray_configuration
    values are valid; use search_ska_capabilities for theoretical background.

    Args:
        telescope: "low" or "mid".
        endpoint:  One of "subarrays", "continuum/calculate", "zoom/calculate",
                   "pss/calculate". "subarrays" takes params={}.

        params:    Query parameters for the endpoint. Common to all: pointing_centre
                   ("HH:MM:SS[.ss] [+|-]DD:MM:SS[.ss]"), weighting_mode ("uniform",
                   "natural", or "robust" — plus robustness if "robust"),
                   subarray_configuration (from "subarrays"; Low alt: num_stations,
                   Mid alt: n_ska+n_meer for a custom subarray).

                   continuum/calculate — Low: freq_centre_mhz, bandwidth_mhz,
                   integration_time_h, elevation_limit (optional). Mid: rx_band
                   (e.g. "Band 1"), freq_centre_hz, bandwidth_hz, integration_time_s
                   (or supplied_sensitivity + sensitivity_unit "Jy/beam"/"K" to solve
                   for integration time instead).
                   Example: {"integration_time_h": 1, "subarray_configuration":
                   "LOW_AAstar_all", "freq_centre_mhz": 200, "bandwidth_mhz": 300,
                   "pointing_centre": "0 0", "weighting_mode": "uniform"}

                   zoom/calculate — one or more zoom windows, one value per window in
                   each array param. Low: freq_centres_mhz, total_bandwidths_khz,
                   spectral_resolutions_hz, integration_time_h. Mid (AA*/AA4/custom
                   only): rx_band, freq_centres_hz, total_bandwidths_hz,
                   spectral_resolutions_hz, integration_time_s.

                   pss/calculate — only valid for subarrays with max baseline < 20 km
                   (check "subarrays"). pulsar_mode ("folded_pulse" or "single_pulse"),
                   dm, intrinsic_pulse_width (ms, required if folded_pulse),
                   pulse_period (ms, ignored if single_pulse). Low: freq_centre_mhz,
                   bandwidth_mhz (ignored if folded_pulse), integration_time_h. Mid:
                   rx_band, freq_centre_hz, bandwidth_hz, integration_time_s.

                   Full parameter reference (advanced overrides like taper, pwv, el,
                   eta_*, t_sys_*): rag/sensitivity_calculator.py docstrings, or the
                   live OpenAPI spec at sensitivity-calculator.skao.int/api/v11/
                   {telescope}/openapi.json.
    """
    try:
        result = query_sensitivity_calculator(telescope, endpoint, params)
    except ValueError as exc:
        return f"Invalid request: {exc}"
    except requests.HTTPError as exc:
        return f"Sensitivity calculator API error: {exc}"
    return str(result)


@mcp.tool()
def validate_observing_setup_tool(obs_config: dict) -> str:
    """Validate an SKA telescope observing configuration.

    Runs the vendored ska-sci-ops-setup-validator rule engine (same rules the
    real setup-validator frontend/backend enforces) against a full observing
    configuration: subarrays, beams, and continuum/zoom/PSS/PST mode settings
    with their output data products.

    Args:
        obs_config: Dict with keys "context" (e.g. "Cycle 0", "SV-AA*"),
            "telescopeType" ("SKA-Low" or "SKA-Mid"), and "subarrays" (a dict
            of per-subarray template/band/beam/mode settings). See
            rag/vendor/setup_validator/schema/obs_config.yaml for the allowed
            context/telescope values and full config shape.
    """
    try:
        result = validate_observing_setup(obs_config)
    except (KeyError, ValueError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def estimate_data_product_size_tool(product_type: str, params: dict) -> str:
    """Estimate the data volume (and rate, where applicable) of an SKA data product.

    Uses the vendored odp-data-size-tool formulas for a single data product
    instance (not a full observing setup).

    Args:
        product_type: One of "Image", "Image Cutout", "Gridded Visibilities",
            "PSS", "PST Folded", "Dynamic Spectrum", "Flowthrough", "VLBI",
            "Transient Dump", "Calibrated Vis".
        params: Parameter values for that product type. Angular values
            (resolution, fov) and durations are unit strings (e.g.
            "1.000 arcsec", "3600.000 s"); bandwidths (bw, cbw) accept a
            plain Hz number or a unit string (e.g. "300.000 MHz"); everything
            else is a plain number/bool/string. See row_to_model in
            rag/vendor/data_size/model_io.py for the exact required keys per
            product_type.
    """
    try:
        result = estimate_data_product_size(product_type, params)
    except Exception as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def list_capability_schemas_tool(telescope: str = "ska_low") -> str:
    """List the setup-validator schema names available to query for a telescope.

    Call this first to discover valid `schema` values for describe_schema_tool,
    e.g. "subarray_config", "continuum_settings", "beam_config",
    "pss_settings", "pst_settings", "zoom_settings", "obs_config_container",
    "odps/calibrated_visibilities", "odps/image_cube", etc.

    Args:
        telescope: "ska_low"/"SKA-Low" or "ska_mid"/"SKA-Mid".
    """
    try:
        result = list_capability_schemas(telescope)
    except ValueError as exc:
        return f"Invalid request: {exc}"
    return "\n".join(result)


@mcp.tool()
def describe_schema_tool(schema: str, context: str | None = None, telescope: str | None = None) -> str:
    """Look up allowed values / min-max ranges / fixed values for a setup-validator schema.

    Reads the same validation schema the real setup-validator enforces —
    this is the authoritative, up-to-date source for "what are my options?"
    questions (e.g. which subarray templates exist for a given context and
    telescope), more reliable than static capability documentation.

    Some numeric bounds contain "$"-prefixed placeholders (e.g. "$rx_low",
    "$rx_high") that are resolved at validation time against the selected
    receiver band — these indicate the bound is band-dependent rather than
    fixed.

    Args:
        schema: Schema name — see list_capability_schemas_tool for the set
            available for a telescope.
        context: Observing context, e.g. "base", "cycle_0", "cycle_1",
            "sv_aa2"/"SV-AA2", "sv_aastar"/"SV-AA*". Omit to load only the
            top-level (non-telescope-specific) schema, if one exists.
        telescope: "ska_low"/"SKA-Low" or "ska_mid"/"SKA-Mid". Required
            whenever context is given.
    """
    try:
        result = describe_schema(schema, context=context, telescope=telescope)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def describe_rules_tool(schema: str, context: str | None = None, telescope: str | None = None) -> str:
    """Look up cross-field validation rules for a setup-validator schema.

    Complements describe_schema_tool: that gives per-parameter bounds, this
    explains *why* a combination of otherwise-individually-valid values can
    still fail validation (e.g. total PST beams across subarrays exceeding a
    context-dependent maximum).

    Args:
        schema: Schema name — see list_capability_schemas_tool for the set
            available for a telescope.
        context: Observing context, e.g. "base", "cycle_0", "cycle_1",
            "sv_aa2"/"SV-AA2", "sv_aastar"/"SV-AA*". Omit to load only the
            top-level (non-telescope-specific) schema, if one exists.
        telescope: "ska_low"/"SKA-Low" or "ska_mid"/"SKA-Mid". Required
            whenever context is given.
    """
    try:
        result = describe_rules(schema, context=context, telescope=telescope)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def get_context_defaults_tool(context: str) -> str:
    """Get default/max continuum, PSS, and PST bandwidth, PST beam count, and
    zoom/spectral channel count values, per telescope (and per receiver band
    for SKA-Mid), for an observing context.

    Same data the setup-validator frontend GUI pre-fills its forms with —
    use this when constructing a new observing_setup from scratch, as a
    starting point before calling validate_observing_setup_tool.

    Args:
        context: Observing context, e.g. "base", "cycle_0", "cycle_1",
            "sv_aa2"/"SV-AA2", "sv_aastar"/"SV-AA*".
    """
    try:
        result = get_context_defaults(context)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def get_subarray_layout_tool(subarray_template: str) -> str:
    """Look up physical layout facts for an SKA subarray template.

    Uses ska_ost_array_config (the SKA-internal antenna-layout package) to
    return the max baseline length and station/baseline counts for a named
    subarray template — e.g. to check what angular resolution (and hence
    field-of-view/pixel count) a given subarray can actually achieve, before
    feeding resolution/fov into estimate_data_product_size_tool.

    Args:
        subarray_template: e.g. "Low_full_AA2", "Mid_full_AA4"
            (case-insensitive — matches the `template` values from
            describe_schema_tool("subarray_config", ...) or an obs_config's
            subarray "template" field).
    """
    try:
        result = get_subarray_layout(subarray_template)
    except ValueError as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def list_odps_tool(obs_config: dict) -> str:
    """List every output data product (ODP) in an observing setup, with what's
    known vs. still missing to estimate its data volume.

    Validates obs_config first (same rules as validate_observing_setup_tool).
    For each ODP, reports the mapped data-size-tool product type (or null if
    unsupported — currently "pst_timing" ODPs aren't auto-mappable, their
    schema doesn't carry the fields needed) plus known_params (derived from
    the setup-validator schema and, for calibrated_visibilities, station
    count via ska_ost_array_config) and missing_params (required fields the
    schema genuinely doesn't track — observation duration always, plus
    resolution/fov for images and raw dump time for calibrated visibilities
    — supply these via estimate_setup_data_volume_tool's extra_params).

    Args:
        obs_config: Same dict shape as validate_observing_setup_tool.
    """
    try:
        result = list_odps(obs_config)
    except (KeyError, ValueError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def estimate_setup_data_volume_tool(obs_config: dict, extra_params: dict | None = None) -> str:
    """Estimate total data volume across every supported ODP in a validated observing setup.

    Call list_odps_tool first to see each ODP's missing_params, then supply
    them here via extra_params (keyed by "{subarray_id}/{beam_id}/{odp_label}",
    each a dict of the missing fields, e.g. {"duration": "3600.000 s"} or, for
    an Image ODP, {"resolution": "1.000 arcsec", "fov": "2.000 deg"}). ODPs
    still missing required params after merging are skipped and listed under
    "unresolved" rather than causing an error; ODP types with no auto-mapping
    (e.g. "pst_timing") are listed under "unsupported".

    Args:
        obs_config: Same dict shape as validate_observing_setup_tool.
        extra_params: Optional dict of caller-supplied fill-ins, see above.
    """
    try:
        result = estimate_setup_data_volume(obs_config, extra_params=extra_params)
    except (KeyError, ValueError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def resolve_subarray_name_tool(name: str, telescope: str | None = None) -> str:
    """Translate a subarray-template-like name across the three SKA naming
    vocabularies this server touches: the live sensitivity calculator, the
    ska_ost_array_config canonical template catalog, and the setup-validator
    schema's per-context allowed templates.

    Use this whenever a name from one tool (e.g. a setup-validator
    subarray_config `template` value) needs to be checked against or used
    with another (e.g. query_sensitivity_calc's subarray_configuration param).

    Args:
        name: e.g. "LOW_AAstar_all" (sensitivity calculator),
            "LOW_FULL_AASTAR" (ska_ost_array_config), or "Low_full_AA2"
            (setup-validator schema).
        telescope: Restrict to "low"/"mid" (or "ska_low"/"ska_mid"). Omit to
            check both.
    """
    try:
        result = resolve_subarray_name(name, telescope=telescope)
    except ValueError as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def resolve_context_subarrays_tool(context: str, telescope: str) -> str:
    """For a setup-validator observing context, resolve each allowed subarray
    template to its live sensitivity-calculator entry.

    Chains describe_schema_tool("subarray_config", context, telescope) ->
    a sensitivity-calculator subarray_configuration value, without guessing
    a name translation. A null sensitivity_calculator_entry means no live
    match was found for that template (rare schema inconsistency — see
    rag/subarray_names.py for a known example).

    Args:
        context: Observing context, e.g. "sv_aa2"/"SV-AA2", "cycle_0".
        telescope: "low"/"mid" or "ska_low"/"ska_mid".
    """
    try:
        result = resolve_context_subarrays(context, telescope)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return f"Invalid request: {exc}"
    return str(result)


@mcp.tool()
def search_google_scholar_key_words(query: str, num_results: int = 5) -> str:
    """Search Google Scholar for articles matching a keyword query.

    Args:
        query: Search query string (e.g. paper title or author).
        num_results: Number of results to return (default: 5).
    """
    try:
        result = _search_google_scholar(query, num_results)
    except requests.HTTPError as exc:
        return f"Google Scholar request failed: {exc}"
    return str(result)


@mcp.tool()
def search_google_scholar_advanced(
    query: str,
    author: str | None = None,
    year_range: tuple[int, int] | None = None,
    num_results: int = 5,
) -> str:
    """Search Google Scholar for articles, filtered by author and/or year range.

    Args:
        query: General search query.
        author: Author name to filter by.
        year_range: (start_year, end_year) to filter by publication year.
        num_results: Number of results to return (default: 5).
    """
    try:
        result = search_google_scholar_advanced_impl(query, author, year_range, num_results)
    except requests.HTTPError as exc:
        return f"Google Scholar request failed: {exc}"
    return str(result)


@mcp.tool()
def get_author_info_tool(author_name: str) -> str:
    """Get an author's affiliation, interests, citation count, and top publications from Google Scholar.

    Args:
        author_name: Name of the author to search for.
    """
    try:
        result = get_author_info(author_name)
    except StopIteration:
        return f"No Google Scholar author found matching {author_name!r}."
    return str(result)


# ---------------------------------------------------------------------------
# Corpus analysis tools (exact counts, verification, metadata) — built for
# multi-agent sweeps like the AASKAII Atlas, where hand-tallied search hits
# and misattributed figures were the weakest points.
# ---------------------------------------------------------------------------


@mcp.tool()
def keyword_coverage_tool(
    terms: list[str], book: str | None = "aaskaii", min_hits_per_paper: int = 1
) -> str:
    """Exact, repeatable counts of which book sections and papers mention each term.

    Use this instead of running one search per keyword and tallying the
    results by hand, e.g. for "how many sections use technique X" rankings.
    It scans every indexed chunk literally (whole-word match), so it measures
    mentions, not relevance: follow up with search_astronomy_docs(paper=...)
    to see how a paper actually uses a term.

    Args:
        terms: Terms to count. Use "|" for synonyms counted as one term, e.g.
            "Faraday rotation|rotation measure|RM". All-caps alternatives
            match case-sensitively and also match their known expansion.
        book: "aaskaii" (default), "aaska2015", "ska_capabilities", or null
            for everything.
        min_hits_per_paper: Mentions a paper needs before it counts as
            covering a term (raise to 3-5 to drop passing mentions).
    """
    try:
        return json.dumps(corpus_tools.keyword_coverage(terms, book, min_hits_per_paper))
    except ValueError as exc:
        return f"Invalid request: {exc}"


@mcp.tool()
def verify_quote_tool(text: str, paper: str | None = None, top_k: int = 5) -> str:
    """Check that a quoted sentence or figure really comes from the paper it is attributed to.

    Run this on every sourced figure before publishing a table. It combines
    ranked search with a literal scan for chunks that contain every number
    in *text*. Verdicts: "supported_by_claimed_paper", "found_in_other_paper"
    (a likely citation swap: see literal_number_matches / search_matches for
    the real source), "not_found", or with no paper "found"/"not_found".

    Args:
        text: The claim, ideally with its numbers, e.g.
            "sigma_Q,U = 0.24 uJy/beam at 50 hours".
        paper: Claimed source, as a Paper ID/filename stem or title substring.
        top_k: Ranked search matches to check across the whole corpus.
    """
    try:
        return json.dumps(corpus_tools.verify_quote(text, paper, top_k))
    except ValueError as exc:
        return f"Invalid request: {exc}"


@mcp.tool()
def get_paper_info_tool(name: str) -> str:
    """Bibliographic details for a paper: title, authors, book, section,
    formatted citation, and how many chunks are indexed (0 = listed in the
    book but not searchable).

    Args:
        name: Paper ID/filename stem (exact), or a title or author substring.
    """
    return json.dumps(corpus_tools.paper_info(name))


@mcp.tool()
def list_missing_papers_tool(book: str | None = None) -> str:
    """List book chapters that are in the bibliography but not indexed (so no
    search can find them), plus indexed files with no bibliography entry.

    Check this before claiming a topic is absent from a book: the paper
    covering it may simply be missing from the index.

    Args:
        book: "aaskaii" or "aaska2015". Default: both.
    """
    try:
        return json.dumps(corpus_tools.missing_papers(book))
    except ValueError as exc:
        return f"Invalid request: {exc}"


@mcp.tool()
def lookup_acronym_tool(acronym: str, book: str = "aaskaii") -> str:
    """Every meaning of an acronym as defined inline in the corpus, with
    spelling variants, usage count, and the papers that use it. Ambiguous
    acronyms (e.g. DM = dispersion measure / dark matter) return one entry
    per meaning.

    Args:
        acronym: e.g. "CSP", "DM".
        book: "aaskaii" (default), "aaska2015", or "ska_capabilities".
    """
    try:
        return json.dumps(corpus_tools.lookup_acronym(acronym, book))
    except ValueError as exc:
        return f"Invalid request: {exc}"


@mcp.tool()
def list_acronyms_tool(book: str = "aaskaii", category: str | None = None, limit: int = 100) -> str:
    """The most-used acronyms in a book, most frequent first.

    Categories come from a keyword heuristic on each expansion, so treat
    them as a browsing aid, not an authoritative taxonomy.

    Args:
        book: "aaskaii" (default), "aaska2015", or "ska_capabilities".
        category: Optional filter: "Methods & Software", "Organizations &
            Programs", "Telescopes & Instruments", or "Science & Astrophysics".
        limit: Max rows (default 100).
    """
    try:
        return json.dumps(corpus_tools.list_acronyms(book, category, limit))
    except ValueError as exc:
        return f"Invalid request: {exc}"


@mcp.tool()
def compare_to_calculator_tool(
    telescope: str,
    params: dict,
    claimed_sensitivity: float,
    unit: str = "uJy/beam",
    quantity: str = "continuum",
    claimed_beam_arcsec: float | None = None,
) -> str:
    """Check a paper's quoted sensitivity against the live sensitivity calculator,
    sweeping every weighting scheme (natural, robust -2..2, uniform).

    Papers often leave the weighting unstated, which is the usual reason a
    quoted figure fails to reproduce. This reports each variant's
    sensitivity and % difference, the closest one, and all variants within
    10%. Pass claimed_beam_arcsec when the paper states a resolution, so the
    match must fit both noise and beam.

    Args:
        telescope: "low" or "mid".
        params: continuum/calculate params as for query_sensitivity_calc,
            minus weighting_mode/robustness (the sweep sets those).
        claimed_sensitivity: The paper's figure.
        unit: "Jy/beam", "mJy/beam", "uJy/beam" (default) or "nJy/beam".
        quantity: "continuum" (default) or "spectral" (per-channel figure).
        claimed_beam_arcsec: Optional stated resolution, arcsec.
    """
    try:
        return json.dumps(compare_to_calculator(
            telescope, params, claimed_sensitivity, unit, quantity, claimed_beam_arcsec
        ))
    except (ValueError, KeyError) as exc:
        return f"Invalid request: {exc}"


@mcp.tool()
def validate_across_contexts_tool(obs_config: dict, contexts: list[str] | None = None) -> str:
    """Validate one observing configuration under every observing context
    ("SV-AA2", "SV-AA*", "Cycle 0", "Cycle 1"), to see at which array-assembly
    stage or cycle a paper's requirement becomes feasible.

    Use this rather than static capability documents when judging whether a
    requirement is deliverable: those describe the full AA4 design.

    Args:
        obs_config: Same shape as validate_observing_setup_tool; its own
            "context" value is ignored.
        contexts: Optional subset of contexts to check.
    """
    try:
        return str(validate_across_contexts(obs_config, contexts))
    except (KeyError, ValueError) as exc:
        return f"Invalid request: {exc}"


# ---------------------------------------------------------------------------
# Astronomical object lookup (SIMBAD / NED via astroquery) — read-only.
# ---------------------------------------------------------------------------


def _sky_call(fn, *args) -> str:
    try:
        return json.dumps(fn(*args))
    except ValueError as exc:
        return f"Invalid request: {exc}"
    except Exception as exc:  # network/service failures from astroquery
        return f"Lookup service error: {exc}"


@mcp.tool()
def resolve_object_tool(name: str) -> str:
    """Resolve an astronomical object name via SIMBAD (falling back to NED).

    Returns position (degrees, sexagesimal, Galactic), SIMBAD object type,
    redshift, and common aliases, or {"found": false}. Use it to turn a
    source named in a paper (e.g. "Cen A", "PSR J0437-4715", "M83") into
    coordinates, e.g. before a visibility/LST check. Survey fields such as
    "EoR0" or "COSMOS" may not be in SIMBAD.

    Args:
        name: Object name as SIMBAD/NED would recognise it.
    """
    return _sky_call(resolve_object, name)


@mcp.tool()
def ned_lookup_tool(name: str) -> str:
    """Look up an extragalactic object in NED: position, NED type, redshift
    and recession velocity. Prefer resolve_object_tool for general use; use
    this when the NED redshift/velocity specifically matters.

    Args:
        name: Object name, e.g. "NGC 5128".
    """
    return _sky_call(ned_lookup, name)


@mcp.tool()
def cone_search_tool(target: str, radius_arcmin: float = 5.0, max_results: int = 50,
                     object_type: str | None = None) -> str:
    """List SIMBAD objects within a radius of a target, nearest first.

    Args:
        target: An object name, "ra dec" in degrees, or sexagesimal
            ("13:25:27.6 -43:01:09").
        radius_arcmin: Search radius in arcminutes (default 5).
        max_results: Max objects returned (n_found gives the full count).
        object_type: Optional SIMBAD object-type code filter, e.g. "Psr"
            (pulsar), "QSO", "G" (galaxy), "HII", "SNR".
    """
    return _sky_call(cone_search, target, radius_arcmin, max_results, object_type)


# ---------------------------------------------------------------------------
# Source database: sources named in the papers, resolved via SIMBAD.
# ---------------------------------------------------------------------------


@mcp.tool()
def build_source_db_tool(retry_unresolved: bool = False) -> str:
    """Build or update the database of astronomical sources named in the
    indexed papers (regex-extracted, resolved via SIMBAD).

    Incremental: only names never seen before are sent to SIMBAD, so reruns
    after an ingest are quick. The first build over the whole corpus takes
    about a minute. Returns counts of sources, names and mentions.

    Args:
        retry_unresolved: Also re-query names SIMBAD previously couldn't resolve.
    """
    try:
        return json.dumps(source_db.build_source_db(retry_unresolved=retry_unresolved))
    except Exception as exc:  # SIMBAD/network failures
        return f"Build failed (rerun to resume; progress is kept): {exc}"


@mcp.tool()
def query_sources_tool(
    object_type: str | None = None,
    book: str | None = None,
    section: str | None = None,
    paper: str | None = None,
    dec_min: float | None = None,
    dec_max: float | None = None,
    min_papers: int = 1,
    limit: int = 100,
) -> str:
    """List sources named in the papers, most-cited first, with position
    (RA/Dec, Galactic), SIMBAD type, redshift, the papers that mention each,
    and the names it appears under.

    Use this to build target lists, e.g. for visibility/LST analyses (use a
    Dec range to keep only sources that rise high enough at a site).
    Objects with no SIMBAD position (e.g. GW events) have null coordinates
    and are dropped by any dec_min/dec_max filter.

    Args:
        object_type: SIMBAD type code, e.g. "Psr", "G", "AGN", "SNR", "GlC".
        book: "aaskaii", "aaska2015" or "ska_capabilities".
        section: Book section, as list_documents names it.
        paper: Paper ID (filename stem).
        dec_min, dec_max: Declination range in degrees.
        min_papers: Only sources mentioned in at least this many papers.
        limit: Max rows (default 100).
    """
    book_tag = corpus_tools.BOOKS.get(book, book) if book else None
    return json.dumps(source_db.query_sources(
        source_db.connect(), object_type, book_tag, section, paper,
        dec_min, dec_max, min_papers, limit,
    ))


@mcp.tool()
def get_source_mentions_tool(name: str, limit: int = 50) -> str:
    """A source and every passage in the papers that mentions it, under any
    of its names (e.g. "Cen A" also finds "NGC 5128" and "Centaurus A"),
    with paper, book, section, page and surrounding text.

    Args:
        name: Any name seen in the papers, or the SIMBAD main identifier.
        limit: Max mentions returned (default 50).
    """
    return json.dumps(source_db.source_mentions(source_db.connect(), name, limit))


@mcp.tool()
def list_unresolved_sources_tool(limit: int = 200) -> str:
    """Names found in the papers that SIMBAD couldn't resolve, most-mentioned
    first: survey fields, truncated designations, typos or regex false
    positives. Candidates for a curated field list.

    Args:
        limit: Max rows (default 200).
    """
    return json.dumps(source_db.unresolved_names(source_db.connect(), limit))


# ---------------------------------------------------------------------------
# Citation database: references cited by the papers, parsed from their
# reference lists (offline; no external lookups).
# ---------------------------------------------------------------------------


@mcp.tool()
def build_citation_db_tool() -> str:
    """Rebuild the database of references cited by the indexed papers, parsed
    from each paper's reference list. Offline, takes ~10 s. Returns counts
    and the papers where no references were parsed.
    """
    return json.dumps(citation_db.build_citation_db())


@mcp.tool()
def top_cited_papers_tool(
    book: str | None = None,
    section: str | None = None,
    since_year: int | None = None,
    min_papers: int = 1,
    limit: int = 50,
) -> str:
    """Works most cited by the corpus, ranked by how many corpus papers cite
    each, with an example reference string, DOI/arXiv id and the citing papers.

    The same work is matched across citation styles by first author, year,
    volume and page (or DOI/arXiv id), so counts are a lower bound where
    papers cite a work inconsistently (e.g. arXiv-only vs journal version).
    Chapters of the two books themselves appear with keys like
    "aaskaii:taoan02" or "aaska14:67" (PoS id).

    Args:
        book: "aaskaii" or "aaska2015" — only count citations from that book.
        section: Book section, as list_documents names it.
        since_year: Only works published in or after this year.
        min_papers: Only works cited by at least this many papers.
        limit: Max rows (default 50).
    """
    book_tag = corpus_tools.BOOKS.get(book, book) if book else None
    return json.dumps(citation_db.top_cited(
        citation_db.connect(), book_tag, section, since_year, min_papers, limit,
    ))


@mcp.tool()
def citing_papers_tool(text: str, limit: int = 100) -> str:
    """Which corpus papers cite a given work: searches every reference string
    (case-insensitive substring) and groups matches by cited work.

    Args:
        text: E.g. "Condon", "1998, AJ, 115" or a DOI/arXiv id.
        limit: Max cited works returned (default 100).
    """
    return json.dumps(citation_db.citing_papers(citation_db.connect(), text, limit))


# Fixed taxonomies from the AASKAII Atlas run, so parallel agents tag the same
# concept the same way without each brief re-pasting the lists.
ATLAS_TAXONOMIES = {
    "observing_modes": [
        "Deep HI intensity mapping", "Deep/wide HI galaxy survey",
        "Wide-field continuum survey", "Ultra-deep continuum imaging",
        "EoR / Cosmic Dawn 21-cm", "Spectral line / maser survey",
        "Polarimetry / RM grid", "VLBI / astrometry", "Pulsar timing array",
        "Pulsar search survey", "Transient / commensal / ToO",
        "Solar/heliospheric imaging", "Weak lensing",
        "Zoom-mode spectroscopy", "SETI / technosignature",
    ],
    "techniques": [
        "Faraday rotation / RM", "VLBI", "HI intensity mapping",
        "Machine learning / AI", "Polarimetry", "Pulsar timing",
        "Foreground removal", "Ionospheric calibration",
        "Deconvolution / imaging algorithms", "Spectral line / maser",
        "Commensal / multi-messenger", "Source finding / classification",
        "VLBI astrometry", "RFI mitigation", "Zeeman effect", "Citizen science",
    ],
}


@mcp.resource("askarry://atlas/taxonomies", mime_type="application/json")
def atlas_taxonomies() -> str:
    """The 15 observing-mode categories and 16 technique keywords the AASKAII
    Atlas fixed before searching; reuse them so repeated analyses stay comparable."""
    return json.dumps(ATLAS_TAXONOMIES)


if __name__ == "__main__":
    mcp.run()
