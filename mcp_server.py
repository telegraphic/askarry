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

import requests
from fastmcp import FastMCP

from rag import store
from rag.bibliography import format_citation, list_toc_entries, lookup_citation
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
from rag.observing_setup_validator import validate_observing_setup
from rag.retrieval import retrieve
from rag.sensitivity_calculator import query_sensitivity_calculator
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
        "the Google Scholar tools all call real external services/packages "
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
    query: str, top_k: int = MCP_TOP_K, include_background: bool = False
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
    """
    return _search(query, top_k, include_background=include_background)


@mcp.tool()
def search_ska_capabilities(
    query: str, top_k: int = MCP_TOP_K, include_background: bool = False
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
    """
    return _search(
        query,
        top_k,
        where={"doc_source": DOC_SOURCE_SKA_CAPABILITIES},
        include_background=include_background,
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


if __name__ == "__main__":
    mcp.run()
