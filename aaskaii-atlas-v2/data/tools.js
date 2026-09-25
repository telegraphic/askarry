(window.ATLAS = window.ATLAS || {})['tools'] = {
 "generated": {
  "git_sha": "a037c7f",
  "date": "2026-09-25",
  "args": {
   "source": "mcp_server.py"
  }
 },
 "tools": [
  {
   "name": "search_astronomy_docs",
   "kind": "tool",
   "summary": "Search the SKA astronomy document database for passages relevant to a query.",
   "params": [
    "query",
    "top_k",
    "include_background",
    "book",
    "section",
    "paper",
    "as_json"
   ]
  },
  {
   "name": "search_ska_capabilities",
   "kind": "tool",
   "summary": "Search the SKA key capabilities technical documents for passages relevant to a query.",
   "params": [
    "query",
    "top_k",
    "include_background",
    "as_json"
   ]
  },
  {
   "name": "search_textbooks",
   "kind": "tool",
   "summary": "Search radio-astronomy textbooks for technical background on a topic.",
   "params": [
    "query",
    "top_k"
   ]
  },
  {
   "name": "list_textbooks",
   "kind": "tool",
   "summary": "List the textbooks available to search_textbooks (one line per book, with its number of PDF files when split into chapters).",
   "params": []
  },
  {
   "name": "list_ska_capability_docs",
   "kind": "tool",
   "summary": "List all SKA key capabilities technical documents available to search.",
   "params": []
  },
  {
   "name": "list_documents",
   "kind": "tool",
   "summary": "List all astronomy papers in the database, grouped by book section.",
   "params": [
    "book"
   ]
  },
  {
   "name": "get_document_chunks",
   "kind": "tool",
   "summary": "Retrieve all indexed chunks from a specific document in reading order.",
   "params": [
    "filename",
    "max_chunks"
   ]
  },
  {
   "name": "query_sensitivity_calc",
   "kind": "tool",
   "summary": "Query the live SKAO sensitivity calculator REST API (sensitivity-calculator.skao.int).",
   "params": [
    "telescope",
    "endpoint",
    "params"
   ]
  },
  {
   "name": "validate_observing_setup_tool",
   "kind": "tool",
   "summary": "Validate an SKA telescope observing configuration.",
   "params": [
    "obs_config"
   ]
  },
  {
   "name": "estimate_data_product_size_tool",
   "kind": "tool",
   "summary": "Estimate the data volume (and rate, where applicable) of an SKA data product.",
   "params": [
    "product_type",
    "params"
   ]
  },
  {
   "name": "list_capability_schemas_tool",
   "kind": "tool",
   "summary": "List the setup-validator schema names available to query for a telescope.",
   "params": [
    "telescope"
   ]
  },
  {
   "name": "describe_schema_tool",
   "kind": "tool",
   "summary": "Look up allowed values / min-max ranges / fixed values for a setup-validator schema.",
   "params": [
    "schema",
    "context",
    "telescope"
   ]
  },
  {
   "name": "describe_rules_tool",
   "kind": "tool",
   "summary": "Look up cross-field validation rules for a setup-validator schema.",
   "params": [
    "schema",
    "context",
    "telescope"
   ]
  },
  {
   "name": "get_context_defaults_tool",
   "kind": "tool",
   "summary": "Get default/max continuum, PSS, and PST bandwidth, PST beam count, and zoom/spectral channel count values, per telescope (and per receiver band for SKA-Mid), for an observing context.",
   "params": [
    "context"
   ]
  },
  {
   "name": "get_subarray_layout_tool",
   "kind": "tool",
   "summary": "Look up physical layout facts for an SKA subarray template.",
   "params": [
    "subarray_template"
   ]
  },
  {
   "name": "list_odps_tool",
   "kind": "tool",
   "summary": "List every output data product (ODP) in an observing setup, with what's known vs. still missing to estimate its data volume.",
   "params": [
    "obs_config"
   ]
  },
  {
   "name": "estimate_setup_data_volume_tool",
   "kind": "tool",
   "summary": "Estimate total data volume across every supported ODP in a validated observing setup.",
   "params": [
    "obs_config",
    "extra_params"
   ]
  },
  {
   "name": "resolve_subarray_name_tool",
   "kind": "tool",
   "summary": "Translate a subarray-template-like name across the three SKA naming vocabularies this server touches: the live sensitivity calculator, the ska_ost_array_config canonical template catalog, and the setup-validator schema's per-context allowed templates.",
   "params": [
    "name",
    "telescope"
   ]
  },
  {
   "name": "resolve_context_subarrays_tool",
   "kind": "tool",
   "summary": "For a setup-validator observing context, resolve each allowed subarray template to its live sensitivity-calculator entry.",
   "params": [
    "context",
    "telescope"
   ]
  },
  {
   "name": "search_google_scholar_key_words",
   "kind": "tool",
   "summary": "Search Google Scholar for articles matching a keyword query.",
   "params": [
    "query",
    "num_results"
   ]
  },
  {
   "name": "search_google_scholar_advanced",
   "kind": "tool",
   "summary": "Search Google Scholar for articles, filtered by author and/or year range.",
   "params": [
    "query",
    "author",
    "year_range",
    "num_results"
   ]
  },
  {
   "name": "get_author_info_tool",
   "kind": "tool",
   "summary": "Get an author's affiliation, interests, citation count, and top publications from Google Scholar.",
   "params": [
    "author_name"
   ]
  },
  {
   "name": "keyword_coverage_tool",
   "kind": "tool",
   "summary": "Exact, repeatable counts of which book sections and papers mention each term.",
   "params": [
    "terms",
    "book",
    "min_hits_per_paper"
   ]
  },
  {
   "name": "verify_quote_tool",
   "kind": "tool",
   "summary": "Check that a quoted sentence or figure really comes from the paper it is attributed to.",
   "params": [
    "text",
    "paper",
    "top_k",
    "chunk_index",
    "book"
   ]
  },
  {
   "name": "get_paper_info_tool",
   "kind": "tool",
   "summary": "Bibliographic details for a paper: title, authors, book, section, formatted citation, and how many chunks are indexed (0 = listed in the book but not searchable).",
   "params": [
    "name"
   ]
  },
  {
   "name": "list_missing_papers_tool",
   "kind": "tool",
   "summary": "List book chapters that are in the bibliography but not indexed (so no search can find them), plus indexed files with no bibliography entry.",
   "params": [
    "book"
   ]
  },
  {
   "name": "lookup_acronym_tool",
   "kind": "tool",
   "summary": "Every meaning of an acronym as defined inline in the corpus, with spelling variants, usage count, and the papers that use it. Ambiguous acronyms (e.g. DM = dispersion measure / dark matter) return one entry per meaning.",
   "params": [
    "acronym",
    "book"
   ]
  },
  {
   "name": "list_acronyms_tool",
   "kind": "tool",
   "summary": "The most-used acronyms in a book, most frequent first.",
   "params": [
    "book",
    "category",
    "limit"
   ]
  },
  {
   "name": "compare_to_calculator_tool",
   "kind": "tool",
   "summary": "Check a paper's quoted sensitivity against the live sensitivity calculator, sweeping every weighting scheme (natural, robust -2..2, uniform) for continuum and zoom; pulsar search (\"pss\") is a single call (no weighting).",
   "params": [
    "telescope",
    "params",
    "claimed_sensitivity",
    "unit",
    "quantity",
    "claimed_beam_arcsec",
    "endpoint",
    "window_index"
   ]
  },
  {
   "name": "validate_across_contexts_tool",
   "kind": "tool",
   "summary": "Validate one observing configuration under every observing context (\"SV-AA2\", \"SV-AA*\", \"Cycle 0\", \"Cycle 1\"), to see at which array-assembly stage or cycle a paper's requirement becomes feasible.",
   "params": [
    "obs_config",
    "contexts"
   ]
  },
  {
   "name": "resolve_object_tool",
   "kind": "tool",
   "summary": "Resolve an astronomical object name via SIMBAD (falling back to NED).",
   "params": [
    "name"
   ]
  },
  {
   "name": "ned_lookup_tool",
   "kind": "tool",
   "summary": "Look up an extragalactic object in NED: position, NED type, redshift and recession velocity. Prefer resolve_object_tool for general use; use this when the NED redshift/velocity specifically matters.",
   "params": [
    "name"
   ]
  },
  {
   "name": "cone_search_tool",
   "kind": "tool",
   "summary": "List SIMBAD objects within a radius of a target, nearest first.",
   "params": [
    "target",
    "radius_arcmin",
    "max_results",
    "object_type"
   ]
  },
  {
   "name": "build_source_db_tool",
   "kind": "tool",
   "summary": "Build or update the database of astronomical sources named in the indexed papers (regex-extracted, resolved via SIMBAD).",
   "params": [
    "retry_unresolved"
   ]
  },
  {
   "name": "query_sources_tool",
   "kind": "tool",
   "summary": "List sources named in the papers, most-cited first, with position (RA/Dec, Galactic), SIMBAD type, redshift, the papers that mention each, and the names it appears under.",
   "params": [
    "object_type",
    "book",
    "section",
    "paper",
    "dec_min",
    "dec_max",
    "min_papers",
    "limit"
   ]
  },
  {
   "name": "get_source_mentions_tool",
   "kind": "tool",
   "summary": "A source and every passage in the papers that mentions it, under any of its names (e.g. \"Cen A\" also finds \"NGC 5128\" and \"Centaurus A\"), with paper, book, section, page and surrounding text.",
   "params": [
    "name",
    "limit"
   ]
  },
  {
   "name": "list_unresolved_sources_tool",
   "kind": "tool",
   "summary": "Names found in the papers that SIMBAD couldn't resolve, most-mentioned first: survey fields, truncated designations, typos or regex false positives. Candidates for a curated field list.",
   "params": [
    "limit"
   ]
  },
  {
   "name": "build_citation_db_tool",
   "kind": "tool",
   "summary": "Rebuild the database of references cited by the indexed papers, parsed from each paper's reference list. Offline, takes ~10 s. Returns counts and the papers where no references were parsed.",
   "params": []
  },
  {
   "name": "top_cited_papers_tool",
   "kind": "tool",
   "summary": "Works most cited by the corpus, ranked by how many corpus papers cite each, with an example reference string, DOI/arXiv id and the citing papers.",
   "params": [
    "book",
    "section",
    "since_year",
    "min_papers",
    "limit"
   ]
  },
  {
   "name": "citing_papers_tool",
   "kind": "tool",
   "summary": "Which corpus papers cite a given work: searches every reference string (case-insensitive substring) and groups matches by cited work.",
   "params": [
    "text",
    "limit"
   ]
  },
  {
   "name": "facility_network_tool",
   "kind": "tool",
   "summary": "Which external facilities (Rubin/LSST, JWST, ngVLA, CTAO, LIGO, IceCube, ALMA, \u2026; about 80, curated in corpus_tools.FACILITIES) the papers name, across how many sections and papers, and which are named together.",
   "params": [
    "book",
    "min_hits_per_paper",
    "min_shared_papers"
   ]
  },
  {
   "name": "verify_citation_tool",
   "kind": "tool",
   "summary": "Check that a paper's reference list really contains a work it is said to cite, e.g. citation=\"Vacca et al. 2025\". Matches first author and year against the citation database; also lists that author's other cited years (a likely year mix-up). Complements verify_quote_tool, which checks which corpus paper a figure comes from.",
   "params": [
    "paper",
    "citation",
    "book"
   ]
  },
  {
   "name": "visibility_windows_tool",
   "kind": "tool",
   "summary": "When a target can be observed from SKA-Low or SKA-Mid over a year.",
   "params": [
    "target",
    "telescope",
    "min_elevation",
    "sun",
    "year_start"
   ]
  },
  {
   "name": "lst_pressure_tool",
   "kind": "tool",
   "summary": "Requested vs available hours per LST bin (and month \u00d7 LST) for a set of observing requests: which LST ranges are oversubscribed.",
   "params": [
    "telescope",
    "requests",
    "from_db",
    "year_start",
    "extracted_by",
    "reviewed_only",
    "years"
   ]
  },
  {
   "name": "add_observing_request_tool",
   "kind": "tool",
   "summary": "Record an observing request stated in a paper (for lst_pressure_tool).",
   "params": [
    "name",
    "hours",
    "telescope",
    "paper",
    "quote",
    "book",
    "ra",
    "dec",
    "ra_range",
    "dec_range",
    "band",
    "freq_mhz",
    "sun",
    "min_elevation",
    "commensal_group",
    "extracted_by",
    "position_note",
    "field",
    "gal_l_range",
    "gal_b_range"
   ]
  },
  {
   "name": "add_survey_field_tool",
   "kind": "tool",
   "summary": "Record a survey field or area as a paper states it: a named field (COSMOS, ELAIS-N1, an EoR field) and/or an area defined only by its characteristics (e.g. \"20,000 deg2\", \"the southern sky\", \"2% of the sky\", \"the Galactic plane, |b| < 5 deg\"). Record only stated values. Refused unless quote appears verbatim in the paper.",
   "params": [
    "paper",
    "quote",
    "name",
    "description",
    "book",
    "ra",
    "dec",
    "ra_range",
    "dec_range",
    "gal_l_range",
    "gal_b_range",
    "area_deg2",
    "extracted_by"
   ]
  },
  {
   "name": "list_survey_fields_tool",
   "kind": "tool",
   "summary": "Survey fields recorded from the papers: how many papers name each, stated area, and best known position with where it came from.",
   "params": []
  },
  {
   "name": "list_observing_requests_tool",
   "kind": "tool",
   "summary": "Observing requests recorded from the papers, each with its id, quote, paper, page and any review decision (review overrides applied).",
   "params": [
    "telescope",
    "extracted_by",
    "include_rejected"
   ]
  },
  {
   "name": "review_observing_request_tool",
   "kind": "tool",
   "summary": "Record a review decision on an extracted request. The extraction is kept unchanged; the decision and any corrections are stored beside it and applied when requests are listed or fed to lst_pressure_tool.",
   "params": [
    "request_id",
    "status",
    "note",
    "hours",
    "commensal_group",
    "field",
    "duplicate_of",
    "reviewed_by"
   ]
  },
  {
   "name": "atlas_taxonomies",
   "kind": "resource",
   "summary": "The 15 observing-mode categories and 16 technique keywords the AASKAII Atlas fixed before searching; reuse them so repeated analyses stay comparable.",
   "params": []
  }
 ]
};
