import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
PDF_DIR = Path("pdfs")                              # Local PDFs folder
CHROMA_DIR = Path("chroma_db")                      # Persistent ChromaDB storage
BIBLIOGRAPHY_PATH = PDF_DIR / "bibliography.json"   # AASKAII title/author lookup (see pdfs/build_bibliography.py)
AASKAII_YEAR = 2026                                  # Publication year for all AASKAII chapters

# All directories to scan during ingestion (edit freely).
# Non-existent paths are silently skipped.
PDF_DIRS: list[Path] = [PDF_DIR]

# File extensions handled throughout the pipeline (ingestion, bibliography
# lookup, the pdfs/ download & bibliography-build tooling).
SUPPORTED_SUFFIXES: set[str] = {".pdf", ".html", ".htm"}

# Book section names, in book order, shared by rag/bibliography.py (Table of
# Contents grouping) and pdfs/download.py (download folder organisation).
SECTION_ORDER: list[str] = [
    "Science Working Group Overviews",
    "Sun, Earth and Planets",
    "Formation and Evolution of Stars",
    "From the Milky Way to Distant Galaxies",
    "The Cosmos",
    "The Extreme Universe",
    "Methods and Techniques",
]

# ---------------------------------------------------------------------------
# Embedding model  (sentence-transformers, runs locally)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# BGE models need a short instruction prefix on queries (not on indexed docs)
EMBEDDING_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Re-ranker  (cross-encoder, runs locally, no API key needed)
# ---------------------------------------------------------------------------
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "gemma4"          # Change to any model pulled via `ollama pull`
OLLAMA_BASE_URL = "http://localhost:11434"
# Context window (tokens) requested from Ollama. The default context window
# for most models loaded via Ollama is only 2048-4096 tokens, which silently
# truncates the retrieved passages on the *left* (oldest messages first) once
# TOP_K chunks + context-window expansion exceed it — this is a common cause
# of answers that ignore half the retrieved evidence. Size this comfortably
# above (TOP_K * ~3 chunks * MAX_CHUNK_TOKENS) + question + answer headroom.
# gemma4 (e4b, the default `latest` tag) supports up to 128K tokens — 8192 is
# a conservative middle ground for latency; raise it if answers still seem to
# be missing context from later passages.
OLLAMA_NUM_CTX = 8192
# Google's documented sampling recipe for Gemma 4 is temperature=1.0,
# top_p=0.95, top_k=64 — the model is tuned/calibrated around this setting,
# so lowering temperature to try to reduce hallucination tends to make
# answers worse, not better. Grounding is enforced via the system prompt
# instructions instead (see generation.py).
OLLAMA_TEMPERATURE = 1.0
OLLAMA_TOP_P = 0.95
OLLAMA_TOP_K = 64

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5                  # Final chunks passed to the LLM
RETRIEVAL_CANDIDATES = 20  # Broad first-stage fetch (per method) before fusion & re-ranking
# Cross-encoder relevance (sigmoid of the reranker logit, 0-1) below which the
# top retrieved chunk is considered a weak match. Used to warn the user in the
# UI and to nudge the LLM to hedge rather than confidently answer from
# marginally-relevant passages. Tune by inspecting real "relevance: NN%"
# values shown in the Source Passages panel for queries that are/aren't
# actually answered by the corpus.
CONFIDENCE_THRESHOLD = 0.3
MAX_CHUNK_TOKENS = 480     # Conservative: BERT max is 512 but needs [CLS]/[SEP] headroom;
                           # HybridChunker sometimes overshoots so we stay well below the limit

# --- Parallel ingestion ---
# Number of worker processes for the parse+chunk stage (the slow part).
# Each worker loads its own DocumentConverter, so RAM usage scales with this.
# Docling uses PyTorch internally — on Apple Silicon it may use Metal, in which
# case more than 2 workers rarely helps and can increase contention.
# Set to 1 to disable parallelism (useful for debugging or low-RAM machines).
INGEST_WORKERS: int = min(os.cpu_count() or 2, 4)

# --- Hybrid retrieval ---
# BM25_WEIGHT controls how much the keyword score contributes to Reciprocal
# Rank Fusion.  0.0 = pure semantic; 1.0 = equal weight for both legs.
# SKA docs are full of acronyms & part numbers so 0.5 is a good starting point.
BM25_WEIGHT = 0.5

# --- HyDE (Hypothetical Document Embeddings) ---
# When True the LLM generates a short hypothetical answer to the query, which
# is then embedded instead of (or alongside) the raw question.  Improves
# recall for highly technical queries at the cost of one fast Ollama call.
USE_HYDE = False

# --- Acronym expansion ---
# Headings (case-insensitive substring match) that signal an acronym/glossary
# section. Chunks under these headings are tagged chunk_type="acronyms" at
# ingest time and used to expand BM25 queries with full forms of abbreviations.
ACRONYM_HEADINGS: list[str] = [
    "acronym",
    "abbreviation",
    "glossary",
    "list of terms",
    "definitions",
]

# Curated fallback acronyms for the SKA/radio-astronomy domain. Most AASKAII
# papers are journal articles with no dedicated glossary/acronym section, so
# relying solely on ACRONYM_HEADINGS-tagged chunks (see rag/retrieval.py::
# _get_acronyms) yields little to nothing. This seed list is merged in for
# any acronym not already discovered in the indexed documents (documents win
# on conflicts, since they're more likely to match the terminology actually
# used in a given corpus).
# Keys must be all-uppercase (letters/digits/-/.) to match the token regex
# _expand_query uses to spot acronyms in a user's query — mixed-case forms
# like "FoV" or plurals like "PTAs" would never be matched, so are omitted.
SEED_ACRONYMS: dict[str, str] = {
    "SKA": "Square Kilometre Array",
    "SKAO": "SKA Observatory",
    "AA0.5": "Array Assembly 0.5",
    "AA1": "Array Assembly 1",
    "AA2": "Array Assembly 2",
    "AA4": "Array Assembly 4",
    "CSP": "Central Signal Processor",
    "SDP": "Science Data Processor",
    "RFI": "Radio Frequency Interference",
    "FOV": "Field of View",
    "SNR": "Signal-to-Noise Ratio",
    "HI": "Neutral Atomic Hydrogen",
    "AGN": "Active Galactic Nucleus",
    "FRB": "Fast Radio Burst",
    "GW": "Gravitational Wave",
    "CMB": "Cosmic Microwave Background",
    "ISM": "Interstellar Medium",
    "IGM": "Intergalactic Medium",
    "PSR": "Pulsar",
    "NS": "Neutron Star",
    "BH": "Black Hole",
    "QSO": "Quasi-Stellar Object (Quasar)",
    "LOFAR": "Low-Frequency Array",
    "VLBI": "Very Long Baseline Interferometry",
    "EOR": "Epoch of Reionization",
    "PTA": "Pulsar Timing Array",
    "LISA": "Laser Interferometer Space Antenna",
    "SED": "Spectral Energy Distribution",
    "CME": "Coronal Mass Ejection",
    "YSO": "Young Stellar Object",
    "GMC": "Giant Molecular Cloud",
    "WD": "White Dwarf",
    "GRB": "Gamma-Ray Burst",
    "DM": "Dispersion Measure",
    "RM": "Rotation Measure",
    "FFT": "Fast Fourier Transform",
    "FPGA": "Field-Programmable Gate Array",
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
}

# ---------------------------------------------------------------------------
# ChromaDB collection name
# ---------------------------------------------------------------------------
COLLECTION_NAME = "astronomy"

# ---------------------------------------------------------------------------
# Answer audience
# ---------------------------------------------------------------------------
# Internal audience keys, shared between rag/generation.py (system-prompt
# instructions) and app.py (UI radio button mapping) so both sides reference
# the same constants instead of matching raw string literals by convention.
AUDIENCE_PHD = "phd"
AUDIENCE_GENERAL = "general"
AUDIENCE_UI_LABELS: dict[str, str] = {
    "PhD astronomer": AUDIENCE_PHD,
    "Non-expert": AUDIENCE_GENERAL,
}
