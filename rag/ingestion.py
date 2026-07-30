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
import re
import shutil
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chromadb
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


# ---------------------------------------------------------------------------
# Acronym discovery
# ---------------------------------------------------------------------------
# The heading-based tagging above (chunk_type="acronyms") only catches
# dedicated glossary/acronym-table sections, which most journal-article PDFs
# simply don't have. In practice, scientific writing instead defines an
# acronym inline on first use — e.g. "the Central Signal Processor (CSP)" or
# "Field of View (FoV)". This scans every indexed chunk for that pattern to
# build a candidate list for a human to review and copy into
# config.SEED_ACRONYMS — it is not wired into retrieval automatically since
# the heuristic isn't perfect (ambiguous/ wrong expansions are possible).

# Matches "<1-6 preceding words> (CANDIDATE)" e.g. "Central Signal Processor (CSP)"
_INLINE_ACRONYM_RE = re.compile(r"((?:[A-Za-z][\w\-]*\s+){1,6})\(([A-Za-z][A-Za-z0-9\-]{1,9})\)")


def _split_words(phrase: str) -> list[str]:
    """Split an expansion phrase into initial-bearing words, treating both
    whitespace *and* hyphens as boundaries.

    Scientific writing routinely hyphenates compound expansions — e.g.
    "Gamma-Ray Burst (GRB)" — but `str.split()` alone would only see 2 tokens
    ("Gamma-Ray", "Burst") for a 3-letter acronym, so the initials-matching in
    `_acronym_expansion` below could never line up and the whole definition
    was silently dropped. Splitting on hyphens too yields 3 tokens (Gamma,
    Ray, Burst) that correctly match "GRB".
    """
    return [w for w in re.split(r"[\s\-]+", phrase.strip()) if w]


def _acronym_expansion(words: list[str], acronym: str) -> str | None:
    """Return the expansion phrase if the last len(acronym) words' initials
    spell out *acronym* (case-insensitively), else None."""
    n = len(acronym)
    if len(words) < n:
        return None
    window = words[-n:]
    initials = "".join(w[0] for w in window)
    if initials.lower() == acronym.lower():
        return " ".join(window)
    return None


def _strip_plural(acronym: str) -> str:
    """Normalize an inline-plural acronym like "GRBs" or "PTAs" back to its
    base form ("GRB", "PTA").

    Papers routinely pluralize acronyms in running text ("...several GRBs
    were observed..."). Without this, "GRBs" would (a) be treated as a
    4-letter acronym distinct from "GRB" and (b) fail `_acronym_expansion`
    outright, since the trailing lowercase "s" has no corresponding word in
    the expansion phrase — the definition was silently dropped instead of
    counting towards "GRB". Only strips a trailing lowercase "s" following an
    otherwise all-uppercase acronym, so genuine acronyms that happen to end
    in a capital S (e.g. "SDSS") are left untouched.
    """
    if len(acronym) > 2 and acronym[-1] == "s" and acronym[:-1].isupper():
        return acronym[:-1]
    return acronym


def _strip_accents(text: str) -> str:
    """Strip diacritics, e.g. "Farèse"/"Farésé" both -> "Farese". Accented
    letters in transliterated names are frequently inconsistent (OCR
    artifacts, author typos, different house styles) across papers, so
    without this a single person's name could fragment an acronym like DEF
    into multiple spurious "different" candidates."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# Irregular Latin/Greek plurals that don't follow the regular "-s"/"-ies"
# rules below (e.g. "nuclei" is not "nucleu" + s). Keyed by plural form.
_IRREGULAR_PLURALS: dict[str, str] = {
    "nuclei": "nucleus",
    "radii": "radius",
    "foci": "focus",
    "axes": "axis",
    "crises": "crisis",
    "bases": "basis",
    "analyses": "analysis",
    "theses": "thesis",
    "criteria": "criterion",
    "phenomena": "phenomenon",
}


def _singularize(word: str) -> str:
    """Fold a plural word to its singular form: irregular Latin/Greek
    plurals ("nuclei" -> "nucleus") via `_IRREGULAR_PLURALS`, "-ies" -> "y"
    ("galaxies" -> "galaxy", "assemblies" -> "assembly", "instabilities" ->
    "instability", "energies" -> "energy"), and the regular trailing "s"
    ("bursts" -> "burst"). Words ending in "-us"/"-is" (nucleus, radius,
    axis, analysis, ...) are excluded from the regular "-s" rule since
    they're already singular despite ending in a literal "s".
    """
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith("ss")
        and not word.endswith("us")
        and not word.endswith("is")
    ):
        return word[:-1]
    return word


def _normalize_expansion(expansion: str) -> str:
    """Canonical form of an expansion phrase, used to decide whether two
    differently-worded phrases mean the same thing so their occurrence counts
    should be merged.

    Accents, case, hyphen-vs-space, British/American spelling, plural/
    singular (including irregular Latin/Greek plurals), "-ment" nominalization,
    and "-ic" adjective-vs-noun differences are all ignored — e.g. "Gamma Ray
    Bursts", "Gamma-Ray Burst" and "gamma-ray burst" all normalize to "gamma
    ray burst"; "equations of state" / "equation of state" both normalize to
    "equation of state"; "Square Kilometre Array" / "Square Kilometer Array"
    both normalize to "square kilometer array" (see `_normalize_spelling`);
    "active galactic nuclei" / "active galactic nucleus" both normalize to
    "active galactic nucleus"; "rotation measurement" normalizes towards
    "rotation measure"; and "Baryonic Acoustic Oscillations" / "Baryon
    Acoustic Oscillation" both normalize to "baryon acoust oscillation".
    Genuinely different meanings (e.g. "dispersion measure" vs "dark
    matter") still normalize differently, so they are *not* merged here —
    see `find_acronym_candidates` / `_senses_are_same` for the further
    single-word-swap clustering pass (e.g. EIRP's "effective"/"equivalent",
    ATCA's "Australia"/"Australian", CTA's "Cerenkov"/"Cherenkov").
    """
    words = re.split(r"[\s\-]+", _strip_accents(expansion).lower().strip())
    normalized = []
    for word in words:
        word = _normalize_spelling(word)
        word = _singularize(word)
        if len(word) > 5 and word.endswith("ment"):
            word = word[:-4]
        if len(word) > 5 and word.endswith("ic"):
            word = word[:-2]
        normalized.append(word)
    return " ".join(w for w in normalized if w)


# British → American spelling variants seen in scientific writing (e.g.
# "kilometre" vs "kilometer", "polarised" vs "polarized"). Suffixes, so they
# also match compounds like "kilometre"/"centimetre". Ordered longest-first
# so e.g. "isation" is matched before the shorter "ise" would otherwise
# apply to the wrong part of the word.
_SPELLING_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("behaviour", "behavior"),
    ("neighbour", "neighbor"),
    ("isation", "ization"),
    ("flavour", "flavor"),
    ("colour", "color"),
    ("centre", "center"),
    ("metre", "meter"),
    ("litre", "liter"),
    ("ising", "izing"),
    ("ised", "ized"),
    ("ogue", "og"),
    ("ise", "ize"),
)


def _normalize_spelling(word: str) -> str:
    """Fold a British spelling suffix to its American form, e.g. "kilometre"
    -> "kilometer", "organised" -> "organized". Requires either an exact
    match (root_len == 0, e.g. "colour" -> "color") or at least 3 leading
    characters before the suffix (root_len >= 3) so short unrelated words
    that merely end the same way — "wise", "rise", "noise" — aren't
    mis-transformed."""
    for british, american in _SPELLING_SUFFIXES:
        if not word.endswith(british):
            continue
        root_len = len(word) - len(british)
        if root_len == 0 or root_len >= 3:
            return word[:root_len] + american
    return word


# ---------------------------------------------------------------------------
# Acronym categorization
# ---------------------------------------------------------------------------
# Best-effort keyword classifier so the Acronyms UI tab can group/filter
# entries (e.g. "Telescopes & Facilities" vs "Science & Astrophysics").
# Categories are checked in order and the first keyword match wins, so more
# specific categories are listed first — e.g. "Convolutional Neural Network"
# must be caught by the "neural network" keyword before the generic
# "network" keyword (meant for telescope/VLBI networks) gets a chance.
CATEGORY_SCIENCE = "Science & Astrophysics"
_ACRONYM_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Methods & Software", (
        "algorithm", "neural network", "machine learning", "deep learning",
        "learning", "regression", "estimator", "analysis", "technique",
        "transform", "simulation", "processing", "pipeline", "software",
        "code", "criterion", "classifier", "clustering",
    )),
    ("Organizations & Programs", (
        "university", "institute", "council", "foundation", "agency",
        "school", "center", "centre", "consortium", "programme", "program",
    )),
    ("Instruments & Hardware", (
        "processor", "converter", "receiver", "amplifier", "detector",
        "camera", "instrument", "polarimeter", "spectrometer", "correlator",
        "antenna", "feed", "field-programmable", "gate array",
    )),
    ("Telescopes & Facilities", (
        "telescope", "array", "observatory", "interferometer",
        "interferometry", "network", "dish",
    )),
]
# Exposed for UI category filters — includes the CATEGORY_SCIENCE fallback,
# which in practice is the largest bucket (astrophysical objects/phenomena
# rarely share a consistent keyword to match on).
ACRONYM_CATEGORIES: list[str] = [c for c, _ in _ACRONYM_CATEGORY_KEYWORDS] + [CATEGORY_SCIENCE]


def classify_acronym(expansion: str) -> str:
    """Best-effort category for an acronym based on keywords found in its
    expansion text. This is a heuristic aid for browsing/filtering, not an
    authoritative taxonomy — falls back to CATEGORY_SCIENCE."""
    text = expansion.lower()
    for category, keywords in _ACRONYM_CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return CATEGORY_SCIENCE


def _senses_are_same(norm_a: str, norm_b: str) -> bool:
    """Decide whether two *already word-normalized* expansions for the same
    acronym letters are the same underlying meaning, expressed with one
    word swapped for a near-synonym/alternate term/typo, rather than a
    genuinely different meaning.

    Requires the same word count with at most one differing position, e.g.:
      - "effective isotropic radiated power" vs "equivalent isotropic
        radiated power" (EIRP — both terms are used interchangeably)
      - "expanded owens valley solar array" vs "extended owens valley solar
        array" (EOVSA)
      - "laser interferometer space antenna" vs "laser interferometry space
        antenna" (LISA)
      - "cerenkov telescope array" vs "cherenkov telescope array" (CTA —
        alternate transliteration)
      - "australia telescope compact array" vs "australian telescope
        compact aarray" (ATCA — demonym variant *and* a typo, still just one
        differing word each pairwise)
      - "global navigation satellite system" vs "global navigation satellite
        signal" (GNSS); "gigahertz peaked spectrum" vs "ghz peaked spectrum"
        (GPS)

    Genuinely different meanings sharing an acronym (e.g. DM = "dispersion
    measure" vs "dark matter", or GPS = the above vs "global positioning
    system") differ in *every* word position, so they are correctly kept
    separate — this is deliberately conservative (single-word-swap only) to
    avoid merging unrelated concepts that merely share a couple of words.
    """
    words_a, words_b = norm_a.split(), norm_b.split()
    if not words_a or len(words_a) != len(words_b):
        return False
    diff = sum(1 for a, b in zip(words_a, words_b) if a != b)
    return diff <= 1


def _cluster_senses(senses: dict[str, dict]) -> list[dict]:
    """Merge normalized-expansion buckets (see `find_acronym_candidates`)
    that `_senses_are_same()` considers the same underlying meaning, summing
    their raw-text occurrence counts and sources together.

    A new normalized form joins the first existing cluster where it matches
    *any* previously-clustered member (not just that cluster's original
    representative), so transitive chains merge correctly — e.g. for ATCA,
    "Australian Telescope Compact Array" merges into "Australia Telescope
    Compact Array" (demonym swap), and "Australian Telescope Compact Aarray"
    (typo) then merges too because it's a one-word swap *from the
    "Australian ..." member*, even though it differs from the original
    "Australia ..." representative in two words at once.
    """
    clustered: list[dict] = []
    for norm, bucket in senses.items():
        target = next(
            (c for c in clustered if any(_senses_are_same(norm, m) for m in c["norms"])),
            None,
        )
        if target is None:
            clustered.append(
                {
                    "norms": [norm],
                    "counts": dict(bucket["counts"]),
                    "sources": set(bucket["sources"]),
                }
            )
        else:
            target["norms"].append(norm)
            for raw_text, n in bucket["counts"].items():
                target["counts"][raw_text] = target["counts"].get(raw_text, 0) + n
            target["sources"] |= bucket["sources"]
    return clustered


def find_acronym_candidates(collection: chromadb.Collection) -> dict[str, dict]:
    """Scan every indexed chunk for inline acronym definitions.

    Returns {KEY: {"expansions": {text: count}, "sources": [absolute source
    paths], "category": str}}, sorted by nothing in particular — callers
    should rank by count. Sources are full paths (as stored in chunk
    metadata) rather than bare filenames so callers can build file links /
    bibliography lookups without re-resolving relative paths; derive a
    display name at render time. `category` is assigned via
    `classify_acronym()` from the most common expansion seen. Intended for
    manual review (e.g. via `python ingest.py --list-acronyms`) or the
    Acronyms UI tab, not for silently feeding retrieval.

    Different phrasings of the *same* meaning are merged so their counts add
    up — e.g. "Gamma Ray Burst", "Gamma-Ray Bursts" and "gamma-ray burst" all
    count towards one "GRB" entry (see `_normalize_expansion`), and plurals
    like "GRBs" are folded into "GRB" (see `_strip_plural`). A further
    clustering pass (see `_senses_are_same`) merges expansions that differ by
    only one swapped word — near-synonyms, alternate transliterations, or
    typos — e.g. EIRP's "effective"/"equivalent", ATCA's
    "Australia"/"Australian" (plus a stray "Aarray" typo), CTA's
    "Cerenkov"/"Cherenkov". Conversely, when the same letters have genuinely
    different meanings in the corpus (e.g. "DM" = Dispersion Measure vs.
    "DM" = Dark Matter — different in *every* word, not just one), each
    meaning is kept as its *own* entry — key `"DM (Dispersion Measure)"`,
    `"DM (Dark Matter)"` — with its own separate occurrence/paper counts,
    instead of being silently summed together into one misleading total.
    """
    # acronym -> normalized expansion -> {"counts": {raw text: count}, "sources": {..}}
    raw: dict[str, dict[str, dict]] = {}
    all_results = collection.get(include=["documents", "metadatas"])

    for text, meta in zip(all_results["documents"], all_results["metadatas"] or []):
        for match in _INLINE_ACRONYM_RE.finditer(text):
            phrase, raw_acronym = match.group(1), match.group(2)
            # Require a plausible acronym: mostly uppercase, e.g. reject "(shown)"
            if sum(1 for c in raw_acronym if c.isupper()) < len(raw_acronym) - 1:
                continue
            acronym = _strip_plural(raw_acronym)
            words = _split_words(phrase)
            expansion = _acronym_expansion(words, acronym)
            if expansion is None:
                continue

            key = acronym.upper()
            norm = _normalize_expansion(expansion)
            bucket = raw.setdefault(key, {}).setdefault(
                norm, {"counts": {}, "sources": set()}
            )
            bucket["counts"][expansion] = bucket["counts"].get(expansion, 0) + 1
            bucket["sources"].add(meta["source"])

    candidates: dict[str, dict] = {}
    for acronym, senses in raw.items():
        ranked_senses = sorted(
            _cluster_senses(senses), key=lambda bucket: -sum(bucket["counts"].values())
        )
        disambiguate = len(ranked_senses) > 1
        for bucket in ranked_senses:
            best_expansion = max(bucket["counts"], key=bucket["counts"].get)
            display_key = f"{acronym} ({best_expansion})" if disambiguate else acronym
            candidates[display_key] = {
                "expansions": bucket["counts"],
                "sources": sorted(bucket["sources"]),
                "category": classify_acronym(best_expansion),
            }

    return candidates
