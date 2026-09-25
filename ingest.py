"""
CLI ingestion script.

Usage:
    python ingest.py                        # index all configured directories
    python ingest.py path/to/dir [more...]  # index specific directories only
    python ingest.py --reindex              # wipe ChromaDB and reindex everything from scratch
    python ingest.py --textbooks            # index pdfs/textbooks into its own store (chroma_textbooks/)
    python ingest.py --textbooks --reindex  # wipe and rebuild only the textbook store
    python ingest.py --list-acronyms        # scan indexed docs for inline acronym definitions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.config import CHROMA_DIR, PDF_DIRS, TEXTBOOKS_CHROMA_DIR, TEXTBOOKS_DIR
from rag.ingestion import find_acronym_candidates, ingest_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs/HTML into ChromaDB.")
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        help="Specific directories to index (default: all configured directories)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete the existing ChromaDB store and reindex all files from scratch",
    )
    parser.add_argument(
        "--textbooks",
        action="store_true",
        help="Index pdfs/textbooks into the separate textbook ChromaDB store "
             "instead of the main one (--reindex then wipes only that store)",
    )
    parser.add_argument(
        "--list-acronyms",
        nargs="?",
        const="-",
        default=None,
        metavar="OUTPUT_JSON",
        help="Scan already-indexed documents for inline acronym definitions "
             "(e.g. 'Central Signal Processor (CSP)') and print a candidate "
             "list for review — copy useful entries into "
             "rag/config.py::SEED_ACRONYMS. Optionally pass a file path to "
             "also save the full results as JSON.",
    )
    args = parser.parse_args()

    if args.list_acronyms is not None:
        list_acronyms(args.list_acronyms)
        return

    if args.textbooks:
        dirs = args.dirs or [TEXTBOOKS_DIR]
        db_dir = TEXTBOOKS_CHROMA_DIR
    else:
        dirs = args.dirs or PDF_DIRS
        db_dir = CHROMA_DIR

    print("Ingestion directories:")
    for d in dirs:
        status = "✓" if d.exists() else "✗ (not found)"
        print(f"  {d}  {status}")
    print()

    results = ingest_files(dirs, reset=args.reindex, db_dir=db_dir)

    if not results:
        print("No new files were indexed.")
        print("Either no supported files were found, or all files are already indexed.")
        sys.exit(0)

    total_chunks = sum(results.values())
    print(f"\nIngestion complete — {len(results)} new file(s), {total_chunks} total chunks.")
    print("Run `python app.py` to start the web UI.")


def list_acronyms(output_path: str) -> None:
    """Scan the existing ChromaDB collection for inline acronym definitions
    and print a ranked candidate list (see `--list-acronyms` help text)."""
    from rag import store

    collection = store.get_chroma_collection()
    if collection.count() == 0:
        print("ChromaDB collection is empty — run ingestion first.")
        sys.exit(1)

    candidates = find_acronym_candidates(collection)
    if not candidates:
        print("No inline acronym definitions found.")
        return

    def total_count(entry: dict) -> int:
        return sum(entry["expansions"].values())

    print(f"Found {len(candidates)} acronym candidate(s):\n")
    for acronym, entry in sorted(candidates.items(), key=lambda kv: -total_count(kv[1])):
        best_expansion = max(entry["expansions"], key=entry["expansions"].get)
        n_files = len(entry["sources"])
        print(
            f"  {acronym:<28} [{entry['category']}] {best_expansion}  "
            f"({total_count(entry)}x across {n_files} file(s))"
        )
        # Flag ambiguous acronyms with more than one distinct expansion seen
        if len(entry["expansions"]) > 1:
            others = [e for e in entry["expansions"] if e != best_expansion]
            print(f"    also seen as: {', '.join(others)}")

    if output_path != "-":
        Path(output_path).write_text(json.dumps(candidates, indent=2, sort_keys=True))
        print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()
