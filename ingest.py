"""
CLI ingestion script.

Usage:
    python ingest.py                        # index all configured directories
    python ingest.py path/to/dir [more...]  # index specific directories only
    python ingest.py --reindex              # wipe ChromaDB and reindex everything from scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import PDF_DIRS
from rag.ingestion import ingest_files


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
    args = parser.parse_args()

    dirs = args.dirs if args.dirs else PDF_DIRS

    print("Ingestion directories:")
    for d in dirs:
        status = "✓" if d.exists() else "✗ (not found)"
        print(f"  {d}  {status}")
    print()

    results = ingest_files(dirs, reset=args.reindex)

    if not results:
        print("No new files were indexed.")
        print("Either no supported files were found, or all files are already indexed.")
        sys.exit(0)

    total_chunks = sum(results.values())
    print(f"\nIngestion complete — {len(results)} new file(s), {total_chunks} total chunks.")
    print("Run `python app.py` to start the web UI.")


if __name__ == "__main__":
    main()
