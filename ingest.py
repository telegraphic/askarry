"""
CLI ingestion script.

Usage:
    python ingest.py                        # index all configured directories
    python ingest.py path/to/dir [more...]  # index specific directories only
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag.config import PDF_DIRS
from rag.ingestion import ingest_files


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else PDF_DIRS

    print("Ingestion directories:")
    for d in dirs:
        status = "✓" if d.exists() else "✗ (not found)"
        print(f"  {d}  {status}")
    print()

    results = ingest_files(dirs)

    if not results:
        print("No new files were indexed.")
        print("Either no supported files were found, or all files are already indexed.")
        sys.exit(0)

    total_chunks = sum(results.values())
    print(f"\nIngestion complete — {len(results)} new file(s), {total_chunks} total chunks.")
    print("Run `python app.py` to start the web UI.")


if __name__ == "__main__":
    main()
