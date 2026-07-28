"""Ensure the repo root is importable as `rag` regardless of how pytest is
invoked (e.g. `pytest` vs `python -m pytest`, or from a subdirectory)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
