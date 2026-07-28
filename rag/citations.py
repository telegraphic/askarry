"""
Citation-marker post-processing for generated answers.

Pulled out of app.py so this pure text-transformation logic (no Gradio, no
I/O) can be unit tested independently of the UI layer.
"""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[\s*(\d+)\s*\]")


def linkify_citations(answer: str, chunks: list[dict]) -> tuple[str, set[int]]:
    """Turn `[n]` markers in the final answer into clickable links to the
    matching Source Passages entry, and flag citation numbers that don't
    correspond to any retrieved chunk.

    Only structural validation is performed (does `[n]` match a retrieved
    passage?) — not whether the passage actually supports the claim.

    Returns the linkified answer plus the set of valid cited chunk numbers,
    so callers can report passages the model never referenced.
    """
    if not chunks:
        return answer, set()

    cited: set[int] = set()

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(chunks):
            cited.add(n)
            return f'<a href="#source-{n}" class="citation-link">[{n}]</a>'
        return (
            f'<span class="citation-invalid" '
            f'title="No retrieved source matches [{n}]">[{n}]⚠️</span>'
        )

    return _CITATION_RE.sub(_replace, answer), cited


def build_uncited_note(chunks: list[dict], cited: set[int]) -> str:
    """Return a small note listing retrieved passages the model never cited,
    or "" if every passage was referenced (or there are none)."""
    uncited = [i for i in range(1, len(chunks) + 1) if i not in cited]
    if not uncited:
        return ""
    labels = ", ".join(f"[{i}]" for i in uncited)
    return (
        f'\n\n<p class="text-sm opacity-70">Not referenced in the answer: '
        f'{labels}</p>'
    )
