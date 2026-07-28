"""
Gradio web UI for the SKA RAG system.

Usage:
    python app.py
"""

from __future__ import annotations

import math
import re
import threading
import warnings
from html import escape
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import ollama

from rag.bibliography import format_citation, list_toc_entries, lookup_citation
from rag.config import CONFIDENCE_THRESHOLD, OLLAMA_MODEL, PDF_DIRS, TOP_K
from rag.generation import stream_answer
from rag.retrieval import retrieve, warmup

# Pre-load models in the background so the first query doesn't block the UI.
threading.Thread(target=warmup, daemon=True, name="rag-warmup").start()


# ---------------------------------------------------------------------------
# Core query function
# ---------------------------------------------------------------------------


def _display_source_name(source: str) -> str:
    """Return a filename-only label for a stored source path."""
    return Path(source).name or source


def _gradio_file_url(path: Path) -> str:
    """Build a Gradio-served HTTP URL for a local file."""
    return f"/gradio_api/file={quote(str(path), safe='/')}"


def _source_links(source: str) -> str:
    """Render a Gradio-served HTTP link to the source file."""
    source_path = Path(source).expanduser()

    citation = lookup_citation(source)
    label = escape(format_citation(citation)) if citation else escape(_display_source_name(source))

    try:
        resolved = source_path.resolve()
    except OSError:
        return label

    file_link = (
        f'<a href="{_gradio_file_url(resolved)}" target="_blank" '
        f'rel="noopener noreferrer">{label}</a>'
    )
    return file_link


_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including embedded newlines) into a
    single space.

    Docling-extracted chunk text often contains raw layout artifacts from the
    source PDF — multi-line author/affiliation lists, blank lines, indented
    columns, etc. Left as-is, a single leading `> ` blockquote marker only
    covers the excerpt's first line: any embedded newline drops the rest of
    the text out of the blockquote, and runs of leading whitespace on those
    lines get reinterpreted by Markdown as an indented code block. Collapsing
    to one line keeps the excerpt a single clean blockquote paragraph.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def _build_sources_markdown(chunks: list[dict]) -> str:
    """Build the Markdown/HTML shown in the source passages accordion.

    Entries are numbered to match the `[n]` inline citations the model is
    instructed to produce in the generated answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f'<a id="source-{i}"></a>**[{i}]** ' + _source_links(chunk["source"])

        page_no = chunk.get("page_no", 0)
        if page_no:
            header += f" &nbsp; p.{page_no}"

        heading = chunk.get("heading", "")
        if heading:
            header += f" &nbsp; · &nbsp; _{_collapse_whitespace(heading)}_"

        caption = chunk.get("caption", "")
        if caption:
            header += f" &nbsp; · &nbsp; _{_collapse_whitespace(caption)}_"

        header += f" &nbsp; relevance: `{chunk['score']:.0%}`"

        text = _collapse_whitespace(chunk["text"])
        excerpt = text[:400].strip() + ("…" if len(text) > 400 else "")
        parts.append(f"{header}\n\n> {excerpt}")

    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


_CITATION_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _linkify_citations(answer: str, chunks: list[dict]) -> tuple[str, set[int]]:
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


def _build_uncited_note(chunks: list[dict], cited: set[int]) -> str:
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


def _toc_entry_html(entry: dict) -> str:
    """Render one Table of Contents entry as a new-tab link plus author list."""
    link = _gradio_file_url(entry["path"].resolve())
    title_html = escape(entry["title"])
    authors_html = escape(", ".join(entry.get("authors", [])))
    return (
        f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
        f'class="font-medium hover:underline">{title_html}</a>'
        f'<br><span class="text-sm opacity-75">{authors_html}</span>'
    )


def _slugify(text: str) -> str:
    """Turn a section title into a stable HTML id fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _render_toc(search: str = "") -> str:
    """Render the searchable Table of Contents: a section side menu plus
    collapsible sections for the entries themselves. Styling is done with
    Tailwind utility classes (loaded via the CDN script injected into
    <head>) rather than a hand-rolled <style> block."""
    grouped = list_toc_entries()
    query = (search or "").strip().lower()

    nav_links = []
    sections_html = []
    for section, entries in grouped.items():
        if query:
            entries = [
                e for e in entries
                if query in e["title"].lower()
                or query in section.lower()
                or any(query in author.lower() for author in e.get("authors", []))
            ]
            if not entries:
                continue

        anchor = f"toc-{_slugify(section)}"
        section_html = escape(section)
        nav_links.append(
            f'<a href="#{anchor}" class="block rounded-md px-2 py-1 text-sm '
            f'leading-tight no-underline hover:bg-gray-500/15">{section_html}</a>'
        )

        items_html = "".join(f'<li class="mb-3">{_toc_entry_html(e)}</li>' for e in entries)
        sections_html.append(
            f'<details id="{anchor}" class="mb-3 rounded-lg border border-gray-500/25 '
            f'px-4 py-2" open>'
            f'<summary class="cursor-pointer py-1 text-lg font-semibold">'
            f'{section_html} <span class="text-sm font-normal opacity-65">({len(entries)})</span>'
            "</summary>"
            f'<ul class="mt-2 mb-1 list-none pl-0">{items_html}</ul>'
            "</details>"
        )

    if not sections_html:
        return "<p><em>No matching papers.</em></p>"

    nav_html = (
        '<nav class="sticky top-2 flex w-56 flex-none flex-col gap-1 pr-4 '
        'border-r border-gray-500/30">' + "".join(nav_links) + "</nav>"
    )
    content_html = '<div class="min-w-0 flex-1">' + "".join(sections_html) + "</div>"
    return f'<div class="flex items-start gap-7">{nav_html}{content_html}</div>'


def _load_theme() -> gr.themes.ThemeClass:
    """Load the requested Hugging Face theme with a local fallback."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This theme was created for Gradio .*",
                category=UserWarning,
            )
            return gr.themes.ThemeClass.from_hub("hmb/midnight")
    except Exception:
        return gr.themes.Soft()


def _allowed_source_paths() -> list[str]:
    """Return the source directories Gradio is allowed to serve."""
    paths = [str(path.resolve()) for path in PDF_DIRS if path.exists()]
    # Also allow the app directory so local assets like logo.png are served
    app_dir = str(Path(__file__).parent.resolve())
    if app_dir not in paths:
        paths.append(app_dir)
    return paths


# Maps the UI radio label to the internal audience key used by
# rag/generation.py ("phd" is the shared default in both places).
_AUDIENCE_MAP = {"PhD astronomer": "phd", "Non-expert": "general"}


def answer_question(query: str, audience: str, progress: gr.Progress = gr.Progress()):
    """Retrieve relevant chunks then stream a grounded answer, yielding status
    updates. *progress* drives Gradio's native progress bar: the retrieval
    half (0-0.5) is driven by real stage callbacks from `retrieve()`, and the
    generation half (0.5-1.0) by an asymptotic curve over the live token
    count (final answer length isn't known ahead of time)."""
    query = query.strip()
    if not query:
        yield "", "", ""
        return

    audience_key = _AUDIENCE_MAP.get(audience, "phd")
    progress(0, desc="Searching knowledge base…")

    def _on_retrieve_progress(fraction: float, desc: str) -> None:
        progress(fraction * 0.5, desc=desc)

    try:
        chunks = retrieve(query, top_k=TOP_K, on_progress=_on_retrieve_progress)
    except RuntimeError as exc:
        yield "", f"**Error:** {exc}", ""
        return

    display_chunks = [
        {**chunk, "source": _display_source_name(chunk["source"])}
        for chunk in chunks
    ]
    sources_md = _build_sources_markdown(chunks)

    best_score = max((c["score"] for c in chunks), default=0.0)
    warning = ""
    if best_score < CONFIDENCE_THRESHOLD:
        warning = (
            "⚠️ _The best-matching passages have low relevance — the "
            "knowledge base may not have a good answer to this question._"
        )

    yield warning, "", sources_md

    try:
        answer = ""
        token_count = 0
        for token_count, token in enumerate(
            stream_answer(query, display_chunks, audience=audience_key), start=1
        ):
            answer += token
            fraction = 0.5 + 0.47 * (1 - math.exp(-token_count / 60))
            progress(fraction, desc=f"Generating answer… ({token_count} tokens)")
            yield warning, answer, sources_md
    except ollama.ResponseError as exc:
        yield "", (
            f"**Ollama error:** {exc}\n\n"
            f"Make sure Ollama is running (`ollama serve`) and that "
            f"`{OLLAMA_MODEL}` has been pulled (`ollama pull {OLLAMA_MODEL}`)."
        ), sources_md
        return

    progress(1.0, desc="Done")

    # Final pass: linkify/validate citations only once the answer has fully
    # settled (mid-stream text can contain partial markers like "[1" before
    # the closing bracket arrives, which the regex must not misinterpret).
    linked_answer, cited = _linkify_citations(answer, display_chunks)
    final_sources_md = sources_md + _build_uncited_note(display_chunks, cited)
    yield warning, linked_answer, final_sources_md


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

_LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": r"\(", "right": r"\)", "display": False},
    {"left": r"\[", "right": r"\]", "display": True},
]

# Tailwind is loaded via CDN and injected into <head> at launch() time (Gradio
# 6 moved `css`/`head`/`theme` from the Blocks constructor to launch()). This
# gives us modern utility classes for the custom HTML/Markdown blocks below
# without hand-rolling one-off <style> tags or inline `style="..."` attributes.
_TAILWIND_HEAD = '<script src="https://cdn.tailwindcss.com"></script>'

# Global overrides: bump the base font size (the Gradio default reads small on
# most displays) and give the source-passages accordion label more presence.
_CUSTOM_CSS = """
html {
    scroll-behavior: smooth;
}
.gradio-container {
    font-size: 17px !important;
}
.gradio-container .prose :where(p, li, blockquote) {
    font-size: 1rem;
    line-height: 1.65;
}
.gradio-container h1 { font-size: 2.1rem !important; }
.gradio-container h2 { font-size: 1.6rem !important; }
.gradio-container h3 { font-size: 1.3rem !important; }
.gradio-container textarea,
.gradio-container input[type='text'],
.gradio-container button {
    font-size: 1rem !important;
}
.source-passages-accordion > .label-wrap span {
    font-size: 1.3rem;
    font-weight: 600;
}
.citation-link {
    font-weight: 600;
    text-decoration: none;
    scroll-margin-top: 1rem;
}
.citation-link:hover {
    text-decoration: underline;
}
.citation-invalid {
    color: #b45309;
    border-bottom: 1px dotted currentColor;
    cursor: help;
}
"""

with gr.Blocks(title="ASKArry: SKA RAG documentation search") as demo:
    gr.Markdown(
                f"""
                <div class="mb-3 flex items-center gap-4">
                    <img src="{_gradio_file_url(Path('logo.png').resolve())}" alt="SKARRY logo" class="h-28 w-28 rounded-2xl object-contain shadow-sm" />
                    <div>
                        <h1 class="m-0 text-3xl font-bold tracking-tight">ASKARRY - Advancing Astrophysics with the SKA II RAG search</h1>
                        <p class="mb-0 mt-1 text-base opacity-80">Ask questions about SKA science. Run <code>python ingest.py</code> first to index your PDFs.</p>
                    </div>
                </div>
                """
    )

    with gr.Tabs():
        with gr.Tab("Ask a Question"):
            with gr.Row(), gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Your question",
                    placeholder="What can SKA tell us about magnetic fields?",
                    lines=2,
                )
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Accordion("Settings", open=False):
                audience_radio = gr.Radio(
                    choices=["PhD astronomer", "Non-expert"],
                    value="PhD astronomer",
                    label="Answer level",
                )

            status_box = gr.Markdown(value="", latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False)
            answer_box = gr.Markdown(label="Answer", latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False)

            with gr.Accordion("Source Passages", open=True, elem_classes=["source-passages-accordion"]):
                sources_box = gr.Markdown(latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False)

            # Wire up events
            submit_btn.click(
                answer_question,
                inputs=[query_box, audience_radio],
                outputs=[status_box, answer_box, sources_box],
            )
            query_box.submit(
                answer_question,
                inputs=[query_box, audience_radio],
                outputs=[status_box, answer_box, sources_box],
            )

        with gr.Tab("Table of Contents"):
            toc_search = gr.Textbox(
                label="Search",
                placeholder="Filter by title, author, or section…",
            )
            toc_display = gr.Markdown(
                value=_render_toc(),
                latex_delimiters=_LATEX_DELIMITERS,
                sanitize_html=False,
            )
            toc_search.change(_render_toc, inputs=[toc_search], outputs=[toc_display])



if __name__ == "__main__":
    demo.launch(
        theme=_load_theme(),
        allowed_paths=_allowed_source_paths(),
        css=_CUSTOM_CSS,
        head=_TAILWIND_HEAD,
    )
