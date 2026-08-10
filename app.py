"""
Gradio web UI for the SKA RAG system.

Usage:
    python app.py
"""

from __future__ import annotations

import math
import queue
import re
import threading
import warnings
from html import escape
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import ollama
from loguru import logger

from rag import store
from rag.bibliography import format_citation, list_toc_entries, lookup_citation
from rag.citations import build_uncited_note, linkify_citations
from rag.config import (
    AUDIENCE_PHD,
    AUDIENCE_UI_LABELS,
    CONFIDENCE_THRESHOLD,
    OLLAMA_MODEL,
    PDF_DIRS,
    TOP_K,
)
from rag.generation import stream_answer
from rag.ingestion import find_acronym_candidates
from rag.retrieval import retrieve, warmup

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
            header += f"<br /> &nbsp; p.{page_no}"

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


# ---------------------------------------------------------------------------
# Acronyms tab
# ---------------------------------------------------------------------------
# Candidate acronyms are discovered by scanning every indexed chunk for
# inline definitions (e.g. "Central Signal Processor (CSP)") — see
# rag.ingestion.find_acronym_candidates. Rebuilding that scan is a bit slow
# (regex over the full corpus text) so it's cached and only rebuilt when the
# ChromaDB collection's chunk count changes, same pattern as the BM25/
# retrieval acronym caches in rag/retrieval.py.
_acronym_cache = store.CountCache()

_ACRONYM_SORT_CHOICES = ["Most frequent", "Most papers", "Alphabetical"]


def _get_acronym_candidates() -> dict[str, dict]:
    collection = store.get_chroma_collection()
    return _acronym_cache.get(collection, lambda: find_acronym_candidates(collection))


def _build_acronym_rows(search: str, sort_by: str) -> list[list]:
    """Build the row data for the Acronyms tab Dataframe: [Acronym,
    Expansion, Occurrences, Papers], filtered by *search* and ordered by
    *sort_by*. The acronym is kept in column 0 so a row-select event
    (`evt.row_value[0]`) can identify which entry was clicked without extra
    bookkeeping state."""
    candidates = _get_acronym_candidates()
    query = (search or "").strip().lower()

    rows = []
    for acronym, entry in candidates.items():
        if query and query not in acronym.lower() and not any(
            query in expansion.lower() for expansion in entry["expansions"]
        ):
            continue
        best_expansion = max(entry["expansions"], key=entry["expansions"].get)
        count = sum(entry["expansions"].values())
        rows.append([acronym, best_expansion, count, len(entry["sources"])])

    if sort_by == "Alphabetical":
        rows.sort(key=lambda row: row[0])
    elif sort_by == "Most papers":
        rows.sort(key=lambda row: (-row[3], -row[2]))
    else:  # "Most frequent"
        rows.sort(key=lambda row: (-row[2], row[0]))

    return rows


def _build_acronym_modal_html(acronym: str) -> str:
    """Render the modal shown when an acronym row is clicked: every distinct
    expansion seen (in case of ambiguity, e.g. DM = dark matter vs.
    dispersion measure) plus a linked list of the papers that use it."""
    entry = _get_acronym_candidates().get(acronym)
    if entry is None:
        return f"### {escape(acronym)}\n\n_No details available._"

    expansions_by_count = sorted(entry["expansions"].items(), key=lambda kv: -kv[1])
    best_expansion, _ = expansions_by_count[0]
    total = sum(entry["expansions"].values())

    lines = [f"### {escape(acronym)} — {escape(best_expansion)}"]
    if len(expansions_by_count) > 1:
        others = ", ".join(escape(text) for text, _ in expansions_by_count[1:])
        lines.append(f"_Also seen as: {others}_")

    lines.append(f"\n**{total} occurrence(s) across {len(entry['sources'])} paper(s):**\n")
    for source in sorted(entry["sources"], key=_display_source_name):
        lines.append(f"- {_source_links(source)}")

    return "\n".join(lines)


def _refresh_acronym_table(search: str, sort_by: str) -> list[list]:
    return _build_acronym_rows(search, sort_by)


def _on_acronym_row_select(evt: gr.SelectData):
    row = evt.row_value
    if not row:
        return gr.update(visible=False), ""
    return gr.update(visible=True), _build_acronym_modal_html(row[0])


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
# rag/generation.py (config.AUDIENCE_UI_LABELS is the single source of truth
# shared between the two modules).


def _progress_bar_html(fraction: float, desc: str) -> str:
    """Render a modern animated progress bar as raw HTML/CSS.

    This replaces Gradio's built-in `gr.Progress` overlay entirely (see
    `_run_retrieve_with_progress` below for why), styled as a rounded
    gradient pill with a moving shimmer and a live percentage/description
    label.
    """
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        '<div class="askarry-progress-wrap">'
        '<div class="askarry-progress-label">'
        f"<span>{escape(desc)}</span>"
        f'<span class="askarry-progress-pct">{pct:.0f}%</span>'
        "</div>"
        '<div class="askarry-progress-track">'
        f'<div class="askarry-progress-fill" style="width: {pct:.1f}%"></div>'
        "</div>"
        "</div>"
    )


def _run_retrieve_with_progress(query: str, top_k: int):
    """Run `retrieve()` on a background thread and yield live progress.

    `retrieve()`'s `on_progress` callback fires synchronously from *inside*
    a blocking, multi-stage pipeline (HyDE, embedding, BM25, cross-encoder
    rerank, context expansion). To surface those updates without relying on
    Gradio's own `gr.Progress` mechanism, `retrieve()` runs on a daemon
    thread and each callback is relayed to this generator through a
    `queue.Queue`, so the caller can `yield` a custom HTML progress bar for
    every stage while still driving a single blocking call underneath.

    Yields `("progress", fraction, desc)` tuples, then a final
    `("result", chunks)` or `("error", exc)` tuple.
    """
    updates: queue.Queue = queue.Queue()

    def _worker() -> None:
        try:
            chunks = retrieve(
                query,
                top_k=top_k,
                on_progress=lambda fraction, desc: updates.put(("progress", fraction, desc)),
            )
            updates.put(("result", chunks))
        except Exception as exc:  # noqa: BLE001 - relayed to the caller, not swallowed
            updates.put(("error", exc))

    threading.Thread(target=_worker, daemon=True, name="rag-retrieve").start()

    while True:
        message = updates.get()
        yield message
        if message[0] != "progress":
            return


def answer_question(query: str, audience: str):
    """Retrieve relevant chunks then stream a grounded answer, yielding status
    updates. The retrieval half (0-0.5) is driven by real stage callbacks
    from `retrieve()` (relayed via `_run_retrieve_with_progress`), and the
    generation half (0.5-1.0) by an asymptotic curve over the live token
    count (final answer length isn't known ahead of time). Progress is
    rendered as a custom HTML bar (`_progress_bar_html`) rather than
    Gradio's built-in `gr.Progress`."""
    query = query.strip()
    if not query:
        yield "", "", ""
        return

    audience_key = AUDIENCE_UI_LABELS.get(audience, AUDIENCE_PHD)
    yield _progress_bar_html(0.0, "Searching knowledge base…"), "", ""

    chunks: list[dict] | None = None
    for message in _run_retrieve_with_progress(query, TOP_K):
        kind = message[0]
        if kind == "progress":
            _, fraction, desc = message
            yield _progress_bar_html(fraction * 0.5, desc), "", ""
        elif kind == "error":
            yield "", f"**Error:** {message[1]}", ""
            return
        else:  # "result"
            chunks = message[1]

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

    status = warning + ("\n\n" if warning else "") + _progress_bar_html(0.5, "Preparing answer…")
    yield status, "", sources_md

    try:
        answer = ""
        token_count = 0
        for token_count, token in enumerate(
            stream_answer(query, display_chunks, audience=audience_key), start=1
        ):
            answer += token
            fraction = 0.5 + 0.47 * (1 - math.exp(-token_count / 60))
            progress_html = _progress_bar_html(fraction, f"Generating answer… ({token_count} tokens)")
            status = warning + ("\n\n" if warning else "") + progress_html
            yield status, answer, sources_md
    except ollama.ResponseError as exc:
        logger.exception("Ollama chat request failed")
        yield "", (
            f"**Ollama error:** {exc}\n\n"
            f"Make sure Ollama is running (`ollama serve`) and that "
            f"`{OLLAMA_MODEL}` has been pulled (`ollama pull {OLLAMA_MODEL}`)."
        ), sources_md
        return

    # Final pass: linkify/validate citations only once the answer has fully
    # settled (mid-stream text can contain partial markers like "[1" before
    # the closing bracket arrives, which the regex must not misinterpret).
    linked_answer, cited = linkify_citations(answer, display_chunks)
    final_sources_md = sources_md + build_uncited_note(display_chunks, cited)
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

# Custom styling (font sizing, citation links, progress bar, etc.) lives in
# static/style.css and is loaded via `css_paths=` in demo.launch() below.
_CUSTOM_CSS_PATH = Path(__file__).parent / "static" / "style.css"

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

            # Wire up events. `show_progress="hidden"` suppresses Gradio's
            # own pending overlay/progress bar entirely — progress is shown
            # via the custom HTML bar yielded into `status_box` instead (see
            # `_progress_bar_html` / `_run_retrieve_with_progress`).
            submit_btn.click(
                answer_question,
                inputs=[query_box, audience_radio],
                outputs=[status_box, answer_box, sources_box],
                show_progress="hidden",
            )
            query_box.submit(
                answer_question,
                inputs=[query_box, audience_radio],
                outputs=[status_box, answer_box, sources_box],
                show_progress="hidden",
            )

        with gr.Tab("Browse Chapters"):
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

        with gr.Tab("Acronyms"):
            gr.Markdown(
                "Acronyms discovered by scanning inline definitions across the "
                "indexed papers (e.g. _\u201cCentral Signal Processor (CSP)\u201d_). "
                "Click a row to see every paper that uses it."
            )
            with gr.Row():
                acronym_search = gr.Textbox(
                    label="Search",
                    placeholder="Filter by acronym or expansion…",
                    scale=3,
                )
                acronym_sort = gr.Radio(
                    choices=_ACRONYM_SORT_CHOICES,
                    value=_ACRONYM_SORT_CHOICES[0],
                    label="Sort by",
                    scale=2,
                )

            acronym_table = gr.Dataframe(
                headers=["Acronym", "Expansion", "Occurrences", "Papers"],
                datatype=["str", "str", "number", "number"],
                value=_build_acronym_rows("", _ACRONYM_SORT_CHOICES[0]),
                interactive=False,
                wrap=True,
            )

            with gr.Group(visible=False, elem_classes=["askarry-modal-overlay"]) as acronym_modal, \
                 gr.Column(elem_classes=["askarry-modal-box"]):
                acronym_modal_close = gr.Button(
                    "✕ Close", size="sm", elem_classes=["askarry-modal-close"]
                )
                acronym_modal_content = gr.Markdown(
                    latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False
                )

            acronym_search.change(
                _refresh_acronym_table,
                inputs=[acronym_search, acronym_sort],
                outputs=[acronym_table],
            )
            acronym_sort.change(
                _refresh_acronym_table,
                inputs=[acronym_search, acronym_sort],
                outputs=[acronym_table],
            )
            acronym_table.select(
                _on_acronym_row_select,
                outputs=[acronym_modal, acronym_modal_content],
            )
            acronym_modal_close.click(
                lambda: gr.update(visible=False),
                outputs=[acronym_modal],
            )



if __name__ == "__main__":
    # Pre-load models in the background so the first query doesn't block the
    # UI. Guarded here (rather than at module import time) so importing
    # app.py — e.g. from tests — has no side effects.
    threading.Thread(target=warmup, daemon=True, name="rag-warmup").start()

    demo.launch(
        theme=_load_theme(),
        allowed_paths=_allowed_source_paths(),
        css_paths=_CUSTOM_CSS_PATH,
        head=_TAILWIND_HEAD,
    )
