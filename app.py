"""
Gradio web UI for the SKA RAG system.

Usage:
    python app.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote
import threading
import warnings

import gradio as gr
import ollama

from rag.bibliography import format_citation, list_toc_entries, lookup_citation
from rag.config import OLLAMA_MODEL, PDF_DIRS, TOP_K
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


def _build_sources_markdown(chunks: list[dict]) -> str:
    """Build the Markdown/HTML shown in the source passages accordion.

    Entries are numbered to match the `[n]` inline citations the model is
    instructed to produce in the generated answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"**[{i}]** " + _source_links(chunk["source"])

        page_no = chunk.get("page_no", 0)
        if page_no:
            header += f" &nbsp; p.{page_no}"

        heading = chunk.get("heading", "")
        if heading:
            header += f" &nbsp; · &nbsp; _{heading}_"

        caption = chunk.get("caption", "")
        if caption:
            header += f" &nbsp; · &nbsp; _{caption}_"

        header += f" &nbsp; relevance: `{chunk['score']:.2f}`"

        text = chunk["text"]
        excerpt = text[:400].strip() + ("…" if len(text) > 400 else "")
        parts.append(f"{header}\n\n> {excerpt}")

    if not parts:
        return ""
    return "## Source Passages\n\n" + "\n\n---\n\n".join(parts)


def _toc_entry_html(entry: dict) -> str:
    """Render one Table of Contents entry as a new-tab link plus author list."""
    link = _gradio_file_url(entry["path"].resolve())
    title_html = escape(entry["title"])
    authors_html = escape(", ".join(entry.get("authors", [])))
    return (
        f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title_html}</a>'
        f'<br><span style="opacity:0.75;">{authors_html}</span>'
    )


def _render_toc(search: str = "") -> str:
    """Render the searchable Table of Contents, grouped by section."""
    grouped = list_toc_entries()
    query = (search or "").strip().lower()

    sections_md = []
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
        items_md = "\n\n".join(f"- {_toc_entry_html(e)}" for e in entries)
        sections_md.append(f"### {section}\n\n{items_md}")

    if not sections_md:
        return "_No matching papers._"
    return "\n\n".join(sections_md)


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


def answer_question(query: str):
    """Retrieve relevant chunks then stream a grounded answer, yielding status updates."""
    query = query.strip()
    if not query:
        yield "", "", ""
        return

    yield "_🔍 Searching knowledge base…_", "", ""

    try:
        chunks = retrieve(query, top_k=TOP_K)
    except RuntimeError as exc:
        yield "", f"**Error:** {exc}", ""
        return

    display_chunks = [
        {**chunk, "source": _display_source_name(chunk["source"])}
        for chunk in chunks
    ]
    sources_md = _build_sources_markdown(chunks)

    yield "_💬 Generating answer…_", "", sources_md

    try:
        answer = ""
        for token in stream_answer(query, display_chunks):
            answer += token
            yield "", answer, sources_md
    except ollama.ResponseError as exc:
        yield "", (
            f"**Ollama error:** {exc}\n\n"
            f"Make sure Ollama is running (`ollama serve`) and that "
            f"`{OLLAMA_MODEL}` has been pulled (`ollama pull {OLLAMA_MODEL}`)."
        ), sources_md
        return

    yield "", answer, sources_md


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

_LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": r"\(", "right": r"\)", "display": False},
    {"left": r"\[", "right": r"\]", "display": True},
]

with gr.Blocks(title="SKArry: SKA RAG documentation search") as demo:
    gr.Markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.9rem; margin-bottom:0.75rem;">
                    <img src="{_gradio_file_url(Path('logo.png').resolve())}" alt="SKARRY logo" style="width:112px; height:112px; object-fit:contain; border-radius:12px;" />
                    <div>
                        <h1 style="margin:0;">ASKARRY - Advancing Astrophysics with the SKA II RAG search</h1>
                        <p style="margin:0.35rem 0 0;">Ask questions about SKA science. Run <code>python ingest.py</code> first to index your PDFs.</p>
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

            status_box = gr.Markdown(value="", latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False)
            answer_box = gr.Markdown(label="Answer", latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False)

            with gr.Accordion("Source passages", open=False):
                sources_box = gr.Markdown(latex_delimiters=_LATEX_DELIMITERS, sanitize_html=False)

            # Wire up events
            submit_btn.click(
                answer_question,
                inputs=[query_box],
                outputs=[status_box, answer_box, sources_box],
            )
            query_box.submit(
                answer_question,
                inputs=[query_box],
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
    demo.launch(theme=_load_theme(), allowed_paths=_allowed_source_paths())
