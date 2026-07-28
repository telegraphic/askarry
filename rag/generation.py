"""
Answer generation using a local Ollama model.
"""

from __future__ import annotations

import ollama
from loguru import logger

from . import store
from .config import (
    AUDIENCE_GENERAL,
    AUDIENCE_PHD,
    CONFIDENCE_THRESHOLD,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
)

_CHAT_OPTIONS = {
    "num_ctx": OLLAMA_NUM_CTX,
    "temperature": OLLAMA_TEMPERATURE,
    "top_p": OLLAMA_TOP_P,
    "top_k": OLLAMA_TOP_K,
}

_SYSTEM_PROMPT = """\
You are an expert technical assistant for the Square Kilometre Array (SKA) project. \
Your job is to answer questions using only the context passages provided, which come \
from SKA documentation, technical specifications, and related astronomy papers. \

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain sufficient information, say so clearly.
- Cite sources inline using bracketed numbers, e.g. [1], [2], matching the
  passage numbers in the context block. Cite every factual claim.
- Keep your answer concise and technically precise.
- Use correct SKA terminology and acronyms where appropriate.
- Avoid superlatives and hyped or promotional language (e.g. "groundbreaking",
  "cutting-edge", "revolutionary", "unprecedented", "vital"); state facts
  plainly and let the evidence speak for itself.
- Write in a classic, restrained scientific prose style: plain declarative
  sentences, no rhetorical flourishes, and no exclamation points.
"""

# Audience-specific instructions appended to `_SYSTEM_PROMPT`, selected via the
# `audience` argument on `generate_answer`/`stream_answer`. AUDIENCE_PHD is the
# default for both the UI (app.py) and these functions, so any caller that
# omits the argument still gets PhD-level output.
_AUDIENCE_INSTRUCTIONS: dict[str, str] = {
    AUDIENCE_PHD: (
        "\nAudience: write for a reader with a PhD in astronomy. Use precise "
        "technical terminology, standard notation and units, and equations "
        "where they aid precision. Do not define standard concepts, "
        "acronyms, or units that a professional astronomer would already "
        "know. When the context passages contain relevant formulas, "
        "parameter values, or other numerical results, include them "
        "explicitly in the answer rather than describing them only "
        "qualitatively."
    ),
    AUDIENCE_GENERAL: (
        "\nAudience: write for a non-expert reader. Use plain language, "
        "avoid unexplained jargon, define any acronyms or technical terms "
        "on first use, and avoid presenting equations without explaining "
        "what they mean in words."
    ),
}


def _build_context_block(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] Source: {c['source']} | relevance: {c['score']:.2f}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def _build_user_message(query: str, chunks: list[dict]) -> str:
    """Assemble the user-turn message: context block, a low-confidence hedge
    instruction when retrieval didn't find a strong match, then the question."""
    context_block = _build_context_block(chunks)

    parts = [f"Context passages:\n\n{context_block}"]

    best_score = max((c["score"] for c in chunks), default=0.0)
    if best_score < CONFIDENCE_THRESHOLD:
        parts.append(
            "Note: none of the retrieved passages score highly for relevance "
            "to this question. Treat them with skepticism — do not force an "
            "answer out of weakly related material. If they don't genuinely "
            "answer the question, say clearly that the knowledge base doesn't "
            "appear to contain a good answer."
        )

    parts.append(f"Question: {query}")
    return "\n\n".join(parts)


def _build_system_prompt(audience: str) -> str:
    """Combine the base system prompt with audience-specific instructions.
    Falls back to the AUDIENCE_PHD (default) audience for unknown keys."""
    return _SYSTEM_PROMPT + _AUDIENCE_INSTRUCTIONS.get(
        audience, _AUDIENCE_INSTRUCTIONS[AUDIENCE_PHD]
    )


def generate_answer(query: str, chunks: list[dict], audience: str = AUDIENCE_PHD) -> str:
    """
    Generate an answer to *query* grounded in the retrieved *chunks*, written
    for the given *audience* (AUDIENCE_PHD or AUDIENCE_GENERAL).

    Raises `ollama.ResponseError` if the Ollama server is unreachable or the
    model has not been pulled yet.
    """
    if not chunks:
        return "No relevant passages were found in the indexed papers."

    user_message = _build_user_message(query, chunks)

    try:
        response = store.get_ollama_client().chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _build_system_prompt(audience)},
                {"role": "user", "content": user_message},
            ],
            options=_CHAT_OPTIONS,
        )
    except ollama.ResponseError as exc:
        logger.error(f"Ollama chat request failed: {exc}")
        raise
    return response["message"]["content"]


def stream_answer(query: str, chunks: list[dict], audience: str = AUDIENCE_PHD):
    """
    Stream an answer token-by-token, written for the given *audience*
    (AUDIENCE_PHD or AUDIENCE_GENERAL).

    Yields successive string tokens. Raises `ollama.ResponseError` if the
    Ollama server is unreachable or the model has not been pulled yet.
    """
    if not chunks:
        yield "No relevant passages were found in the indexed papers."
        return

    user_message = _build_user_message(query, chunks)

    try:
        response = store.get_ollama_client().chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _build_system_prompt(audience)},
                {"role": "user", "content": user_message},
            ],
            options=_CHAT_OPTIONS,
            stream=True,
        )
    except ollama.ResponseError as exc:
        logger.error(f"Ollama streaming chat request failed: {exc}")
        raise
    for chunk in response:
        token = chunk["message"]["content"]
        if token:
            yield token
