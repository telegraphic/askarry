"""
Answer generation using a local Ollama model.
"""

from __future__ import annotations

import ollama

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
)

_client = ollama.Client(host=OLLAMA_BASE_URL)

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
"""


def _build_context_block(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] Source: {c['source']} | relevance: {c['score']:.2f}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate an answer to *query* grounded in the retrieved *chunks*.

    Raises `ollama.ResponseError` if the Ollama server is unreachable or the
    model has not been pulled yet.
    """
    if not chunks:
        return "No relevant passages were found in the indexed papers."

    context_block = _build_context_block(chunks)

    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    response = _client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options=_CHAT_OPTIONS,
    )
    return response["message"]["content"]


def stream_answer(query: str, chunks: list[dict]):
    """
    Stream an answer token-by-token.

    Yields successive string tokens. Raises `ollama.ResponseError` if the
    Ollama server is unreachable or the model has not been pulled yet.
    """
    if not chunks:
        yield "No relevant passages were found in the indexed papers."
        return

    context_block = _build_context_block(chunks)

    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    response = _client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options=_CHAT_OPTIONS,
        stream=True,
    )
    for chunk in response:
        token = chunk["message"]["content"]
        if token:
            yield token
