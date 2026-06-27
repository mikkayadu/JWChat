"""Ollama (local models) provider registration.

Importing this module registers the ``ollama`` LLM builder with the factory, so
``LLM_PROVIDER=ollama`` selects a locally-served model (Llama, Mistral, Qwen, …)
with no other code changes.

Ollama needs no API key — just a running daemon. The endpoint defaults to
``http://localhost:11434`` and is overridable via ``OLLAMA_BASE_URL``.

Only the chat LLM is registered here. To make Ollama embeddings selectable too,
add a sibling ``@register_embeddings("ollama")`` builder that returns
``OllamaEmbeddings(model=..., base_url=...)`` — but note that switching the
embeddings backend means re-ingesting into a fresh Pinecone index, since vectors
from different models are not comparable.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

from .registry import register_llm

DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_BASE_URL = "http://localhost:11434"


def _resolve_base_url(explicit: str | None) -> str:
    return explicit or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)


@register_llm("ollama")
def build_ollama_llm(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,  # accepted and ignored — Ollama is keyless
    **kwargs,
) -> ChatOllama:
    return ChatOllama(
        model=model or os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL),
        base_url=_resolve_base_url(base_url),
        temperature=temperature,
        **kwargs,
    )
