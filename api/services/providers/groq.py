"""Groq provider registration (fast, free-tier hosted open models).

Importing this module registers the ``groq`` LLM builder with the factory, so
``groq`` can be used as a primary or fallback backend (e.g.
``FALLBACKS=groq:openai/gpt-oss-20b,groq:qwen/qwen3.8-27b``).

Groq runs open models (Llama, Qwen, Gemma, …) on its own accelerators and exposes
an OpenAI-compatible API, so responses are fast even on the free tier. Requires a
free ``GROQ_API_KEY`` from https://console.groq.com.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

from .registry import register_llm

# Groq retires hosted models without notice (the Llama 3.x line it launched with
# is already gone, 404ing as "model does not exist"). Keep this pointing at a
# model the catalogue actually serves - check https://api.groq.com/openai/v1/models
# before changing it.
DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"


def _resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY for Groq models")
    return key


@register_llm("groq")
def build_groq_llm(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
    **kwargs,
) -> ChatGroq:
    return ChatGroq(
        model=model or os.getenv("GROQ_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        api_key=_resolve_api_key(api_key),
        temperature=temperature,
        **kwargs,
    )
