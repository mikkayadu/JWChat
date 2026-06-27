"""Groq provider registration (fast, free-tier hosted open models).

Importing this module registers the ``groq`` LLM builder with the factory, so
``groq`` can be used as a primary or fallback backend (e.g.
``FALLBACKS=groq:llama-3.1-8b-instant,groq:llama-3.3-70b-versatile``).

Groq runs open models (Llama, Qwen, Gemma, …) on its own accelerators and exposes
an OpenAI-compatible API, so responses are fast even on the free tier. Requires a
free ``GROQ_API_KEY`` from https://console.groq.com.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

from .registry import register_llm

DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"


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
