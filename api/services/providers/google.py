"""Google (Gemini) provider registrations.

Importing this module registers the ``google`` LLM and embeddings builders with
the factory. To add another backend (e.g. OpenAI), create a sibling module that
calls ``@register_llm("openai")`` / ``@register_embeddings("openai")`` and import
it in ``providers/__init__.py`` — nothing else changes.
"""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .registry import register_llm, register_embeddings


def _resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for Google models")
    return key


def _resolve_model(explicit: str | None, env_var: str, hint: str = "") -> str:
    # No model names in code: Google closes older models to new API keys
    # without notice, so the choice lives in the environment where it can be
    # changed without a deploy.
    model = explicit or os.getenv(env_var)
    if not model:
        raise RuntimeError(f"Missing {env_var} for Google models{hint}")
    return model


@register_llm("google")
def build_google_llm(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=_resolve_model(
            model,
            "CHAT_MODEL",
            " (or INTENT_CHAT_MODEL when Google only classifies intent)",
        ),
        google_api_key=_resolve_api_key(api_key),
        temperature=temperature,
        **kwargs,
    )


@register_embeddings("google")
def build_google_embeddings(
    *,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=_resolve_model(model, "EMBEDDING_MODEL"),
        google_api_key=_resolve_api_key(api_key),
        **kwargs,
    )
