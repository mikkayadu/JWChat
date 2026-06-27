"""Provider registry + factory for swappable model backends.

The *Strategy* interface is supplied by LangChain itself — every chat model is a
``BaseChatModel`` and every embedder is an ``Embeddings``. This module adds the
two missing pieces:

* a **registry** so each provider (Google, OpenAI, …) registers a builder
  function for its LLM and/or embeddings, and
* a **factory** (:class:`ModelFactory`) that resolves a provider name from
  config and hands back the right concrete instance.

Because providers self-register via decorators, adding a new backend means
writing one module and importing it — no edits here or in ``ChatService``
(open/closed principle).
"""

from __future__ import annotations

import os
from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .decorators import DimensionAwareEmbeddings

LLMBuilder = Callable[..., BaseChatModel]
EmbeddingsBuilder = Callable[..., Embeddings]

_LLM_BUILDERS: dict[str, LLMBuilder] = {}
_EMBEDDINGS_BUILDERS: dict[str, EmbeddingsBuilder] = {}

DEFAULT_LLM_PROVIDER = "google"
DEFAULT_EMBEDDINGS_PROVIDER = "google"


def register_llm(provider: str) -> Callable[[LLMBuilder], LLMBuilder]:
    """Decorator: register an LLM builder under ``provider`` (case-insensitive)."""

    def decorator(builder: LLMBuilder) -> LLMBuilder:
        _LLM_BUILDERS[provider.lower()] = builder
        return builder

    return decorator


def register_embeddings(provider: str) -> Callable[[EmbeddingsBuilder], EmbeddingsBuilder]:
    """Decorator: register an embeddings builder under ``provider`` (case-insensitive)."""

    def decorator(builder: EmbeddingsBuilder) -> EmbeddingsBuilder:
        _EMBEDDINGS_BUILDERS[provider.lower()] = builder
        return builder

    return decorator


class ModelFactory:
    """Resolves a provider name to a concrete model instance.

    ``provider=None`` falls back to the ``LLM_PROVIDER`` / ``EMBEDDINGS_PROVIDER``
    env vars (then the package defaults), so callers that don't care about the
    backend can stay provider-agnostic.
    """

    @staticmethod
    def create_llm(
        provider: str | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseChatModel:
        name = (provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)).lower()
        builder = _LLM_BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown LLM provider '{name}'. "
                f"Registered: {sorted(_LLM_BUILDERS)}"
            )
        return builder(model=model, temperature=temperature, **kwargs)

    @staticmethod
    def create_embeddings(
        provider: str | None = None,
        *,
        model: str | None = None,
        target_dim: int = 0,
        **kwargs,
    ) -> Embeddings:
        name = (provider or os.getenv("EMBEDDINGS_PROVIDER", DEFAULT_EMBEDDINGS_PROVIDER)).lower()
        builder = _EMBEDDINGS_BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown embeddings provider '{name}'. "
                f"Registered: {sorted(_EMBEDDINGS_BUILDERS)}"
            )
        base = builder(model=model, **kwargs)
        # Pinecone needs every vector at the index dimension regardless of which
        # provider produced it — so the resize wrapper lives here, once, instead
        # of in each provider builder.
        if target_dim and target_dim > 0:
            return DimensionAwareEmbeddings(base, target_dim)
        return base


def registered_llm_providers() -> list[str]:
    return sorted(_LLM_BUILDERS)


def registered_embeddings_providers() -> list[str]:
    return sorted(_EMBEDDINGS_BUILDERS)
