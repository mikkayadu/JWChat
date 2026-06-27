"""Provider-agnostic embeddings decorators."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from ingestion.utils.embedder import normalize_vector_dimensions


class DimensionAwareEmbeddings(Embeddings):
    """Decorator that resizes any embedder's vectors to a fixed dimension.

    Wraps any ``Embeddings`` implementation (Decorator pattern) so the returned
    vectors always match the Pinecone index dimension, no matter which provider
    generated them. ``target_dim <= 0`` means "leave the model-native size
    alone".
    """

    def __init__(self, base: Embeddings, target_dim: int) -> None:
        self._base = base
        self._target_dim = target_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._base.embed_documents(texts)
        if self._target_dim <= 0:
            return vectors
        return [normalize_vector_dimensions(v, self._target_dim) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._base.embed_query(text)
        if self._target_dim <= 0:
            return vector
        return normalize_vector_dimensions(vector, self._target_dim)
