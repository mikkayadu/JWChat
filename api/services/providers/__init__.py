"""Swappable model providers.

Public entry point is :class:`ModelFactory`. Concrete provider modules are
imported here for their registration side effects — add a new provider module to
this import list and it becomes selectable via ``LLM_PROVIDER`` /
``EMBEDDINGS_PROVIDER`` with no other code changes.
"""

from .decorators import DimensionAwareEmbeddings
from .registry import (
    ModelFactory,
    register_embeddings,
    register_llm,
    registered_embeddings_providers,
    registered_llm_providers,
)

# Side-effect imports: each module self-registers its builders on import.
from . import google  # noqa: F401,E402
from . import ollama  # noqa: F401,E402
from . import groq  # noqa: F401,E402

__all__ = [
    "ModelFactory",
    "DimensionAwareEmbeddings",
    "register_llm",
    "register_embeddings",
    "registered_llm_providers",
    "registered_embeddings_providers",
]
