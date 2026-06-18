"""Registry mapping strategy names — and their common aliases — to classes.

Mirrors the reference ``AGENT_MAP`` shape so ``ChatService``/``RAGEnsemble``
can resolve user-configured strategy names (e.g. via
``ENSEMBLE_STRATEGIES=cot,tot,consistency``) against a single source of truth,
without hardcoding the available set anywhere else.
"""

from .base import ReasoningStrategy
from .consistency import ConsistencyAgent
from .cot import CoTAgent
from .decomposed import DecomposedAgent
from .least_to_most import LeastToMostAgent
from .react import ReActAgent
from .recursive import RecursiveAgent
from .self_reflection import SelfReflectionAgent
from .standard import StandardAgent
from .tot import ToTAgent

STRATEGY_MAP: dict[str, type[ReasoningStrategy]] = {
    "standard": StandardAgent,
    "cot": CoTAgent,
    "chain_of_thought": CoTAgent,
    "tot": ToTAgent,
    "tree_of_thoughts": ToTAgent,
    "react": ReActAgent,
    "self_reflection": SelfReflectionAgent,
    "reflection": SelfReflectionAgent,
    "consistency": ConsistencyAgent,
    "self_consistency": ConsistencyAgent,
    "decomposed": DecomposedAgent,
    "least_to_most": LeastToMostAgent,
    "ltm": LeastToMostAgent,
    "recursive": RecursiveAgent,
    "rlm": RecursiveAgent,
}

__all__ = [
    "ReasoningStrategy",
    "STRATEGY_MAP",
    "StandardAgent",
    "CoTAgent",
    "ToTAgent",
    "ReActAgent",
    "SelfReflectionAgent",
    "ConsistencyAgent",
    "DecomposedAgent",
    "LeastToMostAgent",
    "RecursiveAgent",
]
