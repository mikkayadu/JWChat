"""Self-consistency answer ensemble for the chat pipeline.

Runs several reasoning strategies over the same retrieved context in parallel
and picks a winner by embedding-based majority vote — see
:class:`orchestrator.RAGEnsemble` for the orchestrator and
:data:`strategies.STRATEGY_MAP` for the registry of available strategies.
"""

from .orchestrator import EnsembleResult, RAGEnsemble, StrategyResult, Winner
from .strategies import STRATEGY_MAP, ReasoningStrategy

__all__ = [
    "RAGEnsemble",
    "EnsembleResult",
    "StrategyResult",
    "Winner",
    "STRATEGY_MAP",
    "ReasoningStrategy",
]
