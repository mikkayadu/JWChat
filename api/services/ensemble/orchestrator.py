"""RAGEnsemble: run several reasoning strategies in parallel over the same
question/context and pick a winner via embedding-based majority vote.

Adapted from the ``agent_reasoning.ensemble.ReasoningEnsemble`` reference
shape for this codebase: strategies are LangChain/Gemini-based
``ReasoningStrategy`` instances rather than Ollama agents, concurrency uses
native ``asyncio.gather`` over ``.ainvoke()`` (no thread pool needed — Gemini's
async client is already non-blocking), and voting reuses the chat service's
already-configured embeddings client instead of adding a new ML dependency
(see module docs in ``strategies/consistency.py`` for why *this* vote needs
semantic rather than lexical comparison).
"""

import asyncio
import math
import time
from typing import TypedDict

from langchain_core.embeddings import Embeddings
from loguru import logger

from .strategies.base import ReasoningStrategy


class StrategyResult(TypedDict):
    strategy: str
    answer: str
    duration_ms: float
    error: bool


class Winner(TypedDict):
    strategy: str | None
    answer: str | None
    vote_count: int


class EnsembleResult(TypedDict):
    winner: Winner
    candidates: list[StrategyResult]
    clusters: list[list[int]]
    vote_count: int
    total_duration_ms: float


class RAGEnsemble:
    """Runs registered strategies concurrently and majority-votes on their
    answers via embedding-based semantic-similarity clustering."""

    def __init__(
        self,
        strategies: dict[str, ReasoningStrategy],
        embeddings: Embeddings,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._strategies = strategies
        self._embeddings = embeddings
        self._similarity_threshold = similarity_threshold

    async def run(self, question: str, context: str) -> EnsembleResult:
        start = time.time()
        names = list(self._strategies.keys())
        logger.info(f"[ENSEMBLE] running {len(names)} strategies: {names}")

        candidates = list(
            await asyncio.gather(
                *(
                    self._run_one(name, strategy, question, context)
                    for name, strategy in self._strategies.items()
                )
            )
        )

        winner, clusters = self._majority_vote(candidates)
        total_duration_ms = (time.time() - start) * 1000
        valid_count = sum(1 for c in candidates if not c["error"])

        logger.info(
            f"[ENSEMBLE] winner='{winner['strategy']}' "
            f"votes={winner['vote_count']}/{valid_count} "
            f"clusters={[len(c) for c in clusters]} "
            f"total={total_duration_ms:.0f}ms"
        )

        return {
            "winner": winner,
            "candidates": candidates,
            "clusters": clusters,
            "vote_count": winner["vote_count"],
            "total_duration_ms": total_duration_ms,
        }

    def run_sync(self, question: str, context: str) -> EnsembleResult:
        """Sync entry point for the LangGraph node (which invokes synchronously).

        Safe to call here: ``ChatService.ask()`` is invoked from a *sync*
        FastAPI route (``def chat(...)`` in api/app.py), which FastAPI runs in
        a worker thread with no running event loop — so ``asyncio.run`` won't
        collide with one.
        """
        return asyncio.run(self.run(question, context))

    async def _run_one(
        self, name: str, strategy: ReasoningStrategy, question: str, context: str
    ) -> StrategyResult:
        start = time.time()
        try:
            answer = await strategy.run_logged(question, context)
            return {
                "strategy": name,
                "answer": answer,
                "duration_ms": (time.time() - start) * 1000,
                "error": False,
            }
        except Exception as e:
            logger.warning(f"[ENSEMBLE] strategy '{name}' raised — excluding from vote: {e}")
            return {
                "strategy": name,
                "answer": f"Error: {e}",
                "duration_ms": (time.time() - start) * 1000,
                "error": True,
            }

    def _majority_vote(self, candidates: list[StrategyResult]) -> tuple[Winner, list[list[int]]]:
        """Cluster the valid candidates' answers by embedding similarity and
        return the winner from the largest cluster, plus the clusters
        (as indices into the *valid* candidate list) for logging/inspection.
        """
        valid = [c for c in candidates if not c["error"]]

        if not valid:
            return {"strategy": None, "answer": None, "vote_count": 0}, []

        if len(valid) == 1:
            only = valid[0]
            return (
                {"strategy": only["strategy"], "answer": only["answer"], "vote_count": 1},
                [[0]],
            )

        embeddings = self._embeddings.embed_documents([c["answer"] for c in valid])
        clusters = self._cluster_by_similarity(embeddings)

        largest_size = max(len(cluster) for cluster in clusters)
        largest_clusters = [cluster for cluster in clusters if len(cluster) == largest_size]

        # Tie-break: prefer the cluster — and within it, the candidate — whose
        # strategy was registered earliest. Predictable, and avoids hardcoding
        # a particular strategy name as a fallback (the strategy set is
        # user-configured via ENSEMBLE_STRATEGIES, so no name is guaranteed
        # to be present).
        registration_order = {name: i for i, name in enumerate(self._strategies)}
        winning_cluster = min(
            largest_clusters,
            key=lambda cluster: min(registration_order[valid[i]["strategy"]] for i in cluster),
        )
        winner_idx = min(winning_cluster, key=lambda i: registration_order[valid[i]["strategy"]])
        winner = valid[winner_idx]

        return (
            {
                "strategy": winner["strategy"],
                "answer": winner["answer"],
                "vote_count": len(winning_cluster),
            },
            clusters,
        )

    def _cluster_by_similarity(self, embeddings: list[list[float]]) -> list[list[int]]:
        """Greedy cosine-similarity clustering, in pure Python — no numpy.

        N here is the number of strategies (small — typically 3-9), so the
        O(N^2) pairwise comparison is negligible. Pulling in numpy for this
        would be overkill in a codebase that doesn't otherwise depend on it.
        """
        n = len(embeddings)
        norms = [math.sqrt(sum(x * x for x in vec)) or 1.0 for vec in embeddings]
        assigned = [False] * n
        clusters: list[list[int]] = []

        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
                similarity = dot / (norms[i] * norms[j])
                if similarity >= self._similarity_threshold:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)

        return clusters
