"""Direct single-shot answer — the deterministic anchor candidate.

Mirrors ``ChatService._answer_chain``: one call straight through the
citation-rules prompt with no intermediate reasoning, at temperature 0. It
exists in the ensemble as a stable reference point — if the higher-temperature,
more elaborate strategies all drift away from it in the same direction, that's
a meaningful signal; if they agree with it, that's reassuring too.
"""

from .base import ReasoningStrategy


class StandardAgent(ReasoningStrategy):
    name = "standard"

    async def arun(self, question: str, context: str) -> str:
        return await self._finalize(question, context)
