"""Self-Consistency: sample the baseline answer prompt multiple times at
higher temperature and return the most-repeated response.

Internal voting here uses lexical similarity (``difflib``) rather than
embeddings. That's deliberate, not a shortcut: these samples all come from the
*same* prompt, so genuine agreement between them tends to produce near-verbatim
phrasing — unlike the orchestrator's cross-strategy vote, which must tolerate
substantively-equivalent answers phrased very differently and therefore needs
semantic (embedding) comparison. Keeping this strategy's internal check lexical
also keeps it self-contained — no embeddings client needs to be threaded in
just for one strategy among several.
"""

import asyncio
from difflib import SequenceMatcher

from .base import ReasoningStrategy

_SAMPLES = 3
_SAMPLE_TEMPERATURE = 0.7
_SIMILARITY_THRESHOLD = 0.8


class ConsistencyAgent(ReasoningStrategy):
    name = "consistency"

    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        samples = await asyncio.gather(
            *(
                self._finalize(
                    question,
                    context,
                    answer_language=answer_language,
                    temperature=_SAMPLE_TEMPERATURE,
                )
                for _ in range(_SAMPLES)
            )
        )
        return self._most_repeated(list(samples))

    @staticmethod
    def _most_repeated(samples: list[str]) -> str:
        """Return the sample with the most lexical "agreers" among the set."""
        if len(samples) == 1:
            return samples[0]

        best_idx, best_count = 0, -1
        for i, candidate in enumerate(samples):
            count = sum(
                1
                for other in samples
                if SequenceMatcher(None, candidate, other).ratio() >= _SIMILARITY_THRESHOLD
            )
            if count > best_count:
                best_idx, best_count = i, count

        return samples[best_idx]
