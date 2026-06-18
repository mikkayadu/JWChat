"""Decomposition: split the question into sub-questions, answer each
independently from the context (in parallel), then synthesize one cited
final answer.

Useful for compound questions where a single pass risks dropping one part —
each sub-question gets focused attention before being woven back together.
Sub-answers are solved independently and concurrently; contrast with
``LeastToMostAgent``, which solves them sequentially because later ones
depend on earlier ones.
"""

import asyncio
import re

from langchain_core.prompts import ChatPromptTemplate

from .base import ReasoningStrategy

_MAX_SUBQUESTIONS = 3
_TEMPERATURE = 0.3

_SPLIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Break the question into up to "
            f"{_MAX_SUBQUESTIONS} smaller sub-questions that, answered together, "
            "would fully answer the original. If the question is already simple "
            "and atomic, return just it as the only sub-question. "
            "List one sub-question per line, numbered '1.', '2.', etc. — nothing else.",
        ),
        ("human", "Question:\n{question}"),
    ]
)

_SUBANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer this sub-question using only the numbered context passages "
            "below. Cite sources as [1], [2], etc. Be concise — this feeds into "
            "a larger answer, it isn't the final response.",
        ),
        ("human", "Sub-question:\n{subquestion}\n\nContext:\n{context}"),
    ]
)

_SUBQ_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)


class DecomposedAgent(ReasoningStrategy):
    name = "decomposed"

    async def arun(self, question: str, context: str) -> str:
        subquestions = await self._split(question)

        sub_answers = await asyncio.gather(
            *(
                self._ainvoke(
                    _SUBANSWER_PROMPT,
                    {"subquestion": sq, "context": context},
                    temperature=_TEMPERATURE,
                )
                for sq in subquestions
            )
        )

        notes = "\n\n".join(
            f"Sub-question: {sq}\nSub-answer: {sa}"
            for sq, sa in zip(subquestions, sub_answers)
        )
        return await self._finalize(question, context, notes=notes)

    async def _split(self, question: str) -> list[str]:
        raw = await self._ainvoke(_SPLIT_PROMPT, {"question": question}, temperature=_TEMPERATURE)
        subquestions = [m.group(1).strip() for m in _SUBQ_RE.finditer(raw)]
        return subquestions[:_MAX_SUBQUESTIONS] or [question]
