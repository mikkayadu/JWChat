"""Least-to-Most: decompose into sub-questions ordered from simplest/most
foundational to most complex, solve them in sequence — each building on the
answers before it — then synthesize the final cited answer.

Unlike ``DecomposedAgent`` (independent sub-questions answered in parallel),
this is deliberately sequential: it suits questions that build up an idea
step by step, where a later sub-answer needs an earlier one as a foundation,
rather than questions that simply have several unrelated parts.
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from .base import ReasoningStrategy

_MAX_SUBQUESTIONS = 3
_TEMPERATURE = 0.3

_ORDER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Break the question into up to "
            f"{_MAX_SUBQUESTIONS} sub-questions ordered from simplest/most "
            "foundational to most complex, such that each one builds on the "
            "answers to those before it. If the question is already simple and "
            "atomic, return just it as the only sub-question. "
            "List one per line, numbered '1.', '2.', etc. — nothing else.",
        ),
        ("human", "Question:\n{question}"),
    ]
)

_STEP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer this sub-question using only the numbered context passages "
            "and the answers to the earlier sub-questions below. Cite sources "
            "as [1], [2], etc. Be concise — this feeds into a larger answer, "
            "it isn't the final response.",
        ),
        (
            "human",
            "Earlier sub-questions and answers:\n{prior}\n\n"
            "Sub-question:\n{subquestion}\n\nContext:\n{context}",
        ),
    ]
)

_SUBQ_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)


class LeastToMostAgent(ReasoningStrategy):
    name = "least_to_most"

    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        subquestions = await self._order(question)

        chain: list[tuple[str, str]] = []
        for sq in subquestions:
            prior = (
                "\n\n".join(f"Q: {q}\nA: {a}" for q, a in chain)
                if chain
                else "(none yet)"
            )
            answer = await self._ainvoke(
                _STEP_PROMPT,
                {"prior": prior, "subquestion": sq, "context": context},
                temperature=_TEMPERATURE,
            )
            chain.append((sq, answer))

        notes = "\n\n".join(f"Sub-question: {q}\nSub-answer: {a}" for q, a in chain)
        return await self._finalize(question, context, notes=notes, answer_language=answer_language)

    async def _order(self, question: str) -> list[str]:
        raw = await self._ainvoke(_ORDER_PROMPT, {"question": question}, temperature=_TEMPERATURE)
        subquestions = [m.group(1).strip() for m in _SUBQ_RE.finditer(raw)]
        return subquestions[:_MAX_SUBQUESTIONS] or [question]
