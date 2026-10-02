"""Tree-of-Thoughts (single-level BFS): sketch a few distinct reasoning
branches, self-score them, and finalize from the strongest one.

This is deliberately a *shallow* version of ToT — full breadth/depth search
would multiply LLM calls (and latency/cost) far beyond what one ensemble
member should spend on a single request. Generating the branches at higher
temperature is what makes them genuinely different from each other (and from
the other strategies' single-pass answers).
"""

from langchain_core.prompts import ChatPromptTemplate

from .base import ReasoningStrategy

_BRANCH_COUNT = 3
_BRANCH_TEMPERATURE = 0.6

_BRANCHES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are exploring different ways to approach a question using only "
            "the numbered context passages provided below. Do not write a final "
            "answer — only sketch reasoning approaches.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            f"Sketch {_BRANCH_COUNT} distinct reasoning approaches to answering "
            "this question from the context above — different angles, different "
            "passage groupings, or different ways of structuring the answer. "
            "Label each 'Branch 1:', 'Branch 2:', etc. Keep each to a few sentences.",
        ),
    ]
)

_SCORE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging which of several reasoning approaches would best "
            "support a complete, accurate, well-cited answer to the question "
            "from the given context.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            "Reasoning branches:\n{branches}\n\n"
            "Which single branch would best support the final answer? "
            "Reply with just that branch's full text, nothing else.",
        ),
    ]
)


class ToTAgent(ReasoningStrategy):
    name = "tot"

    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        branches = await self._ainvoke(
            _BRANCHES_PROMPT,
            {"question": question, "context": context},
            temperature=_BRANCH_TEMPERATURE,
        )
        best_branch = await self._ainvoke(
            _SCORE_PROMPT,
            {"question": question, "context": context, "branches": branches},
            temperature=0,
        )
        return await self._finalize(question, context, notes=best_branch, answer_language=answer_language)
