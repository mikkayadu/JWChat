"""Self-Reflection: draft an answer, critique it against the context and
citation rules, then revise into the final answer.

The critique pass is what gives this strategy distinct value over simply
sampling again — it can catch and correct issues a single pass might miss
(unsupported claims, mis-numbered citations, dropped parts of the question)
before the answer is finalized.
"""

from langchain_core.prompts import ChatPromptTemplate

from .base import ReasoningStrategy

_DRAFT_TEMPERATURE = 0.5
_CRITIQUE_TEMPERATURE = 0.3

_DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Draft a candidate answer to the question using only the numbered "
            "context passages below. Cite sources as [1], [2], etc. This is a "
            "draft for review, not the final answer — it's fine if it's rough.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nDraft an answer."),
    ]
)

_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are reviewing a draft answer for a context-only, citation-strict "
            "assistant. Check specifically:\n"
            "1. Is every claim backed by a cited passage?\n"
            "2. Do the citation numbers actually match what they're attached to?\n"
            "3. Does the draft fully address every part of the question using "
            "the available context?\n"
            "4. Does it introduce anything not present in the context?\n"
            "List concrete problems and the corrections to make — be specific "
            "about what to add, remove, or re-cite. If the draft already holds "
            "up, say so briefly.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nDraft:\n{draft}\n\n"
            "Critique this draft.",
        ),
    ]
)


class SelfReflectionAgent(ReasoningStrategy):
    name = "self_reflection"

    async def arun(self, question: str, context: str) -> str:
        draft = await self._ainvoke(
            _DRAFT_PROMPT,
            {"question": question, "context": context},
            temperature=_DRAFT_TEMPERATURE,
        )
        critique = await self._ainvoke(
            _CRITIQUE_PROMPT,
            {"question": question, "context": context, "draft": draft},
            temperature=_CRITIQUE_TEMPERATURE,
        )
        notes = f"Draft answer:\n{draft}\n\nCritique and corrections to apply:\n{critique}"
        return await self._finalize(question, context, notes=notes)
