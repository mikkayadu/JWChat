"""ReAct (Reason + Act), constrained to the provided context: "acting" means
re-focusing on one specific numbered passage already present in the context —
never an external tool or outside knowledge, since the system's citation rules
forbid both. Looping lets the model narrow in on the passages it actually
needs rather than reasoning over everything retrieved at once, which helps
when many passages were retrieved but only a few are truly relevant.
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from .base import ReasoningStrategy

_MAX_STEPS = 3
_STEP_TEMPERATURE = 0.3

_STEP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are working through a question step by step using only the "
            "numbered context passages below. At each step, briefly state your "
            "thought, then end with exactly one of:\n"
            "- 'ACTION: focus [n]' — to look more closely at passage number n\n"
            "- 'ACTION: answer' — once you have enough to answer the question",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            "Progress so far:\n{transcript}\n\nWhat's your next step?",
        ),
    ]
)

_ACTION_RE = re.compile(r"ACTION:\s*(focus\s*\[?(\d+)\]?|answer)", re.IGNORECASE)
_PASSAGE_HEADER_RE = re.compile(r"^\[(\d+)\]\s*Source:")


class ReActAgent(ReasoningStrategy):
    name = "react"

    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        passages = self._index_passages(context)
        transcript_lines: list[str] = []

        for step in range(_MAX_STEPS):
            transcript = "\n".join(transcript_lines) if transcript_lines else "(none yet)"
            step_text = await self._ainvoke(
                _STEP_PROMPT,
                {"question": question, "context": context, "transcript": transcript},
                temperature=_STEP_TEMPERATURE,
            )
            transcript_lines.append(f"Step {step + 1}: {step_text}")

            match = _ACTION_RE.search(step_text)
            if not match or match.group(1).lower().startswith("answer"):
                break

            n = int(match.group(2))
            passage = passages.get(n)
            observation = (
                f"Observation: passage [{n}] reads — {passage}"
                if passage
                else f"Observation: there is no passage [{n}] in the context."
            )
            transcript_lines.append(observation)

        notes = "\n".join(transcript_lines)
        return await self._finalize(question, context, notes=notes, answer_language=answer_language)

    @staticmethod
    def _index_passages(context: str) -> dict[int, str]:
        """Map passage number -> its full text block.

        Relies on `_build_context_node` joining sections shaped like
        '[i] Source: ...\\n<text>' with a blank line between them
        (api/services/chat_service.py:527-548) — if that joining convention
        changes, this parsing needs to change with it.
        """
        passages: dict[int, str] = {}
        for block in context.split("\n\n"):
            block = block.strip()
            header = _PASSAGE_HEADER_RE.match(block)
            if header:
                passages[int(header.group(1))] = block
        return passages
