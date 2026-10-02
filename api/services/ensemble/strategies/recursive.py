"""Recursive map-reduce: chunk the context, draft a partial note per chunk,
then recursively combine pairs of notes into one consolidated set, and
finalize from that — useful when a lot of context was retrieved and a single
pass risks diluting attention across too much text at once.

Depth is bounded by halving the note count each round, so this never runs
away — for the retrieval sizes typical here (a handful of passages), it
resolves in one or two reduce rounds. Each round runs concurrently.
"""

import asyncio
import re

from langchain_core.prompts import ChatPromptTemplate

from .base import ReasoningStrategy

_CHUNK_SIZE = 3
_TEMPERATURE = 0.3

_PARTIAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize what these context passages contribute toward answering "
            "the question, citing them as [1], [2], etc. This is a partial "
            "working note, not a final answer — be concise.",
        ),
        ("human", "Question:\n{question}\n\nPassages:\n{chunk}"),
    ]
)

_COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Combine these partial notes into one consolidated set of notes "
            "toward answering the question — merge overlapping points, drop "
            "nothing with a citation. This is still a working note, not the "
            "final answer.",
        ),
        ("human", "Question:\n{question}\n\nPartial notes:\n{notes}"),
    ]
)

_PASSAGE_HEADER_RE = re.compile(r"^\[\d+\]\s*Source:")


class RecursiveAgent(ReasoningStrategy):
    name = "recursive"

    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        passages = [
            block.strip()
            for block in context.split("\n\n")
            if _PASSAGE_HEADER_RE.match(block.strip())
        ]
        if not passages:
            return await self._finalize(question, context, answer_language=answer_language)

        chunks = [
            "\n\n".join(passages[i : i + _CHUNK_SIZE])
            for i in range(0, len(passages), _CHUNK_SIZE)
        ]

        notes = list(
            await asyncio.gather(
                *(
                    self._ainvoke(
                        _PARTIAL_PROMPT,
                        {"question": question, "chunk": chunk},
                        temperature=_TEMPERATURE,
                    )
                    for chunk in chunks
                )
            )
        )

        while len(notes) > 1:
            pairs = [notes[i : i + 2] for i in range(0, len(notes), 2)]
            notes = list(await asyncio.gather(*(self._reduce(question, pair) for pair in pairs)))

        return await self._finalize(question, context, notes=notes[0], answer_language=answer_language)

    async def _reduce(self, question: str, pair: list[str]) -> str:
        if len(pair) == 1:
            return pair[0]
        combined = "\n\n---\n\n".join(pair)
        return await self._ainvoke(
            _COMBINE_PROMPT,
            {"question": question, "notes": combined},
            temperature=_TEMPERATURE,
        )
