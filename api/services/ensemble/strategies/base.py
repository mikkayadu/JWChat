"""Shared interface and utilities for ensemble reasoning strategies.

Every strategy ultimately funnels its reasoning into the same cited-answer
shape — plain prose with inline ``[n]`` citations and no trailing source list,
matching ``ChatService._answer_prompt`` — so the orchestrator's
embedding-based majority vote compares like-for-like text. Strategies that
returned differently-shaped output (one its reasoning trace, another a terse
answer) would skew that comparison.
"""

import time
from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import languages

# Mirrors the citation/context-only rules baked into ChatService._answer_prompt
# (api/services/chat_service.py:162-187) so every strategy's final answer is
# bound by the same constraints as the baseline.
CITATION_RULES = (
    "You are a knowledgeable assistant specializing in Jehovah's Witness teachings, "
    "beliefs, and publications — drawing exclusively from their official books, magazines, "
    "and website content (e.g., jw.org).\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY from the provided context. Do not use outside knowledge or inference.\n"
    "2. If the answer is not found in the context, respond exactly: "
    "'I could not find information about this in the provided sources.'\n"
    "3. Cite every factual claim with inline source references like [1], [2], etc.\n"
    "4. Explain and restate the source content clearly in your own words — "
    "but never introduce claims, conclusions, or implications not present in the source.\n\n"
    "RESPONSE FORMAT:\n"
    "- Be concise and direct.\n"
    "- Place citations immediately after the sentence they support.\n"
    "- If multiple sources support a point, list all relevant citations: [1][3].\n"
    "- Do not append a list of the sources at the end of your response."
)


def language_rules(answer_language: str) -> str:
    """Prompt rules that make the answer come out in ``answer_language``.

    Shared by ChatService._answer_prompt and every strategy's final step.
    Empty for English, so English prompts are exactly what they were before
    languages existed. Appended after the other rules so its wording rules
    (not-found sentence, book names) take precedence.
    """
    lang = languages.get(answer_language)
    if lang.code == languages.DEFAULT.code:
        return ""
    example = f"{lang.book_names[42]} 3:16"  # John 3:16
    return (
        "\n\nLANGUAGE RULES (these override any wording instruction above):\n"
        f"- Write the whole answer in {lang.name}.\n"
        "- Each context passage is labelled with its language. Translate what you use "
        f"from passages in other languages faithfully into {lang.name}: do not add, "
        "drop or soften anything in translation.\n"
        f"- If any passage you cite is not in {lang.name}, begin the answer with this "
        f'sentence, exactly: "{lang.messages.english_sources_note}"\n'
        "- If the context does not contain the answer, respond exactly: "
        f'"{lang.messages.not_found}"\n'
        f"- Write scripture references with these {lang.name} Bible book names, in "
        f"Bible order: {', '.join(lang.book_names)}. Keep chapter and verse numbers as "
        f'digits, e.g. "{example}".'
    )


FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CITATION_RULES + "{language_rules}"),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            "{notes_block}"
            "Return an answer with citations.",
        ),
    ]
)


class ReasoningStrategy(ABC):
    """Base class for an ensemble reasoning strategy.

    Subclasses implement :meth:`arun`, returning a final cited-answer string.
    Most should produce that string by calling :meth:`_finalize` as their last
    step — it funnels whatever intermediate reasoning ("notes") the strategy
    generated through the same citation-rules prompt every other strategy
    converges on, keeping outputs structurally comparable for the vote.

    The orchestrator calls :meth:`run_logged`, not :meth:`arun` directly — it
    wraps the strategy with ``[ENSEMBLE:{name}]`` start/finish/duration/error
    logging and tracks how many LLM calls the strategy made.
    """

    name: str = "strategy"

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._call_count = 0

    @abstractmethod
    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        """Produce the final cited answer for this strategy, in ``answer_language``.

        The language is a parameter rather than instance state because one
        strategy instance is shared by concurrent requests.
        """

    async def _ainvoke(
        self,
        prompt: ChatPromptTemplate,
        variables: dict,
        *,
        temperature: float | None = None,
    ) -> str:
        """Invoke a ``prompt | llm`` chain once and return its text content.

        Pass ``temperature`` to sample this particular call differently from
        the strategy's base LLM (e.g. higher temperature to generate diverse
        reasoning branches, then 0 to score them deterministically) without
        needing a separate ``BaseChatModel`` instance per setting.
        Also tracks the call for the strategy's ``[ENSEMBLE:{name}]`` log line.
        """
        self._call_count += 1
        llm = self._llm.bind(temperature=temperature) if temperature is not None else self._llm
        chain = prompt | llm
        response = await chain.ainvoke(variables)
        text = getattr(response, "content", "") or str(response)
        return text.strip()

    async def _finalize(
        self,
        question: str,
        context: str,
        notes: str = "",
        *,
        answer_language: str = "en",
        temperature: float = 0.0,
    ) -> str:
        """Funnel intermediate reasoning into the converged, citation-compliant
        final-answer shape every strategy must return.

        Defaults to ``temperature=0`` — the converging step should be precise
        and deterministic regardless of how exploratory the reasoning that fed
        it was. Strategies whose *variation* lives in this final step itself
        (e.g. self-consistency sampling) can override it.
        """
        notes_block = (
            "Working notes from your reasoning so far (for your reference only — "
            f"do not repeat them verbatim, just use them to inform your answer):\n{notes}\n\n"
            if notes.strip()
            else ""
        )
        text = await self._ainvoke(
            FINAL_ANSWER_PROMPT,
            {
                "question": question,
                "context": context,
                "notes_block": notes_block,
                "language_rules": language_rules(answer_language),
            },
            temperature=temperature,
        )
        return text or "I could not generate a response."

    async def run_logged(self, question: str, context: str, answer_language: str = "en") -> str:
        """Entry point the orchestrator calls — wraps :meth:`arun` with
        ``[ENSEMBLE:{name}]`` timing, call-count, and error logging."""
        self._call_count = 0
        start = time.time()
        logger.info(f"[ENSEMBLE:{self.name}] starting")
        try:
            answer = await self.arun(question, context, answer_language)
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            logger.warning(
                f"[ENSEMBLE:{self.name}] failed after {duration_ms:.0f}ms "
                f"({self._call_count} LLM call(s)): {e}"
            )
            raise
        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"[ENSEMBLE:{self.name}] answered in {duration_ms:.0f}ms "
            f"({self._call_count} LLM call(s))"
        )
        return answer
