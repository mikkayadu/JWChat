"""Chain-of-Thought: reason step-by-step through the numbered context
passages internally, then surface only the final cited answer.

A single call with a CoT-flavored prompt run at higher temperature than the
deterministic baseline — different sampling plus a different framing of the
task is what lets this strategy's reasoning path (and therefore its answer)
genuinely diverge from ``StandardAgent``, which is what makes voting between
them meaningful rather than redundant.
"""

from langchain_core.prompts import ChatPromptTemplate

from .base import CITATION_RULES, ReasoningStrategy, language_rules

_TEMPERATURE = 0.4

_COT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CITATION_RULES + "\n\n"
            "REASONING APPROACH:\n"
            "Before answering, work through the numbered context passages "
            "step by step in your head: identify which passages are relevant, "
            "what each one contributes, and how they fit together. "
            "Do NOT show this reasoning in your reply — output only the "
            "final answer in the response format described above."
            "{language_rules}",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\n"
            "Think it through step by step, then give your final answer with citations.",
        ),
    ]
)


class CoTAgent(ReasoningStrategy):
    name = "cot"

    async def arun(self, question: str, context: str, answer_language: str = "en") -> str:
        return await self._ainvoke(
            _COT_PROMPT,
            {
                "question": question,
                "context": context,
                "language_rules": language_rules(answer_language),
            },
            temperature=_TEMPERATURE,
        )
