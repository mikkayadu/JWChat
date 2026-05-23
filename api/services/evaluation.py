"""
RAG Evaluation Framework
─────────────────────
Provides evaluation capabilities using RAGAS framework to measure:
  - Faithfulness: Does the answer match the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Recall: Did we retrieve the relevant documents?
  - Context Precision: Are retrieved documents relevant?

Usage:
    from api.services.evaluation import RAGEvaluator
    
    evaluator = RAGEvaluator()
    results = evaluator.evaluate(test_set)  # test_set = list of {question, answer, contexts}
"""

import os
from typing import Optional
from loguru import logger


class RAGEvaluator:
    """Evaluate RAG system performance using RAGAS metrics."""
    
    def __init__(self) -> None:
        self._gemini_key = os.getenv("GEMINI_API_KEY")
        self._available = self._check_availability()
        
        if self._available:
            self._init_metrics()
        else:
            logger.warning("RAGAS evaluation not available. Install: pip install ragas")

    def _check_availability(self) -> bool:
        """Check if RAGAS is installed."""
        try:
            import ragas
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
            return True
        except ImportError:
            return False

    def _init_metrics(self) -> None:
        """Initialize RAGAS metrics."""
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
            
            self._metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_recall": context_recall,
                "context_precision": context_precision,
            }
            logger.info(f"RAGAS metrics initialized: {list(self._metrics.keys())}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAGAS metrics: {e}")
            self._available = False

    def evaluate(self, test_set: list[dict], threshold: float = 0.75) -> dict:
        """
        Evaluate RAG performance on a test set.
        
        Args:
            test_set: List of dicts with keys:
                - question: The user question
                - answer: The generated answer
                - contexts: List of context strings used
            threshold: Minimum score to consider "passing"
            
        Returns:
            Dict with scores and pass/fail status
        """
        if not self._available:
            return {"error": "RAGAS not available", "available": False}

        try:
            from ragas import evaluate
            from datasets import Dataset
            
            # Convert to RAGAS format
            dataset = Dataset.from_dict({
                "question": [item["question"] for item in test_set],
                "answer": [item["answer"] for item in test_set],
                "contexts": [item.get("contexts", []) for item in test_set],
            })
            
            # Run evaluation
            result = evaluate(dataset, metrics=list(self._metrics.values()))
            
            # Extract scores
            scores = {}
            for metric_name in self._metrics.keys():
                try:
                    scores[metric_name] = float(result[metric_name])
                except (KeyError, TypeError):
                    scores[metric_name] = 0.0
            
            # Determine pass/fail
            passed = all(scores.get(m, 0) >= threshold for m in ["faithfulness", "answer_relevancy"])
            
            return {
                "available": True,
                "scores": scores,
                "passed": passed,
                "threshold": threshold,
                "summary": self._generate_summary(scores, threshold),
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e), "available": True}

    def _generate_summary(self, scores: dict, threshold: float) -> str:
        """Generate a human-readable summary of evaluation results."""
        lines = ["RAG Evaluation Results:", "=" * 40]
        
        for metric, score in scores.items():
            status = "✓ PASS" if score >= threshold else "✗ FAIL"
            lines.append(f"  {metric:25s}: {score:.3f} {status}")
        
        lines.append("=" * 40)
        overall = "PASSED" if all(s >= threshold for s in scores.values()) else "FAILED"
        lines.append(f"Overall: {overall}")
        
        return "\n".join(lines)

    def quick_evaluate(self, question: str, answer: str, contexts: list[str]) -> dict:
        """
        Quick evaluation of a single Q&A pair.
        
        Args:
            question: The user question
            answer: The generated answer
            contexts: List of context strings used
            
        Returns:
            Dict with individual metric scores
        """
        return self.evaluate([{
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }])


def create_test_set_from_qa_pairs(qa_pairs: list[tuple[str, str]]) -> list[dict]:
    """
    Create a test set from question-answer pairs.
    Note: This requires manual context collection for full evaluation.
    
    Args:
        qa_pairs: List of (question, expected_answer) tuples
        
    Returns:
        List of test set dicts (contexts will be empty - need to fill manually)
    """
    return [
        {
            "question": q,
            "expected_answer": a,
            "contexts": [],  # To be filled by calling RAG system
        }
        for q, a in qa_pairs
    ]


# Example test set for JWChat
EXAMPLE_TEST_SET = [
    {
        "question": "What does the Bible say about love?",
        "expected_answer": "The Bible emphasizes love as central to Christian teaching...",
        "contexts": [],  # Fill with actual retrieved contexts
    },
    {
        "question": "How did Arabic become a learned language?",
        "expected_answer": "Arabic became the language of learning in Córdoba...",
        "contexts": [],
    },
]
