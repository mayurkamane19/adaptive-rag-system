"""Service module providing RAG evaluation, answer quality metrics, and benchmark test runners."""

import json
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from app.core.config import settings
from app.core.logging_config import logger
from app.schemas.evaluation import (
    EvaluationSample,
    SampleEvaluationResult,
    EvaluationMetricsSummary,
    EvaluationRunResponse,
)
from app.services.rag_service import RAGService, rag_service as default_rag_service

# Standard common English stop words to exclude from lexical grounding calculation
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "he", "she", "we", "you", "i",
    "me", "my", "our", "us", "your", "his", "her", "as", "not", "no", "if", "than",
}

REFUSAL_PATTERNS = [
    "couldn't find enough information",
    "could not find enough information",
    "cannot find enough information",
    "not enough information in the provided documents",
    "no relevant information",
]


class EvaluationService:
    """Service responsible for computing deterministic RAG evaluation metrics and running benchmarks."""

    def __init__(self, rag_service: Optional[RAGService] = None) -> None:
        self.rag_service = rag_service or default_rag_service
        logger.info("Initialized EvaluationService")

    @staticmethod
    def normalize_text(text: str) -> List[str]:
        """Normalizes text by lowercasing, removing punctuation, and splitting into tokens."""
        if not text:
            return []
        cleaned = text.lower()
        cleaned = cleaned.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        return [t for t in cleaned.split() if t]

    @classmethod
    def compute_token_f1(cls, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """Calculates deterministic token-level precision, recall, and F1 overlap between two texts.

        Returns:
            Tuple of (f1, precision, recall) rounded to 4 decimal places.
        """
        pred_tokens = cls.normalize_text(prediction)
        gold_tokens = cls.normalize_text(ground_truth)

        if not pred_tokens and not gold_tokens:
            return 1.0, 1.0, 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0, 0.0, 0.0

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return round(f1, 4), round(precision, 4), round(recall, 4)

    @staticmethod
    def matches_source(retrieved_item: str, expected_source: str) -> bool:
        """Determines whether a retrieved source filename/path matches an expected source identifier."""
        ret_clean = Path(retrieved_item).name.lower()
        exp_clean = Path(expected_source).name.lower()
        return exp_clean in ret_clean or ret_clean in exp_clean

    @classmethod
    def compute_hit_rate(cls, retrieved_sources: List[str], expected_sources: List[str]) -> Optional[float]:
        """Computes deterministic Recall@K / Hit Rate.

        Returns:
            1.0 if at least one expected source is found in retrieved sources, else 0.0.
            Returns None if expected_sources is empty (not applicable).
        """
        if not expected_sources:
            return None
        if not retrieved_sources:
            return 0.0

        for exp in expected_sources:
            for ret in retrieved_sources:
                if cls.matches_source(ret, exp):
                    return 1.0
        return 0.0

    @classmethod
    def compute_mrr(cls, retrieved_sources: List[str], expected_sources: List[str]) -> Optional[float]:
        """Computes deterministic Mean Reciprocal Rank (MRR) for the first matching source.

        Returns:
            1 / rank (1-indexed) of first matched expected source, or 0.0 if not found.
            Returns None if expected_sources is empty.
        """
        if not expected_sources:
            return None
        if not retrieved_sources:
            return 0.0

        for rank_idx, ret in enumerate(retrieved_sources, 1):
            for exp in expected_sources:
                if cls.matches_source(ret, exp):
                    return round(1.0 / rank_idx, 4)
        return 0.0

    @classmethod
    def is_standard_refusal(cls, answer: str) -> bool:
        """Checks whether the generated answer is a standard insufficient-context refusal."""
        if not answer:
            return False
        ans_lower = answer.lower()
        return any(pattern in ans_lower for pattern in REFUSAL_PATTERNS)

    @classmethod
    def compute_grounding_score(cls, generated_answer: str, context: str) -> float:
        """Computes deterministic lexical grounding ratio of generated answer content tokens in context.

        Returns:
            Float between 0.0 and 1.0 representing percentage of answer content words found in context.
        """
        if not generated_answer or not generated_answer.strip():
            return 0.0

        # If answer is a refusal and context is empty, refusal is 100% grounded/faithful to absence of context
        if cls.is_standard_refusal(generated_answer) and (not context or not context.strip()):
            return 1.0

        if not context or not context.strip():
            return 0.0

        ans_tokens = cls.normalize_text(generated_answer)
        # Filter stopwords and numeric citation markers like [source 1]
        content_tokens = [t for t in ans_tokens if t not in STOP_WORDS and not (t.startswith("source") or t.isdigit())]

        if not content_tokens:
            content_tokens = ans_tokens

        if not content_tokens:
            return 1.0

        context_clean = " ".join(cls.normalize_text(context))
        grounded_count = sum(1 for t in content_tokens if t in context_clean)

        return round(grounded_count / len(content_tokens), 4)

    def load_dataset(self, dataset_path: Optional[str] = None) -> List[EvaluationSample]:
        """Loads and validates evaluation dataset from disk safely."""
        target_path_str = dataset_path or settings.EVALUATION_DATASET_PATH
        target_path = Path(target_path_str).resolve()

        # Security check: ensure path is within project directory
        cwd = Path.cwd().resolve()
        try:
            target_path.relative_to(cwd)
        except ValueError as e:
            logger.error(f"Security validation failed: dataset path '{target_path}' is outside project root '{cwd}'")
            raise ValueError("Dataset path must be located within the application workspace.") from e

        if not target_path.is_file():
            logger.error(f"Evaluation dataset file not found at: '{target_path}'")
            raise FileNotFoundError(f"Evaluation dataset file not found at '{target_path}'")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Evaluation dataset JSON root must be an array of samples.")

            samples = [EvaluationSample(**item) for item in data]
            logger.info(f"Successfully loaded {len(samples)} evaluation samples from '{target_path}'")
            return samples
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {str(e)}")
            raise

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        rag_service: Optional[RAGService] = None,
    ) -> SampleEvaluationResult:
        """Executes RAG pipeline for a single sample and calculates all evaluation metrics."""
        service = rag_service or self.rag_service
        start_time = time.perf_counter()

        try:
            if sample.conversation_history:
                # Use conversational RAG endpoint for multi-turn questions
                chat_res = service.conversational_rag_chat(
                    query=sample.question,
                    session_id=f"eval-session-{sample.id}",
                )
                generated_answer = chat_res.get("answer", "")
                route = chat_res.get("route", "simple")
                raw_sources = chat_res.get("sources", [])
                retrieved_sources = [
                    s.get("original_filename") or s.get("source") or "" for s in raw_sources if isinstance(s, dict)
                ]
                timings = chat_res.get("timings") or {}
                # Extract context if present or build from sources
                context = chat_res.get("context", "")
            else:
                # Use single-turn RAG answer generation
                rag_res = service.generate_rag_answer(query=sample.question)
                generated_answer = rag_res.get("answer", "")
                route = rag_res.get("route", "simple")
                raw_sources = rag_res.get("sources", [])
                retrieved_sources = [
                    s.get("original_filename") or s.get("source") or "" for s in raw_sources if isinstance(s, dict)
                ]
                timings = rag_res.get("timings") or {}
                context = rag_res.get("context", "")

            # If context was not returned directly in response, attempt adaptive context lookup for grounding evaluation
            if not context and retrieved_sources:
                try:
                    ctx_res = service.adaptive_retrieve_context(query=sample.question)
                    context = ctx_res.get("context", "")
                except Exception:
                    context = ""

            total_elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if "total_ms" not in timings:
                timings["total_ms"] = total_elapsed_ms

            # Compute deterministic metrics
            f1, prec, rec = self.compute_token_f1(generated_answer, sample.expected_answer)
            hit_rate = self.compute_hit_rate(retrieved_sources, sample.expected_sources)
            mrr = self.compute_mrr(retrieved_sources, sample.expected_sources)
            grounding_score = self.compute_grounding_score(generated_answer, context)
            is_refusal = self.is_standard_refusal(generated_answer)
            empty_context = len(context.strip()) == 0
            missing_sources = len(retrieved_sources) == 0 and not is_refusal

            refusal_correct = None
            if not sample.is_answerable:
                refusal_correct = is_refusal or empty_context
            else:
                refusal_correct = not is_refusal

            return SampleEvaluationResult(
                id=sample.id,
                question=sample.question,
                category=sample.category,
                is_answerable=sample.is_answerable,
                generated_answer=generated_answer,
                expected_answer=sample.expected_answer,
                route=route,
                retrieved_sources=retrieved_sources,
                expected_sources=sample.expected_sources,
                hit_rate=hit_rate,
                reciprocal_rank=mrr,
                answer_f1=f1,
                answer_precision=prec,
                answer_recall=rec,
                grounding_score=grounding_score,
                is_refusal=is_refusal,
                refusal_correct=refusal_correct,
                empty_context=empty_context,
                missing_sources=missing_sources,
                timings=timings,
                status="success",
                error=None,
            )

        except Exception as e:
            logger.error(f"Evaluation failed for sample '{sample.id}': {str(e)}", exc_info=True)
            total_elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return SampleEvaluationResult(
                id=sample.id,
                question=sample.question,
                category=sample.category,
                is_answerable=sample.is_answerable,
                generated_answer="",
                expected_answer=sample.expected_answer,
                route=None,
                retrieved_sources=[],
                expected_sources=sample.expected_sources,
                hit_rate=0.0 if sample.expected_sources else None,
                reciprocal_rank=0.0 if sample.expected_sources else None,
                answer_f1=0.0,
                answer_precision=0.0,
                answer_recall=0.0,
                grounding_score=0.0,
                is_refusal=False,
                refusal_correct=False,
                empty_context=True,
                missing_sources=False,
                timings={"total_ms": total_elapsed_ms},
                status="failed",
                error=str(e),
            )

    def run_evaluation(
        self,
        dataset_path: Optional[str] = None,
        custom_samples: Optional[List[EvaluationSample]] = None,
        use_reranker: Optional[bool] = None,
    ) -> EvaluationRunResponse:
        """Executes full evaluation suite across configured or custom samples and generates aggregated summary."""
        errors: List[str] = []

        if custom_samples:
            samples = custom_samples
        else:
            try:
                samples = self.load_dataset(dataset_path)
            except Exception as e:
                return EvaluationRunResponse(
                    summary=EvaluationMetricsSummary(
                        total_samples=0,
                        successful_runs=0,
                        failed_runs=0,
                        mean_answer_f1=0.0,
                        mean_grounding_score=0.0,
                    ),
                    sample_results=[],
                    errors=[f"Failed to load dataset: {str(e)}"],
                    status="error",
                )

        if not samples:
            return EvaluationRunResponse(
                summary=EvaluationMetricsSummary(
                    total_samples=0,
                    successful_runs=0,
                    failed_runs=0,
                    mean_answer_f1=0.0,
                    mean_grounding_score=0.0,
                ),
                sample_results=[],
                errors=["Dataset contains no samples to evaluate."],
                status="empty",
            )

        logger.info(f"Starting evaluation benchmark on {len(samples)} sample(s)...")

        # Temporarily override re-ranker setting if explicitly specified
        original_reranker_setting = settings.RERANKER_ENABLED
        if use_reranker is not None:
            settings.RERANKER_ENABLED = use_reranker

        results: List[SampleEvaluationResult] = []
        try:
            for s in samples:
                res = self.evaluate_sample(s)
                results.append(res)
                if res.status != "success" and res.error:
                    errors.append(f"Sample {s.id} error: {res.error}")
        finally:
            if use_reranker is not None:
                settings.RERANKER_ENABLED = original_reranker_setting

        # Compute summary metrics
        successful = [r for r in results if r.status == "success"]
        failed_count = len(results) - len(successful)

        hit_rates = [r.hit_rate for r in successful if r.hit_rate is not None]
        mrrs = [r.reciprocal_rank for r in successful if r.reciprocal_rank is not None]
        f1_scores = [r.answer_f1 for r in successful]
        grounding_scores = [r.grounding_score for r in successful]

        unanswerable_samples = [r for r in successful if not r.is_answerable]
        refusal_correct_count = sum(1 for r in unanswerable_samples if r.refusal_correct)
        refusal_acc = (
            round(refusal_correct_count / len(unanswerable_samples), 4) if unanswerable_samples else None
        )

        mean_hit = round(sum(hit_rates) / len(hit_rates), 4) if hit_rates else None
        mean_mrr = round(sum(mrrs) / len(mrrs), 4) if mrrs else None
        mean_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
        mean_grounding = round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 0.0

        empty_ctx_count = sum(1 for r in successful if r.empty_context)
        missing_src_count = sum(1 for r in successful if r.missing_sources)

        # Average stage latencies
        stage_sums: Dict[str, float] = {}
        stage_counts: Dict[str, int] = {}
        for r in successful:
            for stage, ms in r.timings.items():
                if isinstance(ms, (int, float)) and (stage.endswith("_ms") or stage == "total_ms"):
                    stage_sums[stage] = stage_sums.get(stage, 0.0) + ms
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1

        mean_latencies = {
            stage: round(stage_sums[stage] / stage_counts[stage], 2)
            for stage in stage_sums
        }

        summary = EvaluationMetricsSummary(
            total_samples=len(samples),
            successful_runs=len(successful),
            failed_runs=failed_count,
            mean_hit_rate=mean_hit,
            mean_mrr=mean_mrr,
            mean_answer_f1=mean_f1,
            mean_grounding_score=mean_grounding,
            refusal_accuracy=refusal_acc,
            empty_context_count=empty_ctx_count,
            missing_sources_count=missing_src_count,
            mean_latencies_ms=mean_latencies,
        )

        logger.info(
            f"Evaluation benchmark complete: {len(successful)}/{len(samples)} passed. "
            f"Mean F1: {mean_f1}, Mean Hit Rate: {mean_hit}, Mean MRR: {mean_mrr}"
        )

        return EvaluationRunResponse(
            summary=summary,
            sample_results=results,
            errors=errors,
            status="success" if failed_count == 0 else "partial_success",
        )


evaluation_service = EvaluationService()
