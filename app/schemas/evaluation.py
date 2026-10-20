"""Schemas for RAG evaluation framework, answer quality metrics, and benchmarking."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """Schema representing an individual evaluation benchmark sample."""

    id: str = Field(..., description="Unique sample identifier")
    question: str = Field(..., min_length=1, max_length=2000, description="Evaluation question string")
    expected_answer: str = Field(..., description="Target ground-truth or reference answer string")
    category: str = Field(default="simple", description="Sample category: simple, complex, unanswerable, conversational")
    is_answerable: bool = Field(default=True, description="Whether question is answerable from indexed documents")
    expected_sources: List[str] = Field(default_factory=list, description="List of expected document filenames or chunk IDs")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Optional prior conversation turns")


class SampleEvaluationResult(BaseModel):
    """Schema representing the evaluation results for a single sample."""

    id: str = Field(..., description="Sample identifier")
    question: str = Field(..., description="Question evaluated")
    category: str = Field(..., description="Category of the sample")
    is_answerable: bool = Field(..., description="Whether question was expected to be answerable")
    generated_answer: str = Field(..., description="System generated answer")
    expected_answer: str = Field(..., description="Ground truth reference answer")
    route: Optional[str] = Field(default=None, description="Adaptive routing branch assigned")
    retrieved_sources: List[str] = Field(default_factory=list, description="Filenames/sources retrieved by the system")
    expected_sources: List[str] = Field(default_factory=list, description="Expected source references")
    hit_rate: Optional[float] = Field(default=None, description="Deterministic Recall@K (1.0 if expected source retrieved, else 0.0)")
    reciprocal_rank: Optional[float] = Field(default=None, description="Deterministic MRR score (1/rank of first retrieved expected source)")
    answer_f1: float = Field(..., description="Token-level F1 overlap score between generated and expected answer (0.0 to 1.0)")
    answer_precision: float = Field(..., description="Token-level precision score")
    answer_recall: float = Field(..., description="Token-level recall score")
    grounding_score: float = Field(..., description="Deterministic lexical grounding ratio of answer tokens in context (0.0 to 1.0)")
    is_refusal: bool = Field(..., description="Whether system correctly responded with standard refusal phrase")
    refusal_correct: Optional[bool] = Field(default=None, description="Whether refusal status matched is_answerable expectations")
    empty_context: bool = Field(..., description="Whether retrieved context was empty")
    missing_sources: bool = Field(..., description="Whether non-empty answer was generated without supporting sources")
    timings: Dict[str, float] = Field(default_factory=dict, description="Operational latency breakdown in milliseconds")
    status: str = Field(default="success", description="Sample evaluation execution status")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")


class EvaluationMetricsSummary(BaseModel):
    """Schema summarizing aggregate benchmark evaluation metrics."""

    total_samples: int = Field(..., description="Total number of evaluated samples")
    successful_runs: int = Field(..., description="Number of successfully processed samples")
    failed_runs: int = Field(..., description="Number of samples that encountered errors")
    mean_hit_rate: Optional[float] = Field(default=None, description="Mean Recall@K across answerable samples with expected sources")
    mean_mrr: Optional[float] = Field(default=None, description="Mean Reciprocal Rank across answerable samples with expected sources")
    mean_answer_f1: float = Field(..., description="Mean token F1 score across all evaluated samples")
    mean_grounding_score: float = Field(..., description="Mean lexical grounding score across all evaluated samples")
    refusal_accuracy: Optional[float] = Field(default=None, description="Accuracy of refusal for unanswerable questions")
    empty_context_count: int = Field(default=0, description="Total count of queries resulting in empty context")
    missing_sources_count: int = Field(default=0, description="Count of answers generated without source citations")
    mean_latencies_ms: Dict[str, float] = Field(default_factory=dict, description="Average latencies per operational stage in ms")
    deterministic_metrics_note: str = Field(
        default=(
            "Hit Rate, MRR, Token F1, Lexical Grounding Ratio, and Latencies are deterministic computations. "
            "Semantic equivalence and factual correctness remain approximate and should not be confused with human ground truth."
        ),
        description="Disclaimer clarifying metric limits",
    )


class EvaluationRunRequest(BaseModel):
    """Schema requesting an evaluation benchmark run."""

    dataset_path: Optional[str] = Field(
        default=None,
        description="Optional path to dataset JSON within safe data/ directory (defaults to configured path)",
    )
    custom_samples: Optional[List[EvaluationSample]] = Field(
        default=None,
        description="Optional list of custom evaluation samples to benchmark directly",
    )
    use_reranker: Optional[bool] = Field(
        default=None,
        description="Optional override for re-ranking activation during evaluation",
    )


class EvaluationRunResponse(BaseModel):
    """Schema for complete evaluation benchmark API response."""

    summary: EvaluationMetricsSummary = Field(..., description="Summary aggregate evaluation metrics")
    sample_results: List[SampleEvaluationResult] = Field(default_factory=list, description="Itemized results per sample")
    errors: List[str] = Field(default_factory=list, description="List of non-fatal errors or warnings")
    status: str = Field(default="success", description="Overall API operation status")
