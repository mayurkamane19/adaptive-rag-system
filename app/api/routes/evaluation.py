"""RAG Evaluation, answer quality metrics, and benchmarking API endpoints."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field

from app.schemas.evaluation import (
    EvaluationSample,
    EvaluationRunRequest,
    EvaluationRunResponse,
)
from app.services.evaluation_service import evaluation_service
from app.core.config import settings
from app.core.security import verify_api_key
from app.core.logging_config import logger

router = APIRouter(prefix="/api/v1/evaluation", tags=["RAG Evaluation & Quality"])


class DatasetInfoResponse(BaseModel):
    """Schema for inspection of evaluation benchmark dataset."""

    dataset_path: str = Field(..., description="Resolved dataset path on disk")
    total_samples: int = Field(..., description="Number of evaluation samples in dataset")
    samples: List[EvaluationSample] = Field(..., description="Evaluation benchmark samples")


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def run_evaluation(request: Optional[EvaluationRunRequest] = None) -> EvaluationRunResponse:
    """Runs the deterministic RAG evaluation benchmark across the configured or custom dataset."""
    req = request or EvaluationRunRequest()
    logger.info(
        f"API received evaluation run request (dataset_path='{req.dataset_path}', "
        f"custom_samples_count={len(req.custom_samples) if req.custom_samples else 0}, "
        f"use_reranker={req.use_reranker})"
    )

    try:
        response = evaluation_service.run_evaluation(
            dataset_path=req.dataset_path,
            custom_samples=req.custom_samples,
            use_reranker=req.use_reranker,
        )
        return response
    except FileNotFoundError as e:
        logger.error(f"Evaluation dataset file not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        logger.warning(f"Invalid evaluation request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error executing evaluation run: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation execution failed: {str(e)}",
        )


@router.get("/dataset", response_model=DatasetInfoResponse, status_code=status.HTTP_200_OK)
async def get_evaluation_dataset(
    dataset_path: Optional[str] = Query(default=None, description="Optional path to dataset within workspace"),
) -> DatasetInfoResponse:
    """Retrieves and inspects the benchmark evaluation dataset samples."""
    target_path = dataset_path or settings.EVALUATION_DATASET_PATH
    logger.info(f"API inspecting evaluation dataset at '{target_path}'")

    try:
        samples = evaluation_service.load_dataset(dataset_path=target_path)
        return DatasetInfoResponse(
            dataset_path=target_path,
            total_samples=len(samples),
            samples=samples,
        )
    except FileNotFoundError as e:
        logger.error(f"Evaluation dataset not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        logger.warning(f"Invalid dataset path parameter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error loading evaluation dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation dataset: {str(e)}",
        )
