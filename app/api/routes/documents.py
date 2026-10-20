"""Document management and vector indexing endpoints router."""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.document import DocumentUploadResponse, IndexResponse
from app.services.ingestion_service import (
    DocumentIngestionService,
    InvalidFileError,
    DocumentProcessingError,
)
from app.services.rag_service import RAGService, rag_service
from app.utils.file_utils import is_allowed_file_extension
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
ingestion_service = DocumentIngestionService()
rag_service = rag_service


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Uploads, validates, saves, parses, cleans, and chunks a PDF document."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    if not is_allowed_file_extension(file.filename, allowed_extensions={".pdf"}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format for '{file.filename}'. Only .pdf files are supported.",
        )

    logger.info(f"Received PDF upload request for file: '{file.filename}'")

    try:
        file_bytes = await file.read()
        if not file_bytes or len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty (0 bytes).",
            )

        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_size_bytes:
            raise HTTPException(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                detail=f"Uploaded file exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        saved_file_path = ingestion_service.save_file(
            file_bytes=file_bytes,
            original_filename=file.filename,
        )

        result = ingestion_service.process_document(
            file_path=saved_file_path,
            original_filename=file.filename,
        )

        return DocumentUploadResponse(
            filename=result.filename,
            status=result.status,
            message=result.message,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            file_path=result.file_path,
            file_size_bytes=len(file_bytes),
            chunks_summary=result.chunks,
        )

    except HTTPException:
        raise
    except InvalidFileError as e:
        logger.warning(f"File validation error on upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except DocumentProcessingError as e:
        logger.error(f"Document processing error for '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process PDF document: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during document upload for '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing the document.",
        )


@router.post("/index", response_model=IndexResponse, status_code=status.HTTP_200_OK)
async def index_document(file: UploadFile = File(...)) -> IndexResponse:
    """Uploads, ingests, generates embeddings, and indexes a PDF document into the FAISS vector store."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    if not is_allowed_file_extension(file.filename, allowed_extensions={".pdf"}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format for '{file.filename}'. Only .pdf files are supported for indexing.",
        )

    logger.info(f"Received PDF indexing request for file: '{file.filename}'")

    try:
        file_bytes = await file.read()
        if not file_bytes or len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty (0 bytes).",
            )

        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_size_bytes:
            raise HTTPException(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                detail=f"Uploaded file exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        saved_file_path = ingestion_service.save_file(
            file_bytes=file_bytes,
            original_filename=file.filename,
        )

        index_result = rag_service.index_document(
            file_path=saved_file_path,
            original_filename=file.filename,
        )

        return IndexResponse(
            filename=index_result["filename"],
            status=index_result["status"],
            page_count=index_result["page_count"],
            chunk_count=index_result["chunk_count"],
            message=index_result["message"],
        )

    except HTTPException:
        raise
    except InvalidFileError as e:
        logger.warning(f"File validation error during indexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except DocumentProcessingError as e:
        logger.error(f"Document processing error during indexing for '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process PDF document: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during document indexing for '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while indexing the document into the vector store.",
        )
