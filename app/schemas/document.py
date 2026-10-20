"""Schemas for document upload, ingestion, and vector store indexing endpoints."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunkSummary(BaseModel):
    """Summary representation of an individual document chunk."""

    chunk_id: str = Field(..., description="Unique deterministic chunk identifier")
    page: int = Field(..., description="Page number of origin in the document")
    content_length: int = Field(..., description="Character length of chunk text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class DocumentProcessingResult(BaseModel):
    """Result of full document processing pipeline execution."""

    filename: str = Field(..., description="Sanitized stored filename")
    original_filename: str = Field(..., description="Original filename before sanitization")
    file_path: str = Field(..., description="Path where document is stored")
    page_count: int = Field(..., description="Total pages extracted from PDF")
    chunk_count: int = Field(..., description="Total text chunks generated")
    chunks: List[DocumentChunkSummary] = Field(
        default_factory=list, description="List of generated chunk summaries"
    )
    status: str = Field(default="success", description="Status of document processing")
    message: str = Field(..., description="Details regarding the processing status")


class DocumentUploadResponse(BaseModel):
    """Schema for document upload API response."""

    filename: str = Field(..., description="Sanitized stored filename")
    status: str = Field(..., description="Status of the upload operation")
    message: str = Field(..., description="Details regarding processing outcome")
    page_count: int = Field(..., description="Total number of pages processed")
    chunk_count: int = Field(..., description="Total number of chunks generated")
    file_path: Optional[str] = Field(default=None, description="Saved file location on disk")
    file_size_bytes: Optional[int] = Field(default=None, description="Size of uploaded file in bytes")
    chunks_summary: Optional[List[DocumentChunkSummary]] = Field(
        default=None, description="Summaries of generated document chunks"
    )


class IndexResponse(BaseModel):
    """Schema for document indexing API response."""

    filename: str = Field(..., description="Name of the indexed document file")
    status: str = Field(..., description="Status of vector indexing operation (e.g. indexed, duplicate)")
    page_count: int = Field(..., description="Total number of pages extracted")
    chunk_count: int = Field(..., description="Total number of text chunks generated")
    message: str = Field(..., description="Informational message regarding indexing outcome")
