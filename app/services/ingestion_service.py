"""Service module for document ingestion, PDF text extraction, cleaning, and chunking."""

import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging_config import logger
from app.utils.file_utils import (
    ensure_directory_exists,
    generate_safe_filename,
    is_allowed_file_extension,
    file_exists,
)
from app.schemas.document import DocumentProcessingResult, DocumentChunkSummary


class InvalidFileError(ValueError):
    """Raised when an uploaded or specified file fails validation."""
    pass


class DocumentProcessingError(Exception):
    """Raised when an error occurs during PDF parsing, cleaning, or chunking."""
    pass


class DocumentIngestionService:
    """Service responsible for validating, saving, parsing, cleaning, and chunking PDF documents."""

    def __init__(
        self,
        target_directory: Union[str, Path] = settings.DOCUMENTS_PATH,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        self.target_directory = ensure_directory_exists(target_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized DocumentIngestionService (target_dir='{self.target_directory}', "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
        )

    def validate_file(self, file_path: Union[str, Path], original_filename: Optional[str] = None) -> bool:
        """Validates that a target file exists, is non-empty, and has an allowed (.pdf) extension."""
        path = Path(file_path)
        check_name = original_filename or path.name

        if not is_allowed_file_extension(check_name, allowed_extensions={".pdf"}):
            logger.warning(f"File validation failed: '{check_name}' is not a supported .pdf document.")
            raise InvalidFileError(f"Unsupported file type for '{check_name}'. Only .pdf files are supported.")

        if not path.is_file():
            logger.warning(f"File validation failed: path '{path}' does not exist.")
            raise InvalidFileError(f"File '{check_name}' does not exist on disk.")

        if path.stat().st_size == 0:
            logger.warning(f"File validation failed: file '{path}' is empty (0 bytes).")
            raise InvalidFileError(f"File '{check_name}' is empty (0 bytes).")

        logger.info(f"File validation successful for: '{check_name}' ({path.stat().st_size} bytes)")
        return True

    def save_file(self, file_bytes: bytes, original_filename: str) -> Path:
        """Saves uploaded raw bytes safely to disk under data/documents/ using a sanitized filename."""
        if not file_bytes:
            raise InvalidFileError(f"Cannot save empty file '{original_filename}'.")

        if not is_allowed_file_extension(original_filename, allowed_extensions={".pdf"}):
            raise InvalidFileError(f"Unsupported file type for '{original_filename}'. Only .pdf files are supported.")

        safe_name = generate_safe_filename(original_filename)
        saved_path = self.target_directory / safe_name

        try:
            with open(saved_path, "wb") as f:
                f.write(file_bytes)
            logger.info(f"Document saved successfully: '{original_filename}' -> '{saved_path}'")
            return saved_path
        except Exception as e:
            logger.error(f"Failed to save file '{original_filename}' to disk: {str(e)}")
            raise IOError(f"Could not save file to disk: {str(e)}") from e

    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads and extracts page text from a PDF file using PyPDFLoader."""
        path = Path(file_path)
        self.validate_file(path)

        logger.info(f"Starting PDF text extraction for: '{path}'")
        try:
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            if not pages:
                raise DocumentProcessingError(f"No pages could be extracted from PDF: '{path.name}'")

            # Check if any page lacks textual content and attempt OCR fallback on embedded images
            from app.utils.ocr_utils import extract_ocr_text_from_pdf_page

            for idx, page in enumerate(pages):
                raw_text = page.page_content or ""
                if len(raw_text.strip()) < 10:
                    logger.info(
                        f"Page {idx + 1} of '{path.name}' has minimal text ({len(raw_text.strip())} chars). "
                        f"Attempting OCR fallback on embedded images..."
                    )
                    ocr_text = extract_ocr_text_from_pdf_page(path, idx)
                    if ocr_text.strip():
                        page.page_content = ocr_text
                        page.metadata["ocr_applied"] = True
                        logger.info(
                            f"OCR successfully extracted {len(ocr_text)} characters for page {idx + 1} of '{path.name}'"
                        )

            logger.info(f"PDF extraction completed: Extracted {len(pages)} pages from '{path.name}'")
            return pages
        except InvalidFileError:
            raise
        except Exception as e:
            logger.error(f"Error loading PDF file '{path}': {str(e)}")
            raise DocumentProcessingError(f"Failed to process PDF document: {str(e)}") from e

    def clean_text(self, raw_text: str) -> str:
        """Cleans and normalizes extracted text from PDF documents while preserving paragraph breaks."""
        if not raw_text:
            return ""

        # Normalize carriage returns and non-breaking spaces
        text = raw_text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

        # Split into paragraph blocks
        paragraphs = text.split("\n\n")
        cleaned_paragraphs = []

        for p in paragraphs:
            # Clean individual lines inside each paragraph block
            lines = p.split("\n")
            cleaned_lines = [" ".join(line.split()) for line in lines if line.strip()]
            cleaned_paragraph = " ".join(cleaned_lines)
            if cleaned_paragraph:
                cleaned_paragraphs.append(cleaned_paragraph)

        cleaned_text = "\n\n".join(cleaned_paragraphs)
        return cleaned_text

    def split_documents(self, documents: List[Document], original_filename: str) -> List[Document]:
        """Splits loaded pages into recursive character chunks and attaches metadata."""
        if not documents:
            return []

        # First clean text on each loaded page
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            if cleaned_content:
                cleaned_docs.append(
                    Document(
                        page_content=cleaned_content,
                        metadata=dict(doc.metadata),
                    )
                )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        raw_chunks = splitter.split_documents(cleaned_docs)
        logger.info(f"Split {len(documents)} page(s) into {len(raw_chunks)} text chunk(s)")

        filename_stem = Path(original_filename).stem
        safe_stem = re.sub(r"[^\w\-]", "_", filename_stem)

        final_chunks: List[Document] = []
        chunk_counters: Dict[int, int] = {}

        for chunk in raw_chunks:
            # PyPDFLoader page numbers are 0-indexed in metadata, convert to 1-indexed
            raw_page = chunk.metadata.get("page", 0)
            page_num = raw_page + 1 if isinstance(raw_page, int) else 1

            chunk_counters[page_num] = chunk_counters.get(page_num, 0) + 1
            chunk_idx = chunk_counters[page_num]

            chunk_id = f"{safe_stem}_p{page_num}_{chunk_idx:03d}"

            source_path = str(chunk.metadata.get("source", ""))

            chunk.metadata.update(
                {
                    "source": source_path,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "original_filename": original_filename,
                }
            )
            final_chunks.append(chunk)

        return final_chunks

    def process_document(
        self, file_path: Union[str, Path], original_filename: Optional[str] = None
    ) -> DocumentProcessingResult:
        """Executes full PDF ingestion pipeline: validate -> load -> clean -> split -> metadata."""
        path = Path(file_path)
        orig_name = original_filename or path.name

        logger.info(f"Starting complete document ingestion pipeline for: '{orig_name}'")

        # 1. Load document pages
        pages = self.load_document(path)

        # 2. Split documents and enrich metadata
        chunks = self.split_documents(pages, original_filename=orig_name)

        # 3. Build chunk summaries
        chunk_summaries = [
            DocumentChunkSummary(
                chunk_id=chunk.metadata["chunk_id"],
                page=chunk.metadata["page"],
                content_length=len(chunk.page_content),
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        logger.info(
            f"Successfully processed '{orig_name}': {len(pages)} pages, {len(chunks)} chunks created."
        )

        return DocumentProcessingResult(
            filename=path.name,
            original_filename=orig_name,
            file_path=str(path),
            page_count=len(pages),
            chunk_count=len(chunks),
            chunks=chunk_summaries,
            status="success" if chunks else "warning",
            message=(
                f"Successfully extracted {len(pages)} pages and created {len(chunks)} chunks."
                if chunks
                else f"Extracted {len(pages)} page(s) but no readable text chunks could be produced from '{orig_name}'."
            ),
        )
