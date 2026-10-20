"""Service module for building structured RAG context and source attribution from retrieved document chunks."""

from typing import List, Dict, Any, Tuple, Union, Optional
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging_config import logger


class ContextBuilder:
    """Builder responsible for filtering, formatting, bounding, and structuring retrieved documents into a RAG context."""

    def __init__(self, max_context_chars: int = settings.MAX_CONTEXT_CHARS) -> None:
        self.max_context_chars = max_context_chars
        logger.info(f"Initialized ContextBuilder (max_context_chars={self.max_context_chars})")

    def format_document(self, doc: Document, source_index: int) -> str:
        """Formats an individual document chunk into a standardized human-readable block."""
        meta = doc.metadata or {}
        orig_filename = meta.get("original_filename") or meta.get("source") or "unknown_document"
        page = meta.get("page", 1)
        chunk_id = meta.get("chunk_id", f"chunk_{source_index}")
        rank = meta.get("rank")
        content = doc.page_content.strip()

        rank_line = f"Rank: {rank}\n" if rank is not None else ""

        block = (
            f"[Source {source_index}]\n"
            f"Document: {orig_filename}\n"
            f"Page: {page}\n"
            f"{rank_line}"
            f"Chunk ID: {chunk_id}\n\n"
            f"{content}"
        )
        return block


    def build_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Generates a deduplicated list of source reference metadata from retrieved documents."""
        sources: List[Dict[str, Any]] = []
        seen_keys = set()

        for doc in documents:
            meta = doc.metadata or {}
            orig_filename = meta.get("original_filename") or meta.get("source") or "unknown_document"
            source_path = meta.get("source", "")
            page = meta.get("page", 1)
            chunk_id = meta.get("chunk_id", "")

            dedup_key = (orig_filename, page, chunk_id)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                sources.append(
                    {
                        "source": source_path,
                        "page": page,
                        "chunk_id": chunk_id,
                        "original_filename": orig_filename,
                    }
                )

        return sources

    def build_context(
        self, documents: Union[List[Document], List[Tuple[Document, float]]]
    ) -> Dict[str, Any]:
        """Constructs a deterministic, character-bounded RAG context string and source metadata from retrieved documents."""
        if not documents:
            logger.info("ContextBuilder received empty document list. Returning empty context payload.")
            return {
                "context": "",
                "sources": [],
                "retrieved_count": 0,
                "included_count": 0,
            }

        # Normalize and deduplicate document list if passed as tuples with scores
        clean_docs: List[Document] = []
        seen_chunks = set()
        for item in documents:
            if isinstance(item, tuple):
                doc, _ = item
            else:
                doc = item

            if doc and doc.page_content and doc.page_content.strip():
                meta = doc.metadata or {}
                chunk_key = (
                    meta.get("original_filename") or meta.get("source"),
                    meta.get("page"),
                    meta.get("chunk_id") or doc.page_content.strip()[:100],
                )
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    clean_docs.append(doc)

        retrieved_count = len(clean_docs)
        if retrieved_count == 0:
            return {
                "context": "",
                "sources": [],
                "retrieved_count": 0,
                "included_count": 0,
            }

        context_blocks: List[str] = []
        included_docs: List[Document] = []
        current_length = 0

        for idx, doc in enumerate(clean_docs, 1):
            block = self.format_document(doc, source_index=idx)
            block_length = len(block)

            # Check if adding this block exceeds maximum context character limit
            separator_length = 2 if context_blocks else 0
            if current_length + separator_length + block_length > self.max_context_chars:
                logger.warning(
                    f"Context length limit reached ({current_length + separator_length + block_length} > {self.max_context_chars}). "
                    f"Stopping context inclusion at {len(included_docs)} of {retrieved_count} document(s)."
                )
                break

            context_blocks.append(block)
            included_docs.append(doc)
            current_length += block_length + separator_length

        full_context = "\n\n".join(context_blocks)
        sources = self.build_sources(included_docs)

        logger.info(
            f"Successfully constructed RAG context: {len(full_context)} chars, "
            f"included {len(included_docs)} of {retrieved_count} retrieved document(s)."
        )

        return {
            "context": full_context,
            "sources": sources,
            "retrieved_count": retrieved_count,
            "included_count": len(included_docs),
        }
