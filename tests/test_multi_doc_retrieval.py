"""Regression tests for multi-document indexing, disk synchronization, and retrieval accuracy."""

import pytest
from langchain_core.documents import Document

from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService


@pytest.fixture(scope="module")
def shared_embedding_service() -> EmbeddingService:
    """Fixture providing a shared embedding service for tests."""
    return EmbeddingService()


@pytest.fixture
def multi_doc_retrieval_service(tmp_path, shared_embedding_service) -> RetrievalService:
    """Fixture providing an isolated RetrievalService instance."""
    db_path = tmp_path / "multi_doc_vectorstore"
    return RetrievalService(vector_db_path=db_path, embedding_service=shared_embedding_service)


def test_multi_document_indexing_and_targeted_retrieval(multi_doc_retrieval_service):
    """Verifies that indexing multiple distinct documents returns the correct document for specific queries."""
    service = multi_doc_retrieval_service

    # Document A: Leave policy
    doc_a_chunks = [
        Document(
            page_content="Leave Policy 2026 Employees receive 20 days of paid annual leave.",
            metadata={
                "source": "leave_policy.pdf",
                "page": 1,
                "chunk_id": "leave_policy_p1_001",
                "original_filename": "leave_policy.pdf",
            },
        )
    ]

    # Document B: Resume of Mayur Mohan Kamane
    doc_b_chunks = [
        Document(
            page_content="PERSONAL PROFILE MAYUR MOHAN KAMANE Full Stack AI Developer and Machine Learning Engineer.",
            metadata={
                "source": "Mayur_Mohan_Kamane_Resume.pdf",
                "page": 1,
                "chunk_id": "Mayur_Mohan_Kamane_Resume_p1_001",
                "original_filename": "Mayur_Mohan_Kamane_Resume.pdf",
            },
        ),
        Document(
            page_content="Mayur Mohan Kamane Education: Bachelor of Engineering in Computer Science.",
            metadata={
                "source": "Mayur_Mohan_Kamane_Resume.pdf",
                "page": 2,
                "chunk_id": "Mayur_Mohan_Kamane_Resume_p2_001",
                "original_filename": "Mayur_Mohan_Kamane_Resume.pdf",
            },
        ),
    ]

    # Index doc A
    res_a = service.add_documents(doc_a_chunks, original_filename="leave_policy.pdf")
    assert res_a["status"] == "indexed"

    # Index doc B
    res_b = service.add_documents(doc_b_chunks, original_filename="Mayur_Mohan_Kamane_Resume.pdf")
    assert res_b["status"] == "indexed"
    assert res_b["total_chunks"] == 3

    # Query 1: Search for resume name
    results_resume = service.similarity_search("Mayur Mohan Kamane", top_k=2)
    assert len(results_resume) >= 1
    top_doc, top_score = results_resume[0]
    assert top_doc.metadata["original_filename"] == "Mayur_Mohan_Kamane_Resume.pdf"
    assert "Mayur Mohan Kamane" in top_doc.page_content

    # Query 2: Search for leave policy
    results_leave = service.similarity_search("annual paid leave days", top_k=2)
    assert len(results_leave) >= 1
    top_leave_doc, top_leave_score = results_leave[0]
    assert top_leave_doc.metadata["original_filename"] == "leave_policy.pdf"
    assert "20 days of paid annual leave" in top_leave_doc.page_content


def test_cross_instance_disk_synchronization(tmp_path, shared_embedding_service):
    """Verifies that when Instance B writes new documents to disk, Instance A detects disk mtime update and reloads."""
    db_path = tmp_path / "cross_instance_store"

    # Instance A boots first and indexes Document A
    instance_a = RetrievalService(vector_db_path=db_path, embedding_service=shared_embedding_service)
    doc_a = [
        Document(
            page_content="Leave Policy 2026: Employees receive 20 days paid leave.",
            metadata={"source": "leave.pdf", "page": 1, "chunk_id": "c1", "original_filename": "leave.pdf"},
        )
    ]
    instance_a.add_documents(doc_a, original_filename="leave.pdf")

    # Instance B boots and shares the same disk directory
    instance_b = RetrievalService(vector_db_path=db_path, embedding_service=shared_embedding_service)
    assert instance_b.vector_store is not None

    # Instance B adds Document B (Resume) to the index and persists it to disk
    doc_b = [
        Document(
            page_content="Candidate Name: Mayur Mohan Kamane, AI Software Engineer.",
            metadata={"source": "resume.pdf", "page": 1, "chunk_id": "c2", "original_filename": "resume.pdf"},
        )
    ]
    instance_b.add_documents(doc_b, original_filename="resume.pdf")

    # Instance A was already in memory with only Document A.
    # When Instance A searches for 'Mayur Mohan Kamane', it MUST auto-reload from disk and return Document B!
    results = instance_a.similarity_search("Mayur Mohan Kamane", top_k=1)
    assert len(results) == 1
    top_doc, score = results[0]
    assert top_doc.metadata["original_filename"] == "resume.pdf"
    assert "Mayur Mohan Kamane" in top_doc.page_content


def test_rag_service_retrieve_multiple_documents(tmp_path, shared_embedding_service):
    """Verifies that RAGService properly formats retrieval results containing chunks from multiple indexed documents."""
    db_path = tmp_path / "rag_service_multi_doc"
    retrieval_svc = RetrievalService(vector_db_path=db_path, embedding_service=shared_embedding_service)
    rag_svc = RAGService(retrieval_service=retrieval_svc, embedding_service=shared_embedding_service)

    chunks = [
        Document(
            page_content="Policy: Standard office hours are 9 AM to 5 PM.",
            metadata={"source": "handbook.pdf", "page": 1, "chunk_id": "h_1", "original_filename": "handbook.pdf"},
        ),
        Document(
            page_content="Resume: Mayur Mohan Kamane has expertise in LangChain and FAISS.",
            metadata={"source": "resume.pdf", "page": 1, "chunk_id": "r_1", "original_filename": "resume.pdf"},
        ),
    ]

    retrieval_svc.add_documents(chunks[:1], original_filename="handbook.pdf")
    retrieval_svc.add_documents(chunks[1:], original_filename="resume.pdf")

    retrieved = rag_svc.retrieve("Mayur Mohan Kamane", top_k=2)
    assert len(retrieved) >= 1
    assert retrieved[0]["original_filename"] == "resume.pdf"
    assert "Mayur Mohan Kamane" in retrieved[0]["content"]
