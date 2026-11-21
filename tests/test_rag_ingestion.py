import chromadb
import pytest
import uuid

from unittest.mock import patch

from pypdf.errors import PdfReadError


from app.services.rag_service import RAGService
from app.vectorstore.chroma_service import ChromaService


@pytest.fixture
def isolated_chroma():
    """Create an isolated in-memory ChromaDB collection."""

    client = chromadb.EphemeralClient()

    collection = client.create_collection(
        name=f"klyra_ingestion_test_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )

    service = ChromaService.__new__(ChromaService)
    service.client = client
    service.collection = collection

    return service


def test_document_ingestion_pipeline(isolated_chroma, tmp_path):

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"fake PDF content")

    sample_text = (
        "Artificial intelligence helps computers learn from data. "
        "Machine learning is used in many applications. "
    ) * 100

    rag = RAGService()

    with (
        patch(
            "app.services.rag_service.chroma_service",
            isolated_chroma,
        ),
        patch(
            "app.services.rag_service.document_service.extract_text",
            return_value=sample_text,
        ),
        patch(
            "app.services.rag_service.embedding_service.generate_embeddings",
            side_effect=lambda chunks: [[0.1, 0.2, 0.3] for _ in chunks],
        ),
        patch(
            "app.services.rag_service.keyword_search_service.invalidate_cache"
        ) as invalidate_cache,
    ):

        result = rag.ingest_document(
            file_path=str(pdf_path),
            filename="sample.pdf",
        )

    assert result["status"] == "processed"
    assert result["filename"] == "sample.pdf"
    assert result["total_chunks"] > 0
    assert result["document_id"]

    stored = isolated_chroma.collection.get(
        where={"document_id": result["document_id"]},
        include=["documents", "metadatas"],
    )

    assert len(stored["ids"]) == result["total_chunks"]
    assert all(stored["documents"])
    assert all(metadata["filename"] == "sample.pdf" for metadata in stored["metadatas"])

    invalidate_cache.assert_called_once()


def test_duplicate_document_is_not_ingested(isolated_chroma, tmp_path):

    pdf_path = tmp_path / "duplicate.pdf"
    pdf_path.write_bytes(b"duplicate PDF content")

    rag = RAGService()
    file_hash = rag.calculate_file_hash(str(pdf_path))

    isolated_chroma.add_documents(
        chunks=["Previously stored document content."],
        embeddings=[[0.1, 0.2, 0.3]],
        document_id="existing-document",
        filename="duplicate.pdf",
        file_hash=file_hash,
    )

    with (
        patch(
            "app.services.rag_service.chroma_service",
            isolated_chroma,
        ),
        patch("app.services.rag_service.document_service.extract_text") as extract_text,
        patch(
            "app.services.rag_service.embedding_service.generate_embeddings"
        ) as generate_embeddings,
    ):

        result = rag.ingest_document(
            file_path=str(pdf_path),
            filename="duplicate.pdf",
        )

    assert result["status"] == "duplicate"
    assert result["document_id"] == "existing-document"
    assert result["total_chunks"] == 0

    extract_text.assert_not_called()
    generate_embeddings.assert_not_called()


def test_empty_document_is_rejected(isolated_chroma, tmp_path):

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"empty PDF content")

    rag = RAGService()

    with (
        patch(
            "app.services.rag_service.chroma_service",
            isolated_chroma,
        ),
        patch(
            "app.services.rag_service.document_service.extract_text",
            return_value="   ",
        ),
    ):

        with pytest.raises(
            ValueError,
            match="no extractable text",
        ):
            rag.ingest_document(
                file_path=str(pdf_path),
                filename="empty.pdf",
            )

    assert isolated_chroma.collection.count() == 0


def test_invalid_pdf_is_rejected(isolated_chroma, tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(b"This is not a valid PDF file.")

    rag = RAGService()

    with patch(
        "app.services.rag_service.chroma_service",
        isolated_chroma,
    ):
        with pytest.raises(PdfReadError):
            rag.ingest_document(
                file_path=str(pdf_path),
                filename="invalid.pdf",
            )

    assert isolated_chroma.collection.count() == 0


def test_document_with_no_chunks_is_rejected(isolated_chroma, tmp_path):
    pdf_path = tmp_path / "no_chunks.pdf"
    pdf_path.write_bytes(b"placeholder PDF content")

    rag = RAGService()

    with (
        patch(
            "app.services.rag_service.chroma_service",
            isolated_chroma,
        ),
        patch(
            "app.services.rag_service.document_service.extract_text",
            return_value="Some extracted text.",
        ),
        patch(
            "app.services.rag_service.document_service.split_into_chunks",
            return_value=[],
        ),
    ):
        with pytest.raises(ValueError, match="No document chunks"):
            rag.ingest_document(
                file_path=str(pdf_path),
                filename="no_chunks.pdf",
            )

    assert isolated_chroma.collection.count() == 0


def test_embedding_failure_does_not_store_document(isolated_chroma, tmp_path):
    pdf_path = tmp_path / "embedding_failure.pdf"
    pdf_path.write_bytes(b"placeholder PDF content")

    rag = RAGService()

    sample_text = (
        "Artificial intelligence helps computers learn from data. "
        "Machine learning is used in many applications."
    )

    with (
        patch(
            "app.services.rag_service.chroma_service",
            isolated_chroma,
        ),
        patch(
            "app.services.rag_service.document_service.extract_text",
            return_value=sample_text,
        ),
        patch(
            "app.services.rag_service.embedding_service.generate_embeddings",
            side_effect=RuntimeError("Embedding model failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Embedding model failed"):
            rag.ingest_document(
                file_path=str(pdf_path),
                filename="embedding_failure.pdf",
            )

    assert isolated_chroma.collection.count() == 0
