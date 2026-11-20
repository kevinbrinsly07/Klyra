
import pytest

from app.services.document_service import document_service


def generate_sample_text(sentence_count=300):
    return " ".join(
        f"Sentence number {i} contains information about "
        f"artificial intelligence, document processing, "
        f"and knowledge retrieval."
        for i in range(1, sentence_count + 1)
    )


def test_chunks_do_not_exceed_configured_size():
    text = generate_sample_text()

    chunks = document_service.split_into_chunks(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 100 for chunk in chunks)


def test_chunks_preserve_configured_overlap():
    text = generate_sample_text()

    chunks = document_service.split_into_chunks(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.split()[-20:] == current.split()[:20]


def test_default_chunk_size():
    text = generate_sample_text()

    chunks = document_service.split_into_chunks(text)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 300 for chunk in chunks)


def test_empty_document_returns_no_chunks():
    assert document_service.split_into_chunks("") == []
    assert document_service.split_into_chunks("   ") == []


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        document_service.split_into_chunks("Some text.", chunk_size=0)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        document_service.split_into_chunks(
            "Some text.",
            chunk_size=100,
            chunk_overlap=100,
        )
