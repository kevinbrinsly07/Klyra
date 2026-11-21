
from app.services.context_service import context_service


def test_context_deduplication_and_reranking():

    results = {
        "ids": [[
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4"
        ]],
        "documents": [[
            "Sri Lanka driving licences require a learner permit, valid identification and a written examination.",
            "Sri Lanka driving licences require a learner permit, valid identification and a written examination.",
            "The Department of Motor Traffic issues driving licences in Sri Lanka.",
            "Quantum physics studies matter and energy at very small scales."
        ]],
        "metadatas": [[
            {"filename": "driving.pdf"},
            {"filename": "driving.pdf"},
            {"filename": "driving.pdf"},
            {"filename": "physics.pdf"}
        ]],
        "distances": [[0.2, 0.3, 0.4, 0.8]],
        "hybrid_scores": [[0.03, 0.02, 0.025, 0.01]]
    }

    processed = context_service.process(
        query="What are the requirements for a driving licence in Sri Lanka?",
        results=results,
        max_chunks=3
    )

    documents = processed["documents"][0]

    # Exact duplicate should be removed.
    assert len(documents) == 3
    assert len(set(documents)) == 3

    # The highest-ranked original candidate should remain first.
    assert processed["ids"][0][0] == "chunk_1"

    # Metadata and distances must remain aligned.
    assert len(processed["metadatas"][0]) == len(documents)
    assert len(processed["distances"][0]) == len(documents)
    assert len(processed["rerank_scores"][0]) == len(documents)


def test_empty_results():

    processed = context_service.process(
        query="driving licence",
        results={
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "hybrid_scores": [[]]
        }
    )

    assert processed["documents"] == [[]]
    assert processed["ids"] == [[]]


if __name__ == "__main__":
    test_context_deduplication_and_reranking()
    test_empty_results()

    print("Context processing tests passed!")

def test_context_respects_max_chunks():
    results = {
        "ids": [["chunk_1", "chunk_2", "chunk_3", "chunk_4"]],
        "documents": [[
            "Python is used for backend development.",
            "React is used for frontend development.",
            "PostgreSQL is a relational database.",
            "Docker is used for containerization."
        ]],
        "metadatas": [[{}, {}, {}, {}]],
        "distances": [[0.1, 0.2, 0.3, 0.4]],
        "hybrid_scores": [[0.04, 0.03, 0.02, 0.01]]
    }

    processed = context_service.process(
        query="Python backend development",
        results=results,
        max_chunks=2
    )

    assert len(processed["documents"][0]) == 2


def test_context_removes_empty_chunks():
    results = {
        "ids": [["chunk_1", "chunk_2", "chunk_3"]],
        "documents": [[
            "Python is used for backend development.",
            "",
            "React is used for frontend development."
        ]],
        "metadatas": [[{}, {}, {}]],
        "distances": [[0.1, 0.2, 0.3]],
        "hybrid_scores": [[0.04, 0.03, 0.02]]
    }

    processed = context_service.process(
        query="Python React development",
        results=results
    )

    assert len(processed["documents"][0]) == 2
    assert all(document.strip() for document in processed["documents"][0])


def test_context_handles_missing_optional_fields():
    results = {
        "documents": [[
            "Python is used for backend development."
        ]]
    }

    processed = context_service.process(
        query="Python backend development",
        results=results
    )

    assert processed["ids"] == [["0"]]
    assert processed["metadatas"] == [[{}]]
    assert processed["distances"] == [[None]]
    assert len(processed["rerank_scores"][0]) == 1


def test_context_rejects_invalid_queries_and_limits():
    import pytest

    with pytest.raises(ValueError, match="Query cannot be empty"):
        context_service.process("", {"documents": [[]]})

    with pytest.raises(ValueError, match="Query cannot be empty"):
        context_service.process("   ", {"documents": [[]]})

    with pytest.raises(ValueError, match="max_chunks must be between 1 and 10"):
        context_service.process("Python", {"documents": [[]]}, max_chunks=0)

    with pytest.raises(ValueError, match="max_chunks must be between 1 and 10"):
        context_service.process("Python", {"documents": [[]]}, max_chunks=11)