from unittest.mock import patch

from app.services.hybrid_search_service import HybridSearchService
import pytest

def test_hybrid_search_rejects_invalid_queries():
    service = HybridSearchService()

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        service.search(query="")

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        service.search(query="   ")

    with pytest.raises(ValueError, match="top_k must be between 1 and 10"):
        service.search(query="driving licence", top_k=0)

    with pytest.raises(ValueError, match="top_k must be between 1 and 10"):
        service.search(query="driving licence", top_k=11)
        
def test_hybrid_search_ranks_relevant_document_first():
    service = HybridSearchService()

    semantic_results = {
        "ids": [["relevant-1", "irrelevant-1"]],
        "documents": [
            [
                "A driving licence applicant must pass a written examination.",
                "The document discusses unrelated cooking recipes.",
            ]
        ],
        "metadatas": [
            [
                {"filename": "driving.pdf"},
                {"filename": "cooking.pdf"},
            ]
        ],
        "distances": [[0.15, 0.75]],
    }

    keyword_results = {
        "ids": [["relevant-1"]],
        "documents": [
            [
                "A driving licence applicant must pass a written examination.",
            ]
        ],
        "metadatas": [[{"filename": "driving.pdf"}]],
        "scores": [[4.5]],
    }

    with (
        patch(
            "app.services.hybrid_search_service.embedding_service.generate_embedding",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "app.services.hybrid_search_service.chroma_service.search",
            return_value=semantic_results,
        ),
        patch(
            "app.services.hybrid_search_service.keyword_search_service.search",
            return_value=keyword_results,
        ),
    ):
        results = service.search(
            query="What examination must a driving licence applicant pass?",
            top_k=2,
            min_similarity=0.10,
        )

    assert results["ids"][0][0] == "relevant-1"
    assert results["metadatas"][0][0]["filename"] == "driving.pdf"
    assert results["hybrid_scores"][0][0] > results["hybrid_scores"][0][1]


def test_hybrid_search_returns_empty_results_when_no_documents_match():
    service = HybridSearchService()

    empty_results = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    empty_keyword_results = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "scores": [[]],
    }

    with (
        patch(
            "app.services.hybrid_search_service.embedding_service.generate_embedding",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "app.services.hybrid_search_service.chroma_service.search",
            return_value=empty_results,
        ),
        patch(
            "app.services.hybrid_search_service.keyword_search_service.search",
            return_value=empty_keyword_results,
        ),
    ):
        results = service.search(
            query="quantum physics",
            top_k=3,
        )

    assert results["ids"] == [[]]
    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]
    assert results["distances"] == [[]]
    assert results["hybrid_scores"] == [[]]


def test_hybrid_search_returns_semantic_results_without_keyword_matches():
    service = HybridSearchService()

    semantic_results = {
        "ids": [["semantic-1"]],
        "documents": [
            [
                "A person must complete the required examination before receiving permission to drive."
            ]
        ],
        "metadatas": [[{"filename": "driving.pdf"}]],
        "distances": [[0.20]],
    }

    empty_keyword_results = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "scores": [[]],
    }

    with (
        patch(
            "app.services.hybrid_search_service.embedding_service.generate_embedding",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "app.services.hybrid_search_service.chroma_service.search",
            return_value=semantic_results,
        ),
        patch(
            "app.services.hybrid_search_service.keyword_search_service.search",
            return_value=empty_keyword_results,
        ),
    ):
        results = service.search(
            query="What must someone do before being allowed to drive?",
            top_k=3,
        )

    assert results["ids"][0] == ["semantic-1"]
    assert results["metadatas"][0][0]["filename"] == "driving.pdf"
    assert results["distances"][0] == [0.20]
    assert results["hybrid_scores"][0][0] > 0


def test_hybrid_search_returns_keyword_results_without_semantic_matches():
    service = HybridSearchService()

    empty_semantic_results = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    keyword_results = {
        "ids": [["keyword-1"]],
        "documents": [
            ["Department of Motor Traffic handles driving licence applications."]
        ],
        "metadatas": [[{"filename": "motor_traffic.pdf"}]],
        "scores": [[3.5]],
    }

    with (
        patch(
            "app.services.hybrid_search_service.embedding_service.generate_embedding",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "app.services.hybrid_search_service.chroma_service.search",
            return_value=empty_semantic_results,
        ),
        patch(
            "app.services.hybrid_search_service.keyword_search_service.search",
            return_value=keyword_results,
        ),
    ):
        results = service.search(
            query="Department of Motor Traffic",
            top_k=3,
        )

    assert results["ids"][0] == ["keyword-1"]
    assert results["metadatas"][0][0]["filename"] == "motor_traffic.pdf"
    assert results["hybrid_scores"][0][0] > 0


def test_hybrid_search_results_are_sorted_by_score():
    service = HybridSearchService()

    semantic_results = {
        "ids": [["doc-1", "doc-2", "doc-3"]],
        "documents": [
            [
                "First document.",
                "Second document.",
                "Third document.",
            ]
        ],
        "metadatas": [[{}, {}, {}]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    keyword_results = {
        "ids": [["doc-3", "doc-2"]],
        "documents": [
            [
                "Third document.",
                "Second document.",
            ]
        ],
        "metadatas": [[{}, {}]],
        "scores": [[3.0, 2.0]],
    }

    with (
        patch(
            "app.services.hybrid_search_service.embedding_service.generate_embedding",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "app.services.hybrid_search_service.chroma_service.search",
            return_value=semantic_results,
        ),
        patch(
            "app.services.hybrid_search_service.keyword_search_service.search",
            return_value=keyword_results,
        ),
    ):
        results = service.search(
            query="document ranking",
            top_k=3,
        )

    scores = results["hybrid_scores"][0]

    assert scores == sorted(scores, reverse=True)
    assert len(results["ids"][0]) == 3
    assert len(set(results["ids"][0])) == 3
