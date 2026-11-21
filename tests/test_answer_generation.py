from unittest.mock import patch

import pytest

from app.services.rag_service import RAGService


def test_answer_uses_retrieved_context_and_returns_sources():
    service = RAGService()

    retrieved_results = {"ids": [["chunk_1", "chunk_2"]]}

    processed_context = {
        "ids": [["chunk_1", "chunk_2"]],
        "documents": [
            [
                "Python is a programming language used for backend development.",
                "FastAPI is a Python framework for building APIs.",
            ]
        ],
        "metadatas": [
            [
                {"filename": "python.pdf", "document_id": "doc_1"},
                {"filename": "fastapi.pdf", "document_id": "doc_2"},
            ]
        ],
        "distances": [[0.15, 0.20]],
    }

    with (
        patch.object(
            service, "retrieve_documents", return_value=retrieved_results
        ) as retrieve,
        patch(
            "app.services.rag_service.context_service.process",
            return_value=processed_context,
        ),
        patch(
            "app.services.rag_service.llm_service.generate_response",
            return_value="Python is used for backend development. FastAPI helps build APIs.",
        ) as generate,
    ):
        result = service.answer_question(question="What technologies are mentioned?")

    retrieve.assert_called_once_with(
        "What technologies are mentioned?", 3, return_candidates=True
    )

    prompt = generate.call_args.args[0]

    assert "Python is a programming language" in prompt
    assert "FastAPI is a Python framework" in prompt
    assert "python.pdf" in prompt
    assert "fastapi.pdf" in prompt

    assert result["answer"] == (
        "Python is used for backend development. FastAPI helps build APIs."
    )

    assert len(result["sources"]) == 2
    assert result["sources"][0]["filename"] == "python.pdf"
    assert result["sources"][1]["filename"] == "fastapi.pdf"


def test_answer_returns_fallback_when_no_context_is_found():
    service = RAGService()

    empty_context = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "hybrid_scores": [[]],
        "rerank_scores": [[]],
    }

    with (
        patch.object(service, "retrieve_documents", return_value={}),
        patch(
            "app.services.rag_service.context_service.process",
            return_value=empty_context,
        ),
        patch("app.services.rag_service.llm_service.generate_response") as generate,
    ):
        result = service.answer_question(question="What is quantum computing?")

    assert result["answer"] == (
        "I couldn't find relevant information in the knowledge base."
    )
    assert result["sources"] == []

    # The model must not generate an answer without retrieved context.
    generate.assert_not_called()


def test_answer_propagates_model_errors():
    service = RAGService()

    processed_context = {
        "ids": [["chunk_1"]],
        "documents": [["Python is a programming language."]],
        "metadatas": [[{"filename": "python.pdf", "document_id": "doc_1"}]],
        "distances": [[0.15]],
    }

    with (
        patch.object(service, "retrieve_documents", return_value={}),
        patch(
            "app.services.rag_service.context_service.process",
            return_value=processed_context,
        ),
        patch(
            "app.services.rag_service.llm_service.generate_response",
            side_effect=RuntimeError("AI service error"),
        ),
    ):
        with pytest.raises(RuntimeError, match="AI service error"):
            service.answer_question(question="What is Python?")


def test_follow_up_question_is_contextualized():
    service = RAGService()

    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]

    with patch(
        "app.services.rag_service.llm_service.generate_response",
        return_value="What are the advantages of Python?",
    ) as generate:

        result = service.contextualize_question(
            question="What are its advantages?", history=history
        )

    assert result == "What are the advantages of Python?"

    prompt = generate.call_args.args[0]

    assert "What are its advantages?" in prompt
    assert "Python is a programming language." in prompt


def test_standalone_question_does_not_need_contextualization():
    service = RAGService()

    question = "What is the capital of Sri Lanka?"

    result = service.contextualize_question(question=question, history=[])

    assert result == question

def test_conversation_history_is_not_treated_as_document_evidence():
    service = RAGService()

    history = [
        {
            "role": "user",
            "content": "Tell me about Python."
        },
        {
            "role": "assistant",
            "content": "Python was created in 1991."
        }
    ]

    processed_context = {
        "ids": [["chunk_1"]],
        "documents": [[
            "Python is a programming language used for software development."
        ]],
        "metadatas": [[
            {
                "filename": "python.pdf",
                "document_id": "doc_1"
            }
        ]],
        "distances": [[0.15]]
    }

    with (
        patch.object(
            service,
            "contextualize_question",
            return_value="When was Python created?"
        ),
        patch.object(
            service,
            "retrieve_documents",
            return_value={}
        ),
        patch(
            "app.services.rag_service.context_service.process",
            return_value=processed_context
        ),
        patch(
            "app.services.rag_service.llm_service.generate_response",
            return_value="The document does not specify this."
        ) as generate
    ):
        result = service.answer_question(
            question="When was it created?",
            history=history
        )

    prompt = generate.call_args.args[0]

    assert "Python was created in 1991." in prompt
    assert "DOCUMENT CONTEXT:" in prompt
    assert "Python is a programming language" in prompt
    assert "Never treat previous assistant responses as verified factual sources." in prompt

    assert result["answer"] == "The document does not specify this."