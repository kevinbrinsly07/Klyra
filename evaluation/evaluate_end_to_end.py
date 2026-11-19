"""End-to-end evaluation of Klyra's RAG pipeline.

Uses a temporary ChromaDB collection and the existing RAG services.
The production collection is restored after evaluation.
"""

import uuid
from typing import Any, cast
import re

import chromadb

from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.embedding_service import embedding_service
from app.services.keyword_search_service import keyword_search_service
from app.services.rag_service import rag_service
from app.vectorstore.chroma_service import chroma_service


def build_synthetic_corpus():
    """Flatten the synthetic evaluation documents."""

    ids = []
    documents = []
    metadatas = []

    for source in EVALUATION_DOCUMENTS:
        for chunk in source["chunks"]:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            metadatas.append(
                {
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "chunk_index": len(
                        [
                            item
                            for item in metadatas
                            if item["document_id"] == source["document_id"]
                        ]
                    ),
                }
            )

    return ids, documents, metadatas


def normalize_text(text: str) -> str:
    """Normalize punctuation and whitespace for phrase comparisons."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def evaluate_case(case: dict[str, Any], index: int) -> bool:
    """Run and score one evaluation case."""

    question = case["question"]
    should_find = case["should_find_answer"]
    expected_terms = case["expected_terms"]
    expected_ids = set(case["expected_document_ids"])   

    response = rag_service.answer_question(
    question=question,
    top_k=3,
)

    answer = response["answer"]
    sources = response["sources"]

    normalized_answer = normalize_text(answer)
    source_ids = {
        source["document_id"]
        for source in sources
        if source.get("document_id")
    }

    print(f"\nTest {index}: {question}")

    if should_find:
        answer_found = all(
            normalize_text(term) in normalized_answer for term in expected_terms
        )

        sources_found = expected_ids.issubset(source_ids)

        passed = answer_found and sources_found

        print(
            f"Expected answer terms: "
            f"{sum(normalize_text(term) in normalized_answer for term in expected_terms)}"
            f"/{len(expected_terms)}"
        )
        print(f"Expected sources found: {sources_found}")

    else:
        abstention_markers = (
            "couldn't find relevant information",
            "does not state",
            "does not contain",
            "cannot find",
            "not provided",
            "not specified",
        )

        answer_not_found = any(marker in answer.lower() for marker in abstention_markers)

        source_leakage = bool(sources)

        passed = answer_not_found and not source_leakage

        print(f"Correctly abstained: {answer_not_found}")
        print(f"Sources attached despite abstention: {source_leakage}")

        if source_leakage:
            print("WARNING: Retrieved sources may not support the question.")

    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def evaluate():
    print("\nKLYRA END-TO-END RAG EVALUATION")
    print("================================")

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_e2e_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    original_collection = chroma_service.collection

    try:
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"\nSynthetic documents: {len(EVALUATION_DOCUMENTS)}")
        print(f"Synthetic chunks: {len(documents)}")

        print("\nGenerating embeddings...")
        embeddings = embedding_service.generate_embeddings(documents)

        temporary_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=cast(Any, embeddings),
        )

        chroma_service.collection = temporary_collection
        keyword_search_service.invalidate_cache()

        print("Temporary corpus indexed successfully.")

        results = []

        for index, case in enumerate(
            MULTI_DOCUMENT_EVALUATION_DATASET,
            start=1,
        ):
            try:
                passed = evaluate_case(case, index)
                results.append(passed)

            except Exception as error:
                print(f"\nTest {index} failed with an error: {error}")
                results.append(False)

        passed_count = sum(results)
        total_count = len(results)

        print("\nEVALUATION SUMMARY")
        print("------------------")
        print(f"Passed: {passed_count}/{total_count}")
        print(f"Accuracy: {passed_count / total_count * 100:.2f}%")

        if passed_count == total_count:
            print("All end-to-end RAG tests passed.")
        else:
            print("Some end-to-end RAG tests failed.")

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
