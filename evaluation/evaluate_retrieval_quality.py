"""Evaluate Klyra hybrid retrieval quality using an isolated corpus."""

import uuid
from typing import cast

import chromadb
from chromadb.api.types import Embeddings

from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.embedding_service import embedding_service
from app.services.keyword_search_service import keyword_search_service
from app.services.hybrid_search_service import hybrid_search_service
from app.vectorstore.chroma_service import chroma_service

TOP_K = 3


def build_synthetic_corpus():
    ids = []
    documents = []
    metadatas = []

    for source in EVALUATION_DOCUMENTS:
        for index, chunk in enumerate(source["chunks"]):
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            metadatas.append(
                {
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "chunk_index": index,
                }
            )

    return ids, documents, metadatas


def evaluate_case(case, index):
    question = case["question"]
    expected_ids = set(case["expected_document_ids"])
    should_find = case["should_find_answer"]

    results = hybrid_search_service.search(
        query=question,
        top_k=TOP_K,
        min_similarity=0.50,
    )

    print("\nRetrieved chunk diagnostics:")

    result_ids = (results.get("ids") or [[]])[0]
    result_metadatas = (results.get("metadatas") or [[]])[0]
    result_distances = (results.get("distances") or [[]])[0]
    result_scores = (results.get("hybrid_scores") or [[]])[0]

    for rank, chunk_id in enumerate(result_ids, start=1):
        metadata = result_metadatas[rank - 1] or {}

        distance = (
            result_distances[rank - 1]
            if rank - 1 < len(result_distances)
            else None
        )

        score = result_scores[rank - 1] if rank - 1 < len(result_scores) else None

        print(
            f"Rank {rank}: "
            f"Chunk={chunk_id}, "
            f"Source={metadata.get('filename')}, "
            f"Semantic distance={distance}, "
            f"RRF score={score}"
        )

    retrieved_ids = (results.get("ids") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    retrieved_documents = [
        metadata.get("document_id")
        for metadata in metadatas
        if metadata and metadata.get("document_id")
    ]
    print("\nDEBUG: Exact metric input")
    print("Chunk IDs:", retrieved_ids)
    print("Metadata:", metadatas)
    print("Document IDs:", retrieved_documents)

    # Preserve ranking while counting each source document only once.
    unique_retrieved = list(dict.fromkeys(retrieved_documents))

    print(f"\nTest {index}/7")
    print(f"Question: {question}")
    print(f"Expected documents: {sorted(expected_ids)}")
    print(f"Retrieved chunks: {len(retrieved_ids)}")
    print(f"Retrieved documents: {unique_retrieved}")

    if should_find:
        relevant_retrieved = [
            document_id
            for document_id in unique_retrieved
            if document_id in expected_ids
        ]

        precision = (
            len(relevant_retrieved) / len(unique_retrieved) if unique_retrieved else 0.0
        )

        recall = len(set(relevant_retrieved)) / len(expected_ids)

        reciprocal_rank = 0.0

        for rank, document_id in enumerate(unique_retrieved, start=1):
            if document_id in expected_ids:
                reciprocal_rank = 1 / rank
                break

        print(f"Precision@{TOP_K}: {precision:.2f}")
        print(f"Recall@{TOP_K}: {recall:.2f}")
        print(f"Reciprocal rank: {reciprocal_rank:.2f}")

        return {
            "precision": precision,
            "recall": recall,
            "mrr": reciprocal_rank,
            "false_positives": len(set(unique_retrieved) - expected_ids),
        }

    correctly_rejected = len(unique_retrieved) == 0

    print(f"Correctly rejected: {correctly_rejected}")
    print(f"False positives: {len(unique_retrieved)}")

    return {
        "negative_correct": correctly_rejected,
        "false_positives": len(unique_retrieved),
    }


def evaluate():
    print("\nKLYRA RETRIEVAL QUALITY EVALUATION")
    print("===================================")

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_retrieval_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    original_collection = chroma_service.collection

    try:
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"\nSynthetic documents: {len(EVALUATION_DOCUMENTS)}")
        print(f"Synthetic chunks: {len(documents)}")

        print("\nGenerating embeddings...")
        embeddings = cast(
            Embeddings,
            embedding_service.generate_embeddings(documents),
        )

        temporary_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        chroma_service.collection = temporary_collection
        keyword_search_service.invalidate_cache()

        print("Temporary corpus indexed successfully.")

        positive_results = []
        negative_results = []

        for index, case in enumerate(
            MULTI_DOCUMENT_EVALUATION_DATASET,
            start=1,
        ):
            result = evaluate_case(case, index)

            if case["should_find_answer"]:
                positive_results.append(result)
            else:
                negative_results.append(result)

        print("\nRETRIEVAL QUALITY SUMMARY")
        print("-------------------------")

        if positive_results:
            count = len(positive_results)

            avg_precision = sum(item["precision"] for item in positive_results) / count

            avg_recall = sum(item["recall"] for item in positive_results) / count

            mrr = sum(item["mrr"] for item in positive_results) / count

            false_positives = sum(item["false_positives"] for item in positive_results)

            print(f"Mean Precision@{TOP_K}: {avg_precision:.2%}")
            print(f"Mean Recall@{TOP_K}: {avg_recall:.2%}")
            print(f"Mean Reciprocal Rank: {mrr:.2%}")
            print(f"Positive-query false positives: {false_positives}")

        if negative_results:
            correct = sum(item["negative_correct"] for item in negative_results)

            false_positives = sum(item["false_positives"] for item in negative_results)

            print(
                f"Negative queries correctly rejected: "
                f"{correct}/{len(negative_results)}"
            )
            print(f"Negative-query false positives: {false_positives}")

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
