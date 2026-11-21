"""Compare semantic, BM25 and hybrid retrieval using an isolated corpus."""

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
MIN_SIMILARITY = 0.10


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


def run_search(method, question):
    if method == "semantic":
        query_embedding = embedding_service.generate_embedding(question)

        return chroma_service.search(
            query_embedding=query_embedding,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
        )

    if method == "bm25":
        return keyword_search_service.search(
            query=question,
            top_k=TOP_K,
        )

    if method == "hybrid":
        return hybrid_search_service.search(
            query=question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
        )

    raise ValueError(f"Unknown retrieval method: {method}")


def evaluate_method(method):
    positive_results = []
    negative_results = []

    print(f"\n{'=' * 65}")
    print(f"METHOD: {method.upper()}")
    print("=" * 65)

    for index, case in enumerate(
        MULTI_DOCUMENT_EVALUATION_DATASET,
        start=1,
    ):
        question = case["question"]
        expected_ids = set(case["expected_document_ids"])
        should_find = case["should_find_answer"]

        results = run_search(method, question)

        chunk_ids = (results.get("ids") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        retrieved_documents = [
            metadata.get("document_id")
            for metadata in metadatas
            if metadata and metadata.get("document_id")
        ]

        unique_retrieved = list(dict.fromkeys(retrieved_documents))

        print(f"\nTest {index}/7: {question}")
        print(f"Retrieved chunks: {chunk_ids}")
        print(f"Retrieved documents: {unique_retrieved}")

        if should_find:
            relevant = [
                document_id
                for document_id in unique_retrieved
                if document_id in expected_ids
            ]

            precision = (
                len(relevant) / len(unique_retrieved) if unique_retrieved else 0.0
            )

            recall = len(set(relevant)) / len(expected_ids) if expected_ids else 0.0

            reciprocal_rank = 0.0

            for rank, document_id in enumerate(
                unique_retrieved,
                start=1,
            ):
                if document_id in expected_ids:
                    reciprocal_rank = 1 / rank
                    break

            positive_results.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "mrr": reciprocal_rank,
                    "false_positives": len(set(unique_retrieved) - expected_ids),
                }
            )

        else:
            negative_results.append(
                {
                    "correct": len(unique_retrieved) == 0,
                    "false_positives": len(unique_retrieved),
                }
            )

    positive_count = len(positive_results)
    negative_count = len(negative_results)

    if positive_count:
        precision = sum(item["precision"] for item in positive_results) / positive_count

        recall = sum(item["recall"] for item in positive_results) / positive_count

        mrr = sum(item["mrr"] for item in positive_results) / positive_count

        positive_false_positives = sum(
          item["false_positives"] for item in positive_results
        )

        print(f"\nMean Precision@{TOP_K}: {precision:.2%}")
        print(f"Mean Recall@{TOP_K}: {recall:.2%}")
        print(f"Mean Reciprocal Rank: {mrr:.2%}")
        print(f"Positive-query false positives: {positive_false_positives}")

    if negative_count:
        correct = sum(item["correct"] for item in negative_results)

        false_positives = sum(item["false_positives"] for item in negative_results)

        print(f"Negative queries correctly rejected: " f"{correct}/{negative_count}")
        print(f"Negative-query false positives: {false_positives}")

    return {
        "precision": precision if positive_count else 0.0,
        "recall": recall if positive_count else 0.0,
        "mrr": mrr if positive_count else 0.0,
        "positive_false_positives": (
            positive_false_positives if positive_count else 0
        ),
        "negative_correct": correct if negative_count else 0,
        "negative_total": negative_count,
        "negative_false_positives": (false_positives if negative_count else 0),
    }


def evaluate():
    print("\nKLYRA RETRIEVAL ABLATION EVALUATION")
    print("===================================")

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_ablation_{uuid.uuid4().hex[:12]}",
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

        results = {}

        for method in ("semantic", "bm25", "hybrid"):
            results[method] = evaluate_method(method)

        print("\n\nFINAL ABLATION COMPARISON")
        print("=" * 80)

        print(
            f"{'Method':<12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'MRR':>12}"
            f"{'Positive FP':>14}"
            f"{'Negative FP':>14}"
        )

        for method, metrics in results.items():
            print(
                f"{method:<12}"
                f"{metrics['precision']:>11.2%}"
                f"{metrics['recall']:>11.2%}"
                f"{metrics['mrr']:>11.2%}"
                f"{metrics['positive_false_positives']:>14}"
                f"{metrics['negative_false_positives']:>14}"
            )

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
