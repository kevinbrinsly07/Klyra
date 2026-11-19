"""Evaluate query-coverage thresholds using chunk-level relevance labels."""

import uuid
from typing import cast

import chromadb
from chromadb.api.types import Embeddings

from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.context_service import context_service
from app.services.embedding_service import embedding_service
from app.services.hybrid_search_service import hybrid_search_service
from app.services.keyword_search_service import keyword_search_service
from app.vectorstore.chroma_service import chroma_service

TOP_K = 3
MIN_SIMILARITY = 0.50

THRESHOLDS = [0.0, 0.20, 0.30, 0.40, 0.50]


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


def calculate_query_coverage(question, document):
    query_tokens = {
        token
        for token in context_service.tokenize(question)
        if token not in context_service.STOP_WORDS
    }

    document_tokens = set(context_service.tokenize(document))

    return context_service.calculate_coverage(
        query_tokens,
        document_tokens,
    )


def evaluate_case(case, threshold):
    question = case["question"]

    direct_relevant = set(case["relevant_chunk_ids"])
    supporting = set(case["supporting_chunk_ids"])

    results = hybrid_search_service.search(
        query=question,
        top_k=TOP_K,
        min_similarity=MIN_SIMILARITY,
        return_candidates=True,
    )

    candidate_ids = (results.get("ids") or [[]])[0]
    candidate_documents = (results.get("documents") or [[]])[0]

    candidates = []

    for index, chunk_id in enumerate(candidate_ids):
        if index >= len(candidate_documents):
            continue

        document = candidate_documents[index]

        coverage = calculate_query_coverage(
            question,
            document,
        )

        if coverage >= threshold:
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "coverage": coverage,
                }
            )

    selected = candidates[:TOP_K]

    if not case["should_find_answer"]:
        return {
            "precision": 0.0,
            "direct_recall": 0.0,
            "mrr": 0.0,
            "supporting": 0,
            "false_positives": len(selected),
            "negative_correct": len(selected) == 0,
        }

    retrieved_ids = [candidate["chunk_id"] for candidate in selected]

    direct_retrieved = [
        chunk_id for chunk_id in retrieved_ids if chunk_id in direct_relevant
    ]

    supporting_retrieved = [
        chunk_id for chunk_id in retrieved_ids if chunk_id in supporting
    ]

    irrelevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id not in direct_relevant and chunk_id not in supporting
    ]

    # Direct and supporting chunks both count as useful context.
    useful_count = len(direct_retrieved) + len(supporting_retrieved)

    precision = useful_count / len(retrieved_ids) if retrieved_ids else 0.0

    direct_recall = (
        len(set(direct_retrieved)) / len(direct_relevant) if direct_relevant else 0.0
    )

    mrr = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in direct_relevant:
            mrr = 1 / rank
            break

    return {
        "precision": precision,
        "direct_recall": direct_recall,
        "mrr": mrr,
        "supporting": len(supporting_retrieved),
        "false_positives": len(irrelevant_retrieved),
        "negative_correct": None,
    }


def evaluate_threshold(threshold):
    positive_results = []
    negative_results = []

    print("\n" + "=" * 75)
    print(f"MINIMUM QUERY COVERAGE: {threshold:.0%}")
    print("=" * 75)

    for index, case in enumerate(
        MULTI_DOCUMENT_EVALUATION_DATASET,
        start=1,
    ):
        result = evaluate_case(case, threshold)

        if case["should_find_answer"]:
            positive_results.append(result)
        else:
            negative_results.append(result)

        print(
            f"\nTest {index}/{len(MULTI_DOCUMENT_EVALUATION_DATASET)}: {case['question']}"
        )
        print(f"Direct relevance: {case['relevant_chunk_ids']}")
        print(f"Supporting context: {case['supporting_chunk_ids']}")

        print(f"Useful precision: {result['precision']:.2%}")
        print(f"Direct recall: {result['direct_recall']:.2%}")
        print(f"Direct MRR: {result['mrr']:.2%}")
        print(f"Supporting chunks retrieved: {result['supporting']}")
        print(f"Irrelevant chunks retrieved: {result['false_positives']}")

        if not case["should_find_answer"]:
            print(
                "Correctly rejected:",
                result["negative_correct"],
            )

    positive_count = len(positive_results)
    negative_count = len(negative_results)

    avg_precision = (
        sum(item["precision"] for item in positive_results) / positive_count
        if positive_count
        else 0.0
    )

    avg_recall = (
        sum(item["direct_recall"] for item in positive_results) / positive_count
        if positive_count
        else 0.0
    )

    avg_mrr = (
        sum(item["mrr"] for item in positive_results) / positive_count
        if positive_count
        else 0.0
    )

    positive_false_positives = sum(item["false_positives"] for item in positive_results)

    negative_correct = sum(item["negative_correct"] for item in negative_results)

    negative_false_positives = sum(item["false_positives"] for item in negative_results)

    summary = {
        "threshold": threshold,
        "precision": avg_precision,
        "recall": avg_recall,
        "mrr": avg_mrr,
        "positive_fp": positive_false_positives,
        "negative_correct": negative_correct,
        "negative_total": negative_count,
        "negative_fp": negative_false_positives,
    }

    print("\nTHRESHOLD SUMMARY")
    print(f"Mean useful precision: {avg_precision:.2%}")
    print(f"Mean direct recall: {avg_recall:.2%}")
    print(f"Mean direct MRR: {avg_mrr:.2%}")
    print(f"Positive-query irrelevant chunks: {positive_false_positives}")
    print(
        f"Negative queries correctly rejected: " f"{negative_correct}/{negative_count}"
    )
    print(f"Negative-query false positives: {negative_false_positives}")

    return summary


def evaluate():
    print("\nKLYRA RELEVANCE THRESHOLD EVALUATION")
    print("====================================")

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_thresholds_{uuid.uuid4().hex[:12]}",
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

        summaries = []

        for threshold in THRESHOLDS:
            summaries.append(evaluate_threshold(threshold))

        print("\n\nFINAL THRESHOLD COMPARISON")
        print("=" * 100)

        print(
            f"{'Threshold':<12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'MRR':>12}"
            f"{'Positive FP':>14}"
            f"{'Negative FP':>14}"
            f"{'Negative Reject':>18}"
        )

        for item in summaries:
            print(
                f"{item['threshold']:<12.0%}"
                f"{item['precision']:>11.2%}"
                f"{item['recall']:>11.2%}"
                f"{item['mrr']:>11.2%}"
                f"{item['positive_fp']:>14}"
                f"{item['negative_fp']:>14}"
                f"{item['negative_correct']}/{item['negative_total']:>15}"
            )

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
