
"""Evaluate cross-encoder score thresholds on the synthetic corpus."""

import uuid
from typing import cast

import chromadb
from chromadb.api.types import Embeddings
from sentence_transformers import CrossEncoder

from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.embedding_service import embedding_service
from app.services.keyword_search_service import keyword_search_service
from app.services.hybrid_search_service import hybrid_search_service
from app.vectorstore.chroma_service import chroma_service


TOP_K = 3
MIN_SIMILARITY = 0.50

THRESHOLDS = [0.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0]

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_synthetic_corpus():
    ids = []
    documents = []
    metadatas = []

    for source in EVALUATION_DOCUMENTS:
        for index, chunk in enumerate(source["chunks"]):
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            metadatas.append({
                "document_id": source["document_id"],
                "filename": source["filename"],
                "chunk_index": index,
            })

    return ids, documents, metadatas


def evaluate_results(case, chunk_ids):
    direct = set(case["relevant_chunk_ids"])
    supporting = set(case["supporting_chunk_ids"])

    if not case["should_find_answer"]:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "false_positives": len(chunk_ids),
            "negative_correct": len(chunk_ids) == 0,
        }

    useful = [
        chunk_id for chunk_id in chunk_ids
        if chunk_id in direct or chunk_id in supporting
    ]

    precision = len(useful) / len(chunk_ids) if chunk_ids else 0.0

    direct_found = [
        chunk_id for chunk_id in chunk_ids if chunk_id in direct
    ]

    recall = len(set(direct_found)) / len(direct) if direct else 0.0

    mrr = 0.0
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id in direct:
            mrr = 1 / rank
            break

    false_positives = sum(
        1 for chunk_id in chunk_ids
        if chunk_id not in direct and chunk_id not in supporting
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "false_positives": false_positives,
        "negative_correct": None,
    }


def run_threshold(threshold, cross_encoder):
    positive_results = []
    negative_results = []

    print(f"\n{'=' * 65}")
    print(f"CROSS-ENCODER THRESHOLD: {threshold}")
    print("=" * 65)

    for index, case in enumerate(
        MULTI_DOCUMENT_EVALUATION_DATASET,
        start=1,
    ):
        question = case["question"]

        candidates = hybrid_search_service.search(
            query=question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            return_candidates=True,
        )

        candidate_ids = (candidates.get("ids") or [[]])[0]
        candidate_documents = (candidates.get("documents") or [[]])[0]

        pairs = [
            (question, document)
            for document in candidate_documents
        ]

        if pairs:
            scores = cross_encoder.predict(pairs)

            scored_candidates = [
                {
                    "chunk_id": candidate_ids[i],
                    "score": float(scores[i]),
                }
                for i in range(len(pairs))
            ]

            scored_candidates.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            selected_ids = [
                item["chunk_id"]
                for item in scored_candidates
                if item["score"] >= threshold
            ][:TOP_K]

        else:
            scored_candidates = []
            selected_ids = []

        metrics = evaluate_results(case, selected_ids)

        print(f"\nTest {index}/"
              f"{len(MULTI_DOCUMENT_EVALUATION_DATASET)}")
        print(f"Question: {question}")

        print("All candidate scores:")
        for item in scored_candidates:
            status = (
                "SELECTED"
                if item["chunk_id"] in selected_ids
                else "FILTERED"
            )
            print(
                f"  {item['chunk_id']}: "
                f"{item['score']:.4f} [{status}]"
            )

        print(f"Selected chunks: {selected_ids}")

        if case["should_find_answer"]:
            positive_results.append(metrics)
        else:
            negative_results.append(metrics)
            print(
                f"Correctly rejected: "
                f"{metrics['negative_correct']}"
            )

    positive_count = len(positive_results)
    negative_count = len(negative_results)

    summary = {
        "precision": (
            sum(x["precision"] for x in positive_results)
            / positive_count
            if positive_count else 0.0
        ),
        "recall": (
            sum(x["recall"] for x in positive_results)
            / positive_count
            if positive_count else 0.0
        ),
        "mrr": (
            sum(x["mrr"] for x in positive_results)
            / positive_count
            if positive_count else 0.0
        ),
        "positive_fp": sum(
            x["false_positives"] for x in positive_results
        ),
        "negative_fp": sum(
            x["false_positives"] for x in negative_results
        ),
        "negative_correct": sum(
            x["negative_correct"] for x in negative_results
        ),
        "negative_total": negative_count,
    }

    print("\nTHRESHOLD SUMMARY")
    print(f"Mean precision: {summary['precision']:.2%}")
    print(f"Mean direct recall: {summary['recall']:.2%}")
    print(f"Mean MRR: {summary['mrr']:.2%}")
    print(f"Positive false positives: {summary['positive_fp']}")
    print(f"Negative false positives: {summary['negative_fp']}")
    print(
        f"Negative queries rejected: "
        f"{summary['negative_correct']}/{negative_count}"
    )

    return summary


def evaluate():
    print("\nKLYRA CROSS-ENCODER THRESHOLD EVALUATION")
    print("========================================")

    print(f"\nLoading model: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_threshold_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    original_collection = chroma_service.collection

    try:
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"Synthetic documents: {len(EVALUATION_DOCUMENTS)}")
        print(f"Synthetic chunks: {len(documents)}")
        print("Generating embeddings...")

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

        for threshold in THRESHOLDS:
            results[threshold] = run_threshold(
                threshold,
                cross_encoder,
            )

        print("\n\nFINAL THRESHOLD COMPARISON")
        print("=" * 95)

        print(
            f"{'Threshold':<12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'MRR':>12}"
            f"{'Positive FP':>14}"
            f"{'Negative FP':>14}"
            f"{'Rejected':>12}"
        )

        for threshold, metrics in results.items():
            print(
                f"{threshold:<12.1f}"
                f"{metrics['precision']:>11.2%}"
                f"{metrics['recall']:>11.2%}"
                f"{metrics['mrr']:>11.2%}"
                f"{metrics['positive_fp']:>14}"
                f"{metrics['negative_fp']:>14}"
                f"{metrics['negative_correct']}/"
                f"{metrics['negative_total']:>9}"
            )

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
