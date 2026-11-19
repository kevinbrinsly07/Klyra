
"""Evaluate cross-encoder thresholds on the synthetic corpus.

Retrieval and cross-encoder scores are calculated once per question.
All thresholds use the same candidate rankings for a fair comparison.
The production ChromaDB collection is never modified.
"""

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


def retrieve_and_score_cases(cross_encoder):
    """Retrieve and score candidates once for every evaluation question."""

    scored_cases = []

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
        else:
            scored_candidates = []

        scored_cases.append({
            "case": case,
            "scored_candidates": scored_candidates,
        })

        print(
            f"Scored question {index}/"
            f"{len(MULTI_DOCUMENT_EVALUATION_DATASET)}"
        )

    return scored_cases


def evaluate_results(case, chunk_ids):
    direct = set(case["relevant_chunk_ids"])
    supporting = set(case["supporting_chunk_ids"])

    useful = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id in direct or chunk_id in supporting
    ]

    false_positives = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in direct and chunk_id not in supporting
    ]

    if not case["should_find_answer"]:
        return {
            "retrieved": len(chunk_ids),
            "useful": 0,
            "false_positives": len(chunk_ids),
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "negative_correct": len(chunk_ids) == 0,
        }

    direct_found = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id in direct
    ]

    precision = len(useful) / len(chunk_ids) if chunk_ids else 0.0

    recall = (
        len(set(direct_found)) / len(direct)
        if direct else 0.0
    )

    mrr = 0.0

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id in direct:
            mrr = 1 / rank
            break

    return {
        "retrieved": len(chunk_ids),
        "useful": len(useful),
        "false_positives": len(false_positives),
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "negative_correct": None,
    }


def run_threshold(threshold, scored_cases):
    positive_results = []
    negative_results = []

    print(f"\n{'=' * 65}")
    print(f"CROSS-ENCODER THRESHOLD: {threshold}")
    print("=" * 65)

    for index, entry in enumerate(scored_cases, start=1):
        case = entry["case"]
        scored_candidates = entry["scored_candidates"]

        selected_ids = [
            item["chunk_id"]
            for item in scored_candidates
            if item["score"] >= threshold
        ][:TOP_K]

        metrics = evaluate_results(case, selected_ids)

        print(
            f"\nTest {index}/"
            f"{len(scored_cases)}"
        )
        print(f"Question: {case['question']}")

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

    total_useful = sum(
        item["useful"] for item in positive_results
    )

    total_retrieved = sum(
        item["retrieved"] for item in positive_results
    )

    total_false_positives = (
        sum(item["false_positives"] for item in positive_results)
        + sum(item["false_positives"] for item in negative_results)
    )

    summary = {
        "macro_precision": (
            sum(item["precision"] for item in positive_results)
            / positive_count
            if positive_count else 0.0
        ),
        "micro_precision": (
            total_useful / total_retrieved
            if total_retrieved else 0.0
        ),
        "recall": (
            sum(item["recall"] for item in positive_results)
            / positive_count
            if positive_count else 0.0
        ),
        "mrr": (
            sum(item["mrr"] for item in positive_results)
            / positive_count
            if positive_count else 0.0
        ),
        "positive_fp": sum(
            item["false_positives"]
            for item in positive_results
        ),
        "negative_fp": sum(
            item["false_positives"]
            for item in negative_results
        ),
        "total_fp": total_false_positives,
        "negative_correct": sum(
            item["negative_correct"]
            for item in negative_results
        ),
        "negative_total": negative_count,
    }

    print("\nTHRESHOLD SUMMARY")
    print(
        f"Macro precision: "
        f"{summary['macro_precision']:.2%}"
    )
    print(
        f"Micro precision: "
        f"{summary['micro_precision']:.2%}"
    )
    print(f"Mean direct recall: {summary['recall']:.2%}")
    print(f"Mean MRR: {summary['mrr']:.2%}")
    print(f"Positive false positives: {summary['positive_fp']}")
    print(f"Negative false positives: {summary['negative_fp']}")
    print(f"Total false positives: {summary['total_fp']}")
    print(
        f"Negative queries rejected: "
        f"{summary['negative_correct']}/"
        f"{negative_count}"
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

        print("\nRetrieving and scoring candidates once...")
        scored_cases = retrieve_and_score_cases(cross_encoder)

        results = {}

        for threshold in THRESHOLDS:
            results[threshold] = run_threshold(
                threshold,
                scored_cases,
            )

        print("\n\nFINAL THRESHOLD COMPARISON")
        print("=" * 115)

        print(
            f"{'Threshold':<11}"
            f"{'Macro P':>11}"
            f"{'Micro P':>11}"
            f"{'Recall':>11}"
            f"{'MRR':>11}"
            f"{'Positive FP':>13}"
            f"{'Negative FP':>13}"
            f"{'Total FP':>11}"
            f"{'Rejected':>12}"
        )

        for threshold, metrics in results.items():
            print(
                f"{threshold:<11.1f}"
                f"{metrics['macro_precision']:>10.2%}"
                f"{metrics['micro_precision']:>11.2%}"
                f"{metrics['recall']:>11.2%}"
                f"{metrics['mrr']:>11.2%}"
                f"{metrics['positive_fp']:>13}"
                f"{metrics['negative_fp']:>13}"
                f"{metrics['total_fp']:>11}"
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
