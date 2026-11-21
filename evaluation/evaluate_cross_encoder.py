"""Evaluate cross-encoder reranking against standard hybrid retrieval."""

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
CANDIDATE_K = 10
MIN_SIMILARITY = 0.10

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id in direct or chunk_id in supporting
    ]

    precision = len(useful) / len(chunk_ids) if chunk_ids else 0.0

    direct_found = [chunk_id for chunk_id in chunk_ids if chunk_id in direct]

    recall = len(set(direct_found)) / len(direct) if direct else 0.0

    mrr = 0.0
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id in direct:
            mrr = 1 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "false_positives": len(
            [
                chunk_id
                for chunk_id in chunk_ids
                if chunk_id not in direct and chunk_id not in supporting
            ]
        ),
        "negative_correct": None,
    }


def build_summary(results):
    positive = [r for r in results if r["should_find"]]
    negative = [r for r in results if not r["should_find"]]

    return {
        "precision": (
            sum(r["precision"] for r in positive) / len(positive) if positive else 0.0
        ),
        "recall": (
            sum(r["recall"] for r in positive) / len(positive) if positive else 0.0
        ),
        "mrr": (sum(r["mrr"] for r in positive) / len(positive) if positive else 0.0),
        "positive_fp": sum(r["false_positives"] for r in positive),
        "negative_fp": sum(r["false_positives"] for r in negative),
        "negative_correct": sum(r["negative_correct"] for r in negative),
        "negative_total": len(negative),
    }


def print_summary(name, summary):
    print(f"\n{name}")
    print("-" * 45)
    print(f"Mean precision: {summary['precision']:.2%}")
    print(f"Mean direct recall: {summary['recall']:.2%}")
    print(f"Mean MRR: {summary['mrr']:.2%}")
    print(f"Positive false positives: {summary['positive_fp']}")
    print(f"Negative false positives: {summary['negative_fp']}")
    print(
        "Negative queries rejected: "
        f"{summary['negative_correct']}/{summary['negative_total']}"
    )


def evaluate():
    print("\nKLYRA CROSS-ENCODER RERANKING EVALUATION")
    print("========================================")

    print(f"\nLoading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    temporary_client = chromadb.EphemeralClient()
    temporary_collection = temporary_client.create_collection(
        name=f"klyra_cross_encoder_{uuid.uuid4().hex[:12]}",
        metadata={"hnsw:space": "cosine"},
    )

    original_collection = chroma_service.collection

    try:
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"\nSynthetic documents: {len(EVALUATION_DOCUMENTS)}")
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

        baseline_results = []
        reranked_results = []

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

            # Baseline: use the original hybrid ranking.
            baseline_ids = candidate_ids[:TOP_K]

            baseline_metric = evaluate_results(case, baseline_ids)
            baseline_metric["should_find"] = case["should_find_answer"]
            baseline_results.append(baseline_metric)

            # Cross-encoder: score each question and candidate chunk together.
            pairs = [(question, document) for document in candidate_documents]

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

                reranked_ids = [item["chunk_id"] for item in scored_candidates[:TOP_K]]
            else:
                scored_candidates = []
                reranked_ids = []

            reranked_metric = evaluate_results(case, reranked_ids)
            reranked_metric["should_find"] = case["should_find_answer"]
            reranked_results.append(reranked_metric)

            print(f"\nTest {index}/" f"{len(MULTI_DOCUMENT_EVALUATION_DATASET)}")
            print(f"Question: {question}")

            print("All candidate scores:")
            for item in scored_candidates:
                status = (
                    "SELECTED"
                    if item["chunk_id"] in reranked_ids
                    else "FILTERED"
                )
                print(
                    f"  {item['chunk_id']}: "
                    f"{item['score']:.4f} [{status}]"
                )

            print(f"Selected chunks: {reranked_ids}")

            for item in scored_candidates:
                print(f"  {item['chunk_id']}: " f"{item['score']:.4f}")

            print(f"Cross-encoder top {TOP_K}: {reranked_ids}")
            print(f"Baseline precision: " f"{baseline_metric['precision']:.2%}")
            print(f"Reranked precision: " f"{reranked_metric['precision']:.2%}")

        baseline_summary = build_summary(baseline_results)
        reranked_summary = build_summary(reranked_results)

        print("\n\nFINAL COMPARISON")
        print("=" * 60)

        print_summary("STANDARD HYBRID", baseline_summary)
        print_summary("CROSS-ENCODER RERANKING", reranked_summary)

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
