
"""Compare standard hybrid retrieval with a semantic-gated hybrid."""

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
    ids, documents, metadatas = [], [], []

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

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "false_positives": len([
            chunk_id for chunk_id in chunk_ids
            if chunk_id not in direct and chunk_id not in supporting
        ]),
        "negative_correct": None,
    }


def run_experiment(method):
    positive, negative = [], []

    print(f"\n{'=' * 65}")
    print(f"METHOD: {method.upper()}")
    print("=" * 65)

    for index, case in enumerate(MULTI_DOCUMENT_EVALUATION_DATASET, start=1):
        question = case["question"]

        # Retrieve the full hybrid candidate list.
        hybrid = hybrid_search_service.search(
            query=question,
            top_k=TOP_K,
            min_similarity=MIN_SIMILARITY,
            return_candidates=True,
        )

        hybrid_ids = (hybrid.get("ids") or [[]])[0]

        if method == "hybrid":
            selected_ids = hybrid_ids[:TOP_K]

        elif method == "hybrid_semantic_gate":
            embedding = embedding_service.generate_embedding(question)

            semantic = chroma_service.search(
                query_embedding=embedding,
                top_k=20,
                min_similarity=MIN_SIMILARITY,
            )

            semantic_ids = set((semantic.get("ids") or [[]])[0])

            # Keep hybrid-ranked candidates only if semantic search found them.
            selected_ids = [
                chunk_id for chunk_id in hybrid_ids
                if chunk_id in semantic_ids
            ][:TOP_K]

        else:
            raise ValueError(f"Unknown method: {method}")

        result = evaluate_results(case, selected_ids)

        print(f"\nTest {index}/{len(MULTI_DOCUMENT_EVALUATION_DATASET)}")
        print(f"Question: {question}")
        print(f"Retrieved chunks: {selected_ids}")
        print(f"Precision: {result['precision']:.2%}")
        print(f"Direct recall: {result['recall']:.2%}")
        print(f"MRR: {result['mrr']:.2%}")
        print(f"False positives: {result['false_positives']}")

        if case["should_find_answer"]:
            positive.append(result)
        else:
            negative.append(result)
            print(f"Correctly rejected: {result['negative_correct']}")

    count = len(positive)
    negative_count = len(negative)

    summary = {
        "precision": sum(x["precision"] for x in positive) / count,
        "recall": sum(x["recall"] for x in positive) / count,
        "mrr": sum(x["mrr"] for x in positive) / count,
        "positive_fp": sum(x["false_positives"] for x in positive),
        "negative_correct": sum(
            x["negative_correct"] for x in negative
        ),
        "negative_total": negative_count,
        "negative_fp": sum(x["false_positives"] for x in negative),
    }

    print("\nSUMMARY")
    print(f"Mean precision: {summary['precision']:.2%}")
    print(f"Mean direct recall: {summary['recall']:.2%}")
    print(f"Mean MRR: {summary['mrr']:.2%}")
    print(f"Positive false positives: {summary['positive_fp']}")
    print(
        "Negative queries rejected: "
        f"{summary['negative_correct']}/{negative_count}"
    )
    print(f"Negative false positives: {summary['negative_fp']}")

    return summary


def evaluate():
    print("\nKLYRA HYBRID SEMANTIC-GATE EVALUATION")
    print("====================================")

    temporary_client = chromadb.EphemeralClient()
    temporary_collection = temporary_client.create_collection(
        name=f"klyra_gate_{uuid.uuid4().hex[:12]}",
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

        standard = run_experiment("hybrid")
        gated = run_experiment("hybrid_semantic_gate")

        print("\n\nFINAL COMPARISON")
        print("=" * 85)
        print(
            f"{'Method':<24}{'Precision':>12}{'Recall':>12}"
            f"{'MRR':>12}{'Positive FP':>14}{'Negative FP':>14}"
        )

        for name, result in (
            ("Standard hybrid", standard),
            ("Semantic-gated hybrid", gated),
        ):
            print(
                f"{name:<24}"
                f"{result['precision']:>11.2%}"
                f"{result['recall']:>11.2%}"
                f"{result['mrr']:>11.2%}"
                f"{result['positive_fp']:>14}"
                f"{result['negative_fp']:>14}"
            )

        print(
            "\nNegative queries rejected: "
            f"Standard {standard['negative_correct']}/"
            f"{standard['negative_total']}, "
            f"Gated {gated['negative_correct']}/"
            f"{gated['negative_total']}"
        )

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
