
"""Diagnose relevance scores for hybrid retrieval candidates."""

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


def diagnose_case(case, index):
    question = case["question"]
    expected_ids = set(case["expected_document_ids"])
    should_find = case["should_find_answer"]

    results = hybrid_search_service.search(
        query=question,
        top_k=TOP_K,
        min_similarity=MIN_SIMILARITY,
        return_candidates=True,
    )

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    hybrid_scores = (results.get("hybrid_scores") or [[]])[0]

    query_tokens = {
        token
        for token in context_service.tokenize(question)
        if token not in context_service.STOP_WORDS
    }

    max_hybrid_score = max(hybrid_scores, default=0.0)

    print("\n" + "=" * 90)
    print(f"TEST {index}/{len(MULTI_DOCUMENT_EVALUATION_DATASET)}")
    print(f"Question: {question}")
    print(f"Expected documents: {sorted(expected_ids)}")
    print(f"Meaningful query tokens: {sorted(query_tokens)}")
    print(f"Candidates returned: {len(ids)}")
    print("-" * 90)

    false_positive_count = 0
    relevant_count = 0

    for position, chunk_id in enumerate(ids):
        document = documents[position]
        metadata = (
            metadatas[position]
            if position < len(metadatas) and metadatas[position]
            else {}
        )

        document_id = metadata.get("document_id", "unknown")
        filename = metadata.get("filename", "unknown")

        tokens = set(context_service.tokenize(document))

        coverage = context_service.calculate_coverage(
            query_tokens,
            tokens,
        )

        hybrid_score = (
            float(hybrid_scores[position])
            if position < len(hybrid_scores)
            else 0.0
        )

        normalized_rrf = (
            hybrid_score / max_hybrid_score
            if max_hybrid_score > 0
            else 0.0
        )

        rerank_score = (
            0.85 * normalized_rrf
            + 0.15 * coverage
        )

        distance = (
            distances[position]
            if position < len(distances)
            else None
        )

        is_relevant = document_id in expected_ids

        if is_relevant:
            relevant_count += 1
            relevance_label = "RELEVANT"
        else:
            false_positive_count += 1
            relevance_label = "FALSE POSITIVE"

        if not should_find:
            relevance_label = "FALSE POSITIVE (NEGATIVE QUERY)"

        print(f"\nRank: {position + 1}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Source: {filename}")
        print(f"Document ID: {document_id}")
        print(f"Expected relevance: {relevance_label}")
        print(f"Semantic distance: {distance}")
        print(f"RRF score: {hybrid_score:.6f}")
        print(f"Normalized RRF: {normalized_rrf:.4f}")
        print(f"Query coverage: {coverage:.4f}")
        print(f"Current rerank score: {rerank_score:.4f}")
        print(f"Matched query terms: {sorted(query_tokens.intersection(tokens))}")
        print(f"Chunk text: {document}")

    print("\nCASE SUMMARY")
    print(f"Relevant candidates: {relevant_count}")
    print(f"False-positive candidates: {false_positive_count}")

    if not ids:
        print("No candidates retrieved.")


def evaluate():
    print("\nKLYRA CONTEXT RELEVANCE DIAGNOSTICS")
    print("===================================")

    temporary_client = chromadb.EphemeralClient()

    temporary_collection = temporary_client.create_collection(
        name=f"klyra_diagnostics_{uuid.uuid4().hex[:12]}",
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

        for index, case in enumerate(
            MULTI_DOCUMENT_EVALUATION_DATASET,
            start=1,
        ):
            diagnose_case(case, index)

    finally:
        chroma_service.collection = original_collection
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
