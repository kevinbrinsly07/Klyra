
import uuid

import chromadb
import numpy as np

from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.embedding_service import embedding_service
from app.services.hybrid_search_service import hybrid_search_service
from app.services.keyword_search_service import keyword_search_service
from app.services.context_service import context_service
from app.vectorstore.chroma_service import chroma_service


MIN_SIMILARITY = 0.50
TOP_K = 3


def build_synthetic_corpus():
    """Flatten the synthetic documents into searchable chunks."""

    chunks = []
    metadatas = []
    ids = []

    for document in EVALUATION_DOCUMENTS:
        for chunk in document["chunks"]:
            ids.append(chunk["chunk_id"])
            chunks.append(chunk["content"])

            metadatas.append({
                "document_id": document["document_id"],
                "filename": document["filename"],
            })

    return ids, chunks, metadatas


def evaluate():
    print("\nKLYRA MULTI-DOCUMENT RETRIEVAL EVALUATION")
    print("=========================================")

    original_collection = chroma_service.collection
    temporary_client = chromadb.EphemeralClient()

    collection_name = f"klyra_eval_{uuid.uuid4().hex[:12]}"

    temporary_collection = temporary_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        # 1. Prepare the synthetic corpus.
        ids, documents, metadatas = build_synthetic_corpus()

        print(f"\nSynthetic documents: {len(EVALUATION_DOCUMENTS)}")
        print(f"Synthetic chunks: {len(documents)}")

        # 2. Generate embeddings using the existing MiniLM model.
        print("\nGenerating embeddings with the existing model...")

        embeddings = embedding_service.generate_embeddings(documents)

        # 3. Insert only into the temporary in-memory collection.
        temporary_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )

        print("Synthetic corpus indexed successfully.")

        # 4. Temporarily route existing services to the test collection.
        chroma_service.collection = temporary_collection
        keyword_search_service.invalidate_cache()

        passed = 0
        total = len(MULTI_DOCUMENT_EVALUATION_DATASET)

        for index, test_case in enumerate(
            MULTI_DOCUMENT_EVALUATION_DATASET,
            start=1,
        ):
            question = test_case["question"]

            expected_document_ids = set(
                test_case["expected_document_ids"]
            )

            expected_terms = [
                term.lower()
                for term in test_case["expected_terms"]
            ]

            should_find = test_case["should_find_answer"]

            # 5. Run the actual hybrid retrieval pipeline.
            hybrid_results = hybrid_search_service.search(
                query=question,
                top_k=TOP_K,
                min_similarity=MIN_SIMILARITY,
                return_candidates=True,
            )

            # 6. Apply the actual context reranking and deduplication.
            processed = context_service.process(
                query=question,
                results=hybrid_results,
                max_chunks=TOP_K,
            )

            retrieved_documents = (
                processed.get("documents") or [[]]
            )[0]

            retrieved_metadatas = (
                processed.get("metadatas") or [[]]
            )[0]

            retrieved_document_ids = {
                metadata.get("document_id")
                for metadata in retrieved_metadatas
                if metadata
            }

            combined_text = " ".join(
                retrieved_documents
            ).lower()

            # 7. Evaluate the result.
            if should_find:
                documents_found = expected_document_ids.issubset(
                    retrieved_document_ids
                )

                terms_found = all(
                    term in combined_text
                    for term in expected_terms
                )

                success = documents_found and terms_found

            else:
                # Unsupported questions should return no context.
                success = len(retrieved_documents) == 0

            if success:
                passed += 1

            print(f"\nTest {index}/{total}")
            print(f"Question: {question}")
            print(f"Retrieved chunks: {len(retrieved_documents)}")
            print(
                "Retrieved documents: "
                f"{sorted(retrieved_document_ids)}"
            )
            print(
                "Expected documents: "
                f"{sorted(expected_document_ids)}"
            )

            if expected_terms:
                found_terms = [
                    term
                    for term in expected_terms
                    if term in combined_text
                ]

                print(
                    f"Expected terms found: "
                    f"{len(found_terms)}/{len(expected_terms)}"
                )

            print(f"Result: {'PASS' if success else 'FAIL'}")

        # 8. Print the final evaluation summary.
        accuracy = passed / total * 100 if total else 0

        print("\nEVALUATION SUMMARY")
        print("------------------")
        print(f"Passed: {passed}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")

        if passed == total:
            print("All multi-document retrieval tests passed.")
        else:
            print("Some retrieval tests need investigation.")

    finally:
        # 9. Always restore the production collection reference.
        chroma_service.collection = original_collection

        # Clear any temporary BM25 index.
        keyword_search_service.invalidate_cache()

        print("\nProduction ChromaDB collection restored.")
        print("Temporary evaluation collection was isolated.")


if __name__ == "__main__":
    evaluate()
