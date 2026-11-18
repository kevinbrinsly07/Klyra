
from evaluation.multi_document_dataset import (
    EVALUATION_DOCUMENTS,
    MULTI_DOCUMENT_EVALUATION_DATASET,
)
from app.services.context_service import context_service


def build_synthetic_results():
    """Build Chroma-like results from synthetic evaluation documents."""

    ids = []
    documents = []
    metadatas = []
    distances = []
    hybrid_scores = []

    for document in EVALUATION_DOCUMENTS:
        for chunk in document["chunks"]:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])

            metadatas.append({
                "document_id": document["document_id"],
                "filename": document["filename"],
            })

            # Synthetic ranking values for deterministic testing.
            distances.append(None)
            hybrid_scores.append(0.0)

    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
        "hybrid_scores": [hybrid_scores],
    }


def evaluate_context_processing():

    print("\nKLYRA MULTI-DOCUMENT CONTEXT EVALUATION")
    print("=======================================")

    results = build_synthetic_results()

    passed = 0
    total = len(MULTI_DOCUMENT_EVALUATION_DATASET)

    for index, test_case in enumerate(
        MULTI_DOCUMENT_EVALUATION_DATASET,
        start=1
    ):

        question = test_case["question"]
        should_find = test_case["should_find_answer"]

        processed = context_service.process(
            query=question,
            results=results,
            max_chunks=3,
        )

        retrieved_documents = (
            processed.get("documents") or [[]]
        )[0]

        retrieved_metadata = (
            processed.get("metadatas") or [[]]
        )[0]

        combined_text = " ".join(
            retrieved_documents
        ).lower()

        retrieved_document_ids = {
            metadata.get("document_id")
            for metadata in retrieved_metadata
        }

        expected_document_ids = set(
            test_case["expected_document_ids"]
        )

        expected_terms = [
            term.lower()
            for term in test_case["expected_terms"]
        ]

        if not should_find:

            # Context processing does not perform relevance filtering.
            # This check is intentionally reported as informational.
            success = True
            status = "NOT APPLICABLE"

        else:

            correct_document_found = bool(
                retrieved_document_ids.intersection(
                    expected_document_ids
                )
            )

            terms_found = all(
                term in combined_text
                for term in expected_terms
            )

            success = correct_document_found and terms_found
            status = "PASS" if success else "FAIL"

            if success:
                passed += 1

        print(f"\nTest {index}/{total}")
        print(f"Question: {question}")
        print(f"Retrieved chunks: {len(retrieved_documents)}")
        print(f"Retrieved documents: {sorted(retrieved_document_ids)}")
        print(f"Expected documents: {sorted(expected_document_ids)}")
        print(f"Result: {status}")

    print("\nEVALUATION SUMMARY")
    print("------------------")
    print(f"Relevant-answer tests: {passed}/5")
    print(
        "Note: This evaluates context selection, not actual "
        "semantic or hybrid retrieval."
    )


if __name__ == "__main__":
    evaluate_context_processing()
