
from evaluation.retrieval_dataset import EVALUATION_DATASET

from app.services.embedding_service import embedding_service
from app.services.hybrid_search_service import hybrid_search_service
from app.vectorstore.chroma_service import chroma_service


def evaluate_search():

    print("\nKLYRA RETRIEVAL EVALUATION")
    print("=========================")

    if chroma_service.collection.count() == 0:
        print("Knowledge base is empty. Upload documents first.")
        return

    semantic_hits = 0
    hybrid_hits = 0
    total = len(EVALUATION_DATASET)

    for index, test_case in enumerate(EVALUATION_DATASET, start=1):

        question = test_case["question"]
        expected_terms = [
            term.lower()
            for term in test_case["expected_terms"]
        ]
        should_find = test_case["should_find_answer"]

        print(f"\nTest {index}/{total}")
        print(f"Question: {question}")

        # Semantic search
        query_embedding = embedding_service.generate_embedding(
            question
        )

        semantic_results = chroma_service.search(
            query_embedding=query_embedding,
            top_k=3,
            min_similarity=0.50
        )

        semantic_documents = (
            semantic_results.get("documents") or [[]]
        )[0]

        # Hybrid search
        hybrid_results = hybrid_search_service.search(
            query=question,
            top_k=3,
            min_similarity=0.50
        )

        hybrid_documents = (
            hybrid_results.get("documents") or [[]]
        )[0]

        def is_relevant(documents):

            combined_text = " ".join(
                documents
            ).lower()

            if not should_find:
                return len(documents) == 0

            if not documents:
                return False

            return all(
                term in combined_text
                for term in expected_terms
            )

        semantic_passed = is_relevant(semantic_documents)
        hybrid_passed = is_relevant(hybrid_documents)

        if semantic_passed:
            semantic_hits += 1

        if hybrid_passed:
            hybrid_hits += 1

        print(
            f"Semantic search: "
            f"{'PASS' if semantic_passed else 'FAIL'}"
        )

        print(
            f"Hybrid search: "
            f"{'PASS' if hybrid_passed else 'FAIL'}"
        )

    semantic_accuracy = semantic_hits / total * 100
    hybrid_accuracy = hybrid_hits / total * 100

    print("\nEVALUATION SUMMARY")
    print("------------------")

    print(
        f"Semantic search: "
        f"{semantic_hits}/{total} "
        f"({semantic_accuracy:.2f}%)"
    )

    print(
        f"Hybrid search: "
        f"{hybrid_hits}/{total} "
        f"({hybrid_accuracy:.2f}%)"
    )


if __name__ == "__main__":
    evaluate_search()
