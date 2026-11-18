
from app.services.hybrid_search_service import hybrid_search_service


def test_hybrid_search():

    print("\nKlyra Hybrid Search Test")
    print("------------------------")

    queries = [
        "Department of Motor Traffic",
        "What are the requirements to obtain a driving licence?",
        "How is Advanced Level used for university admission?",
        "quantum physics"
    ]

    for query in queries:

        print(f"\nQuery: {query}")

        results = hybrid_search_service.search(
            query=query,
            top_k=3
        )

        documents = results["documents"][0]
        scores = results["hybrid_scores"][0]
        distances = results["distances"][0]

        if not documents:
            print("No relevant documents found.")
            continue

        for index, document in enumerate(documents):

            print(f"\nResult {index + 1}")
            print(f"Hybrid score: {scores[index]:.6f}")
            print(f"Semantic distance: {distances[index]}")
            print(f"Content: {document[:200]}...")


if __name__ == "__main__":
    test_hybrid_search()
