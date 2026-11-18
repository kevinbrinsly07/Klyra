
from app.services.keyword_search_service import keyword_search_service


def test_keyword_search():

    print("\nKlyra Keyword Search Test")
    print("-------------------------")

    queries = [
        "Department of Motor Traffic",
        "driving licence",
        "Advanced Level university admission",
        "quantum physics"
    ]

    for query in queries:

        print(f"\nQuery: {query}")

        results = keyword_search_service.search(
            query=query,
            top_k=3
        )

        documents = results["documents"][0]
        scores = results["scores"][0]

        if not documents:
            print("No keyword matches found.")
            continue

        for index, document in enumerate(documents):

            print(f"Result {index + 1}")
            print(f"BM25 score: {scores[index]:.4f}")
            print(f"Content: {document[:200]}...")


if __name__ == "__main__":
    test_keyword_search()
