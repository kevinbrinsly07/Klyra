
from app.services.keyword_search_service import keyword_search_service


def test_keyword_cache():

    print("\nKlyra BM25 Cache Test")
    print("---------------------")

    # First search builds the index.
    keyword_search_service.search(
        "driving licence"
    )

    first_index = keyword_search_service._bm25

    assert first_index is not None

    print("First search: Index built.")

    # Second search should reuse the same index.
    keyword_search_service.search(
        "university admission"
    )

    second_index = keyword_search_service._bm25

    assert first_index is second_index

    print("Second search: Cached index reused.")

    # Explicit invalidation should clear the index.
    keyword_search_service.invalidate_cache()

    assert keyword_search_service._bm25 is None

    print("Cache invalidation: Passed.")

    # Next search should rebuild the index.
    keyword_search_service.search(
        "driving licence"
    )

    third_index = keyword_search_service._bm25

    assert third_index is not None
    assert third_index is not first_index

    print("Cache rebuild: Passed.")

    print("\nAll cache tests passed!")


if __name__ == "__main__":
    test_keyword_cache()
