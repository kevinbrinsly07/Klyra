from app.services.embedding_service import embedding_service
from app.services.keyword_search_service import keyword_search_service
from app.vectorstore.chroma_service import chroma_service


class HybridSearchService:
    """Combines semantic and keyword retrieval using RRF."""

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float | None = 0.50,
        rrf_k: int = 60,
        return_candidates: bool = False,
    ) -> dict:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10.")

        if min_similarity is not None and not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        # Retrieve more candidates than the final requested results.
        candidate_k = min(20, max(top_k * 3, 10))

        # 1. Semantic retrieval.
        query_embedding = embedding_service.generate_embedding(query)

        semantic_results = chroma_service.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            min_similarity=min_similarity,
        )

        # 2. Keyword retrieval.
        keyword_results = keyword_search_service.search(query=query, top_k=candidate_k)

        fused_results = {}

        # 3. Combine rankings using Reciprocal Rank Fusion.
        retrieval_sets = [semantic_results, keyword_results]

        for retrieval_index, results in enumerate(retrieval_sets):

            result_ids = (results.get("ids") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]

            distances = (
                (results.get("distances") or [[]])[0] if retrieval_index == 0 else []
            )

            for rank, document_id in enumerate(result_ids, start=1):

                if document_id not in fused_results:

                    document_index = rank - 1

                    fused_results[document_id] = {
                        "id": document_id,
                        "document": documents[document_index],
                        "metadata": (
                            metadatas[document_index]
                            if document_index < len(metadatas)
                            else {}
                        ),
                        "semantic_distance": None,
                        "rrf_score": 0.0,
                    }

                # RRF formula: 1 / (k + rank)
                fused_results[document_id]["rrf_score"] += 1 / (rrf_k + rank)

                # Preserve the semantic distance when available.
                if retrieval_index == 0 and rank - 1 < len(distances):
                    fused_results[document_id]["semantic_distance"] = distances[
                        rank - 1
                    ]

        # 4. Sort by combined RRF score.
        result_limit = candidate_k if return_candidates else top_k

        ranked_results = sorted(
            fused_results.values(), key=lambda item: item["rrf_score"], reverse=True
        )[:result_limit]

        return {
            "ids": [[item["id"] for item in ranked_results]],
            "documents": [[item["document"] for item in ranked_results]],
            "metadatas": [[item["metadata"] for item in ranked_results]],
            "distances": [[item["semantic_distance"] for item in ranked_results]],
            "hybrid_scores": [[item["rrf_score"] for item in ranked_results]],
        }


hybrid_search_service = HybridSearchService()
