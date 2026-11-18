
import re


class ContextService:
    """Deduplicates and reranks retrieved document chunks."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "does", "for", "from", "how", "i", "in", "is",
        "it", "of", "on", "or", "the", "to", "what",
        "when", "where", "which", "who", "why", "with",
        "have", "has", "had", "was", "were", "will",
        "would", "could", "should", "can", "do"
    }

    NEAR_DUPLICATE_THRESHOLD = 0.90

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def calculate_coverage(
        self,
        query_tokens: set[str],
        document_tokens: set[str]
    ) -> float:
        """Calculate how many meaningful query terms occur in a chunk."""

        if not query_tokens:
            return 0.0

        matched = query_tokens.intersection(document_tokens)

        return len(matched) / len(query_tokens)

    @staticmethod
    def calculate_jaccard(
        first: set[str],
        second: set[str]
    ) -> float:
        """Calculate similarity between two token sets."""

        union = first.union(second)

        if not union:
            return 0.0

        intersection = first.intersection(second)

        return len(intersection) / len(union)

    def process(
        self,
        query: str,
        results: dict,
        max_chunks: int = 3
    ) -> dict:
        """Deduplicate and rerank hybrid search results."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not 1 <= max_chunks <= 10:
            raise ValueError("max_chunks must be between 1 and 10.")

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        hybrid_scores = (results.get("hybrid_scores") or [[]])[0]

        query_tokens = {
            token
            for token in self.tokenize(query)
            if token not in self.STOP_WORDS
        }

        candidates = []

        for index, document in enumerate(documents):

            if not document or not document.strip():
                continue

            tokens = set(self.tokenize(document))

            hybrid_score = (
                float(hybrid_scores[index])
                if index < len(hybrid_scores)
                else 0.0
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )

            candidates.append({
                "id": ids[index] if index < len(ids) else str(index),
                "document": document,
                "metadata": metadata,
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
                "hybrid_score": hybrid_score,
                "tokens": tokens,
                "normalized": self.normalize(document),
                "coverage": self.calculate_coverage(
                    query_tokens,
                    tokens
                )
            })

        if not candidates:
            return self._empty_results()

        max_hybrid_score = max(
            (item["hybrid_score"] for item in candidates),
            default=0.0
        )

        # Combine the original RRF ranking with lexical relevance.
        for item in candidates:

            normalized_rrf = (
                item["hybrid_score"] / max_hybrid_score
                if max_hybrid_score > 0
                else 0.0
            )

            item["rerank_score"] = (
                0.85 * normalized_rrf
                + 0.15 * item["coverage"]
            )

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        # Remove exact duplicates and highly similar chunks.
        unique_candidates = []

        for candidate in candidates:

            is_duplicate = False

            for selected in unique_candidates:

                # Exact duplicate detection.
                if candidate["normalized"] == selected["normalized"]:
                    is_duplicate = True
                    break

                # Avoid aggressive comparisons on very short chunks.
                if (
                    len(candidate["tokens"]) >= 20
                    and len(selected["tokens"]) >= 20
                ):
                    similarity = self.calculate_jaccard(
                        candidate["tokens"],
                        selected["tokens"]
                    )

                    if similarity >= self.NEAR_DUPLICATE_THRESHOLD:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_candidates.append(candidate)

            if len(unique_candidates) >= max_chunks:
                break

        return {
            "ids": [[item["id"] for item in unique_candidates]],
            "documents": [
                [item["document"] for item in unique_candidates]
            ],
            "metadatas": [
                [item["metadata"] for item in unique_candidates]
            ],
            "distances": [
                [item["distance"] for item in unique_candidates]
            ],
            "hybrid_scores": [
                [item["hybrid_score"] for item in unique_candidates]
            ],
            "rerank_scores": [
                [item["rerank_score"] for item in unique_candidates]
            ]
        }

    @staticmethod
    def _empty_results() -> dict:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "hybrid_scores": [[]],
            "rerank_scores": [[]]
        }


context_service = ContextService()
