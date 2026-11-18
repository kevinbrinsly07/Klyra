
import re

from rank_bm25 import BM25Plus

from app.vectorstore.chroma_service import chroma_service


class KeywordSearchService:
    """Searches stored document chunks using BM25 keyword ranking."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "does", "for", "from", "how", "i", "in", "is",
        "it", "of", "on", "or", "the", "to", "what",
        "when", "where", "which", "who", "why", "with",
        "have", "has", "had", "was", "were", "will",
        "would", "could", "should", "can", "do"
    }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> dict:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        query_tokens = self.tokenize(query)

        # Remove common words before checking keyword overlap.
        meaningful_tokens = {
            token for token in query_tokens
            if token not in self.STOP_WORDS
        }

        if not meaningful_tokens:
            return self._empty_results()

        results = chroma_service.collection.get(
            include=["documents", "metadatas"]
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        if not ids or not documents:
            return self._empty_results()

        corpus = []
        valid_ids = []
        valid_documents = []
        valid_metadatas = []

        for index, document in enumerate(documents):

            if not document or not document.strip():
                continue

            tokens = self.tokenize(document)

            if not tokens:
                continue

            corpus.append(tokens)
            valid_ids.append(ids[index])
            valid_documents.append(document)

            metadata = (
                metadatas[index]
                if index < len(metadatas) and metadatas[index]
                else {}
            )

            valid_metadatas.append(metadata)

        if not corpus:
            return self._empty_results()

        bm25 = BM25Plus(corpus)
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        matched_indices = []

        for index in ranked_indices:

            document_tokens = set(corpus[index])

            # Require at least one meaningful exact keyword match.
            if not meaningful_tokens.intersection(document_tokens):
                continue

            if scores[index] <= 0:
                continue

            matched_indices.append(index)

            if len(matched_indices) >= top_k:
                break

        return {
            "ids": [
                [valid_ids[index] for index in matched_indices]
            ],
            "documents": [
                [valid_documents[index] for index in matched_indices]
            ],
            "metadatas": [
                [valid_metadatas[index] for index in matched_indices]
            ],
            "scores": [
                [float(scores[index]) for index in matched_indices]
            ],
        }

    @staticmethod
    def _empty_results() -> dict:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "scores": [[]],
        }


keyword_search_service = KeywordSearchService()
