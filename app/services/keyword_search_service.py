
import re

from rank_bm25 import BM25Plus

from app.vectorstore.chroma_service import chroma_service


class KeywordSearchService:
    """Searches stored document chunks using BM25 keyword ranking."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into lowercase searchable tokens."""

        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> dict:
        """Retrieve the most relevant chunks using keyword matching."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return self._empty_results()

        # Retrieve existing chunks without modifying ChromaDB.
        results = chroma_service.collection.get(
            include=["documents", "metadatas"]
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        if not ids or not documents:
            return self._empty_results()

        # Keep valid documents and their corresponding metadata.
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

        # BM25Plus also produces useful scores for small collections.
        bm25 = BM25Plus(corpus)

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        matched_indices = [
            index
            for index in ranked_indices
            if scores[index] > 0
        ][:top_k]

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
