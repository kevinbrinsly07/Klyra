
import re
from threading import RLock

from rank_bm25 import BM25Plus

from app.vectorstore.chroma_service import chroma_service


class KeywordSearchService:
    """Searches stored document chunks using a cached BM25 index."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "does", "for", "from", "how", "i", "in", "is",
        "it", "of", "on", "or", "the", "to", "what",
        "when", "where", "which", "who", "why", "with",
        "have", "has", "had", "was", "were", "will",
        "would", "could", "should", "can", "do"
    }

    def __init__(self):

        self._bm25 = None
        self._ids = []
        self._documents = []
        self._metadatas = []
        self._corpus = []

        self._cached_count = -1
        self._initialized = False

        self._lock = RLock()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE
        )

    def invalidate_cache(self):
        """Invalidate the index after documents change."""

        with self._lock:

            self._bm25 = None
            self._ids = []
            self._documents = []
            self._metadatas = []
            self._corpus = []

            self._cached_count = -1
            self._initialized = False

    def _build_index(self):
        """Load documents and build the BM25 index."""

        results = chroma_service.collection.get(
            include=["documents", "metadatas"]
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

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

        bm25 = BM25Plus(corpus) if corpus else None

        self._bm25 = bm25
        self._ids = valid_ids
        self._documents = valid_documents
        self._metadatas = valid_metadatas
        self._corpus = corpus

        self._cached_count = chroma_service.collection.count()
        self._initialized = True

    def _get_index(self):
        """Return the cached index or rebuild it when necessary."""

        current_count = chroma_service.collection.count()

        with self._lock:

            if (
                not self._initialized
                or current_count != self._cached_count
            ):
                self._build_index()

            return (
                self._bm25,
                self._ids,
                self._documents,
                self._metadatas,
                self._corpus
            )

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

        meaningful_tokens = {
            token
            for token in query_tokens
            if token not in self.STOP_WORDS
        }

        if not meaningful_tokens:
            return self._empty_results()

        bm25, ids, documents, metadatas, corpus = (
            self._get_index()
        )

        if bm25 is None:
            return self._empty_results()

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        matched_indices = []

        for index in ranked_indices:

            document_tokens = set(corpus[index])

            if not meaningful_tokens.intersection(document_tokens):
                continue

            if scores[index] <= 0:
                continue

            matched_indices.append(index)

            if len(matched_indices) >= top_k:
                break

        return {
            "ids": [
                [ids[index] for index in matched_indices]
            ],
            "documents": [
                [documents[index] for index in matched_indices]
            ],
            "metadatas": [
                [metadatas[index] for index in matched_indices]
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
