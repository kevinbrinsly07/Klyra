
import chromadb

from pathlib import Path
from typing import cast

from chromadb.api.types import (
    Embedding,
    Metadata,
)


class ChromaService:
    """Manages document embeddings in ChromaDB."""

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]
        database_path = project_root / "data" / "chroma"

        database_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(database_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="lankamind_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
        file_hash: str,
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings count must match."
            )

        ids = [
            f"{document_id}_{index}"
            for index in range(len(chunks))
        ]

        typed_embeddings = cast(
            list[Embedding],
            embeddings
        )

        metadatas: list[Metadata] = [
            {
                "document_id": document_id,
                "filename": filename,
                "file_hash": file_hash,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=typed_embeddings,
            metadatas=metadatas
        )

    def find_document_by_hash(self, file_hash: str):

        results = self.collection.get(
            where={"file_hash": file_hash},
            include=["metadatas"]
        )

        metadatas = results["metadatas"] or []

        if not metadatas:
            return None

        metadata = metadatas[0]

        return {
            "document_id": metadata["document_id"],
            "filename": metadata["filename"]
        }

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        min_similarity: float | None = None
    ):
        """Search relevant chunks using cosine similarity."""

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        if min_similarity is not None and not (
            0 <= min_similarity <= 1
        ):
            raise ValueError(
                "min_similarity must be between 0 and 1."
            )

        total_documents = self.collection.count()

        if total_documents == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_documents),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        if min_similarity is None:
            return results

        documents = []
        metadatas = []
        distances = []
        ids = []

        distances_result = results["distances"] or [[]]
        documents_result = results["documents"] or [[]]
        metadatas_result = results["metadatas"] or [[]]

        for index, distance in enumerate(
            distances_result[0]
        ):

            similarity = 1 - distance

            if similarity < min_similarity:
                continue

            documents.append(
                documents_result[0][index]
            )

            metadatas.append(
                metadatas_result[0][index]
            )

            distances.append(distance)

            ids.append(
                results["ids"][0][index]
            )

        results["documents"] = [documents]
        results["metadatas"] = [metadatas]
        results["distances"] = [distances]
        results["ids"] = [ids]

        return results

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict[str, str | int]]:

        if limit < 1 or offset < 0:
            raise ValueError("Invalid pagination parameters.")

        results = self.collection.get(
            include=["metadatas"],
            limit=None
        )

        documents = {}

        for metadata in results["metadatas"] or []:

            document_id = metadata["document_id"]
            filename = metadata["filename"]

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": filename,
                    "total_chunks": 0,
                }

            documents[document_id]["total_chunks"] += 1

        all_documents = list(documents.values())

        return all_documents[offset:offset + limit]

    def delete_document(self, document_id: str) -> int:

        results = self.collection.get(
            where={"document_id": document_id},
            include=[]
        )

        ids = results["ids"]

        if not ids:
            return 0

        self.collection.delete(ids=ids)

        return len(ids)


chroma_service = ChromaService()
