
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
        """Store document chunks, embeddings and metadata."""

        ids: list[str] = [
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
        """Find an existing document using its SHA-256 hash."""

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
        top_k: int = 3
    ):
        """Find the most relevant document chunks."""

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def list_documents(self) -> list[dict[str, str | int]]:
        """List all stored documents and their chunk counts."""

        results = self.collection.get(
            include=["metadatas"]
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

        return list(documents.values())

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""

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
