
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
        filename: str
    ):
        """Store document chunks and their embeddings."""

        ids: list[str] = [
            f"{document_id}_{index}"
            for index in range(len(chunks))
        ]

        typed_embeddings = cast(list[Embedding], embeddings)

        metadatas: list[Metadata] = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index
            }
            for index in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=typed_embeddings,
            metadatas=metadatas
        )

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


chroma_service = ChromaService()
