
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Converts text into numerical embeddings."""

    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """Generate embeddings for multiple text chunks."""

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()


embedding_service = EmbeddingService()
