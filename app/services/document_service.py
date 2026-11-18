
from pathlib import Path
from pypdf import PdfReader


class DocumentService:
    """Extracts text and splits documents into smaller chunks."""

    def extract_text(self, file_path: str) -> str:
        """Extract text from a PDF file."""

        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)

        return "\n\n".join(pages)

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ) -> list[str]:
        """Split text into overlapping chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be between 0 and chunk_size"
            )

        words = text.split()
        chunks = []

        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - chunk_overlap

        return chunks


document_service = DocumentService()
