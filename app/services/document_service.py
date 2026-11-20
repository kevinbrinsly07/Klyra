import re

from pypdf import PdfReader


class DocumentService:
    """Extracts text and splits documents into sentence-aware chunks."""

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
        self, text: str, chunk_size: int = 300, chunk_overlap: int = 50
    ) -> list[str]:
        """
        Split text into sentence-aware chunks with word-based overlap.

        Args:
            text: Document text.
            chunk_size: Maximum number of words per chunk.
            chunk_overlap: Number of words shared between consecutive chunks.

        Returns:
            A list of text chunks.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")

        if not text or not text.strip():
            return []

        # Split at sentence endings and paragraph boundaries.
        sentences = re.split(r"(?<=[.!?।෴])\s+|\n+", text.strip())

        chunks = []
        current_words = []

        for sentence in sentences:

            sentence_words = sentence.split()

            if not sentence_words:
                continue

            while sentence_words:

                available_space = chunk_size - len(current_words)

                # The sentence fits inside the current chunk.
                if len(sentence_words) <= available_space:

                    current_words.extend(sentence_words)
                    sentence_words = []

                else:

                    # Save the current chunk before adding more content.
                    if current_words:

                        chunks.append(" ".join(current_words))

                        # Preserve the configured overlap.
                        if chunk_overlap > 0:
                            current_words = current_words[-chunk_overlap:]
                        else:
                            current_words = []

                        available_space = chunk_size - len(current_words)

                    # Handle sentences longer than the remaining space.
                    words_to_add = min(available_space, len(sentence_words))

                    current_words.extend(sentence_words[:words_to_add])

                    sentence_words = sentence_words[words_to_add:]

        # Save the final chunk.
        if current_words:
            chunks.append(" ".join(current_words))

        return chunks


document_service = DocumentService()
