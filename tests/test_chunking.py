
from app.services.document_service import document_service


# Generate sample text containing many sentences.
sample_text = " ".join(
    [
        f"Sentence number {i} contains information about "
        f"artificial intelligence, document processing, "
        f"and knowledge retrieval."
        for i in range(1, 301)
    ]
)

# Test chunk generation.
chunks = document_service.split_into_chunks(
    text=sample_text,
    chunk_size=100,
    chunk_overlap=20
)

print("\nCHUNKING TEST")
print("=============")

print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks):

    word_count = len(chunk.split())

    print(f"\nChunk {index + 1}")
    print("Word count:", word_count)
    print("Preview:", chunk[:150])

    assert word_count <= 100, "Chunk exceeds maximum size!"

# Verify overlap between consecutive chunks.
for index in range(len(chunks) - 1):

    previous_words = chunks[index].split()
    next_words = chunks[index + 1].split()

    assert (
        previous_words[-20:] == next_words[:20]
    ), "Chunk overlap is incorrect!"

# Test empty document.
assert document_service.split_into_chunks("") == []

print("\nAll chunking tests passed!")
