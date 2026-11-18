
from app.services.embedding_service import embedding_service

sentences = [
    "How do I apply for a driving licence?",
    "What is the process to get a driver's permit?",
    "How can I bake a chocolate cake?"
]

embeddings = embedding_service.generate_embeddings(sentences)

print("Number of sentences:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))
print("First embedding sample:", embeddings[0][:5])
