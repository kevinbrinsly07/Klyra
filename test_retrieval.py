
from app.vectorstore.chroma_service import chroma_service
from app.services.embedding_service import embedding_service

print("\n1. CHROMADB STATUS")
print("==================")

print("Stored chunks:", chroma_service.collection.count())
print("Stored documents:", chroma_service.list_documents())

print("\n2. RAW RETRIEVAL (WITHOUT SIMILARITY FILTER)")
print("============================================")

question = "What information is available in my uploaded document?"

query_embedding = embedding_service.generate_embedding(question)

raw_results = chroma_service.search(
    query_embedding=query_embedding,
    top_k=5,
    min_similarity=None
)

print("Documents:", raw_results.get("documents"))
print("Distances:", raw_results.get("distances"))
print("Metadata:", raw_results.get("metadatas"))

print("\n3. FILTERED RETRIEVAL (SIMILARITY >= 0.50)")
print("==========================================")

filtered_results = chroma_service.search(
    query_embedding=query_embedding,
    top_k=5,
    min_similarity=0.50
)

print("Documents:", filtered_results.get("documents"))
print("Distances:", filtered_results.get("distances"))


from app.services.rag_service import rag_service

questions = [
    "What are the requirements for obtaining a driving licence in Sri Lanka?",
    "What examinations must an applicant pass to get a driving licence?",
    "How does university admission work in Sri Lanka?"
]

for question in questions:

    print("\nQUESTION:", question)

    results = rag_service.retrieve_documents(
        question,
        top_k=3
    )

    print("DOCUMENTS:", results["documents"])
    print("DISTANCES:", results["distances"])