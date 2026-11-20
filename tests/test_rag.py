
from app.services.rag_service import rag_service

print("LankaMind AI - RAG Test")
print("-----------------------")

# Ingest sample document
result = rag_service.ingest_document(
    file_path="sample_knowledge.pdf",
    filename="sample_knowledge.pdf"
)

print("\nDocument ingestion result:")
print(result)

# Ask a question
question = "What examinations must someone pass to obtain a driving licence in Sri Lanka?"

result = rag_service.answer_question(question)

print("\nLankaMind AI Answer:")
print(result["answer"])

print("\nSources:")
print(result["sources"])
