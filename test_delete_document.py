from app.vectorstore.chroma_service import chroma_service

document_id = "8f29fc87-b5a4-49f9-b974-a0970cdcc9a1"

deleted_chunks = chroma_service.delete_document(document_id)

print(f"Deleted chunks: {deleted_chunks}")

print("\nRemaining documents:")
for document in chroma_service.list_documents():
    print(document)