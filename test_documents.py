from app.vectorstore.chroma_service import chroma_service

documents = chroma_service.list_documents()

print("\nStored Documents")
print("----------------")

for document in documents:
    print(document)

print(f"\nTotal documents: {len(documents)}")