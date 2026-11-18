
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    status: str
    message: str | None = None


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    total_chunks: int


class DocumentListResponse(BaseModel):
    total_documents: int
    limit: int
    offset: int
    documents: list[DocumentItem]


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str
    deleted_chunks: int
