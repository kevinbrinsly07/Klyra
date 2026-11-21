import uuid

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.services.rag_service import rag_service
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from app.vectorstore.chroma_service import chroma_service
from app.services.keyword_search_service import keyword_search_service

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

MAX_FILE_SIZE = 10 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    filename = Path(file.filename).name

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temporary_path = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"

    try:
        file_size = 0

        with temporary_path.open("wb") as buffer:

            while chunk := await file.read(1024 * 1024):

                file_size += len(chunk)

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 10 MB limit.",
                    )

                buffer.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        result = rag_service.ingest_document(
            file_path=str(temporary_path),
            filename=filename,
        )

        return DocumentUploadResponse(**result)

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        ) from error

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

        await file.close()


@router.get(
    "/",
    response_model=DocumentListResponse,
)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):

    documents = chroma_service.list_documents(
        limit=limit,
        offset=offset,
    )

    return {
        "total_documents": len(documents),
        "limit": limit,
        "offset": offset,
        "documents": documents,
    }


@router.get("/search")
def search_documents(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(default=10, ge=1, le=20),
):
    try:
        results = keyword_search_service.search(
            query=q,
            top_k=top_k,
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        scores = results["scores"][0]

        return {
            "query": q,
            "total_results": len(ids),
            "results": [
                {
                    "chunk_id": ids[index],
                    "content": documents[index],
                    "filename": metadatas[index].get("filename", "Unknown source"),
                    "document_id": metadatas[index].get("document_id"),
                    "score": scores[index],
                }
                for index in range(len(ids))
            ],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
def delete_document(document_id: str):

    deleted_chunks = chroma_service.delete_document(document_id)

    if deleted_chunks > 0:
        keyword_search_service.invalidate_cache()

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
    }
