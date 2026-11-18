
import shutil
import uuid

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.rag_service import rag_service
from app.schemas.document import DocumentUploadResponse


router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
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
                        detail="File exceeds the 10 MB limit."
                    )

                buffer.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        result = rag_service.ingest_document(
            file_path=str(temporary_path),
            filename=Path(file.filename).name
        )

        return DocumentUploadResponse(**result)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        ) from error

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

        await file.close()
