
from fastapi import FastAPI, HTTPException, status
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import rag_service
from app.routers.documents import router as documents_router


app = FastAPI(
    title="Klyra AI",
    description="An intelligent knowledge assistant powered by RAG.",
    version="0.2.0",
)

app.include_router(documents_router)


@app.get("/")
def home():
    return {
        "name": "Klyra AI",
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": settings.OPENAI_MODEL,
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["AI Chat"],
)
def chat(request: ChatRequest):

    try:
        result = rag_service.answer_question(
            question=request.message,
            top_k=request.top_k,
        )

        return ChatResponse(
            response=result["answer"],
            sources=result["sources"],
            model=settings.OPENAI_MODEL,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except RateLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="AI usage limit reached. Please try again later.",
        ) from error

    except APITimeoutError as error:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI provider request timed out.",
        ) from error

    except APIConnectionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to the AI provider.",
        ) from error

    except APIStatusError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider returned an error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI response generation failed.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected AI service error occurred.",
        ) from error
