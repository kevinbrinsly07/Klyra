
from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIStatusError, RateLimitError

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import llm_service


app = FastAPI(
    title="LankaMind AI",
    description=(
        "An intelligent AI assistant powered by "
        "large language models."
    ),
    version="0.1.0"
)


@app.get("/")
def home():
    return {
        "name": "LankaMind AI",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": settings.OPENAI_MODEL
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["AI Chat"]
)
def chat(request: ChatRequest):

    try:
        answer = llm_service.generate_response(
            request.message
        )

        return ChatResponse(
            response=answer,
            model=settings.OPENAI_MODEL
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="AI usage limit reached. Check API billing."
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to the AI provider."
        )

    except APIStatusError:
        raise HTTPException(
            status_code=502,
            detail="AI provider returned an error."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected AI service error occurred."
        )
