
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="Message sent to Klyra AI"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve"
    )

    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID for continuing a chat"
    )


class ChatSource(BaseModel):
    filename: str
    document_id: str | None = None
    distance: float | None = None


class ChatResponse(BaseModel):
    response: str

    sources: list[ChatSource] = Field(
        default_factory=list
    )

    model: str

    conversation_id: str
