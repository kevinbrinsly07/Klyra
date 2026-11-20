from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1, max_length=4000, description="Message sent to Klyra AI"
    )

    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of document chunks to retrieve"
    )

    conversation_id: str | None = Field(
        default=None, description="Existing conversation ID for continuing a chat"
    )


class ChatSource(BaseModel):
    filename: str
    document_id: str | None = None
    distance: float | None = None


class ChatResponse(BaseModel):
    response: str

    sources: list[ChatSource] = Field(default_factory=list)

    model: str

    conversation_id: str


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    id: str
    title: str | None = None
    created_at: str
    updated_at: str
    message_count: int
    last_message: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = Field(default_factory=list)
    limit: int
    offset: int


class RenameConversationRequest(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=100,
        description="New conversation title",
    )
