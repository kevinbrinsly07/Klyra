
from openai import OpenAI

from app.core.config import settings


class LLMService:

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

    def generate_response(self, prompt: str) -> str:

        response = self.client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=(
                "You are LankaMind AI, a helpful, "
                "accurate and friendly AI assistant. "
                "Answer clearly and honestly. "
                "If you do not know something, say so."
            ),
            input=prompt,
            max_output_tokens=512
        )

        return response.output_text


llm_service = LLMService()
