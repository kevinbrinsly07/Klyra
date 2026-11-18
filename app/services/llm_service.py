
from openai import OpenAI, APIError, APITimeoutError

from app.core.config import settings


class LLMService:

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=30.0,
            max_retries=2
        )

    def generate_response(self, prompt: str) -> str:

        try:
            response = self.client.responses.create(
                model=settings.OPENAI_MODEL,
                instructions=(
                    "You are LankaMind AI, a helpful, "
                    "accurate and friendly AI assistant. "
                    "Answer clearly and honestly. "
                    "Never invent information."
                ),
                input=prompt,
                max_output_tokens=512
            )

            return response.output_text

        except APITimeoutError:
            raise RuntimeError(
                "The AI service timed out. Please try again."
            )

        except APIError as error:
            raise RuntimeError(
                f"AI service error: {error.message}"
            )


llm_service = LLMService()
