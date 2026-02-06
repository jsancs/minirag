from typing import Generator
import os

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package is required for OpenAI backend. Install it with: uv sync --extra openai"
    )

from minirag.backends.base import Backend


class OpenAIBackend(Backend):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI backend"
            )
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_embeddings(
        self, text: str, model_name: str = "text-embedding-3-small"
    ) -> list[float]:
        response = self.client.embeddings.create(
            model=model_name,
            input=text,
        )
        return response.data[0].embedding

    def chat_streaming(
        self,
        model: str,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(  # type: ignore[no-overload-argument]
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def check_model_exists(self, model_name: str) -> bool:
        try:
            self.client.models.retrieve(model_name)
            return True
        except Exception:
            return False

    def pull_model(self, model_name: str) -> None:
        print(
            f"OpenAI models do not support pulling. Please ensure the model name is correct."
        )
        raise ValueError(
            "OpenAI models do not support pulling. Please ensure the model name is correct."
        )
