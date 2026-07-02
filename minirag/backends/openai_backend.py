import os
from typing import Generator

_openai_available = False
try:
    from openai import OpenAI

    _openai_available = True
except ImportError:
    class OpenAI:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("openai package is not installed.")

        @property
        def embeddings(self) -> "OpenAI":
            return self

        @property
        def chat(self) -> "OpenAI":
            return self

        @property
        def completions(self) -> "OpenAI":
            return self

        def create(self, *args, **kwargs) -> None:
            raise ImportError("openai package is not installed.")

        @property
        def models(self) -> "OpenAI":
            return self

        def retrieve(self, *args, **kwargs) -> None:
            raise ImportError("openai package is not installed.")

from minirag.backends.base import Backend


class OpenAIBackend(Backend):
    def __init__(self):
        if not _openai_available:
            raise ImportError(
                "OpenAI backend is not available. Please install it with `uv sync --extra openai`."
            )
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
            model=model_name or "text-embedding-3-small",
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
