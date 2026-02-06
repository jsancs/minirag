import ollama
from typing import Generator

from minirag.backends.base import Backend


class OllamaBackend(Backend):
    def generate_embeddings(
        self, text: str, model_name: str = "all-minilm"
    ) -> list[float]:
        emb = ollama.embeddings(
            model=model_name,
            prompt=text,
        )
        return list(emb["embedding"])

    def chat_streaming(
        self,
        model: str,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        stream = ollama.chat(  # type: ignore[no-overload-argument]
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            yield chunk["message"]["content"]

    def check_model_exists(self, model_name: str) -> bool:
        all_models = ollama.list()
        ollama_model_names = [model["name"] for model in all_models["models"]]
        return model_name in ollama_model_names

    def pull_model(self, model_name: str) -> None:
        print(f"Model {model_name} not found. Pulling model...")
        try:
            ollama.pull(model_name)
            print("Model pulled successfully.")
        except ollama.ResponseError as e:
            print(f"Error: {e}")
            raise ValueError("Model not found. Please provide a valid model name.")
