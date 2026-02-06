from abc import ABC, abstractmethod
from typing import Generator


class Backend(ABC):
    @abstractmethod
    def generate_embeddings(self, text: str, model_name: str = "") -> list[float]:
        pass

    @abstractmethod
    def chat_streaming(
        self,
        model: str,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def check_model_exists(self, model_name: str) -> bool:
        pass

    @abstractmethod
    def pull_model(self, model_name: str) -> None:
        pass
