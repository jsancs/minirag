from minirag.backends.base import Backend
from minirag.backends.ollama_backend import OllamaBackend
from minirag.backends.openai_backend import OpenAIBackend


BACKENDS = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
}


def get_backend(backend_name: str = "ollama") -> Backend:
    backend_class = BACKENDS.get(backend_name)
    if backend_class is None:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {', '.join(BACKENDS.keys())}"
        )
    return backend_class()
