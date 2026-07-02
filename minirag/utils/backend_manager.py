from typing import cast

from minirag.backends.base import Backend
from minirag.backends import get_backend

_backend: Backend | None = None


def set_backend(backend_name: str = "ollama") -> None:
    global _backend
    _backend = get_backend(backend_name)


def get_backend_instance() -> Backend:
    global _backend
    if _backend is None:
        set_backend()
    return cast(Backend, _backend)
