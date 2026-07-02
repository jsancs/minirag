from dataclasses import dataclass


@dataclass
class Chunk:
    document_name: str
    text: str
    embedding: list[float]
    chunk_id: str = ""
    chunk_index: int = 0
    page_number: int | None = None
    similarity: float = 0.0
