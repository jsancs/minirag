import numpy as np
from typing import Sequence

from minirag.utils.stats_utils import track_stats
from minirag.models import Chunk
from minirag.utils.backend_manager import get_backend_instance
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RagService:
    @staticmethod
    def get_splitter(
        chunk_size: int = 500, chunk_overlap: int = 20
    ) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def generate_embeddings(
        src_text: str,
        model_name: str = "",
    ) -> list[float]:
        backend = get_backend_instance()
        return backend.generate_embeddings(src_text, model_name)

    @staticmethod
    @track_stats
    def similarity_search(
        query: str,
        collection: list[Chunk],
        top_k: int = 5,
    ) -> str:
        query_emb = RagService.generate_embeddings(query)
        scored_records = []
        for index, record in enumerate(collection):
            record.similarity = np.dot(record.embedding, query_emb)
            scored_records.append((index, record))

        top_records = sorted(
            scored_records,
            key=lambda item: item[1].similarity,
            reverse=True,
        )[:top_k]
        top_records.sort(key=lambda item: item[0])

        return "\n\n".join([record.text for _, record in top_records])
