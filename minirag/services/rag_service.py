import numpy as np

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
    def cosine_similarity(
        left_embedding: list[float],
        right_embedding: list[float],
    ) -> float:
        left_vector = np.array(left_embedding, dtype=float)
        right_vector = np.array(right_embedding, dtype=float)

        left_norm = np.linalg.norm(left_vector)
        right_norm = np.linalg.norm(right_vector)
        if left_norm == 0 or right_norm == 0:
            return 0.0

        return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))

    @staticmethod
    def retrieve_chunks(
        query: str,
        collection: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        query_emb = RagService.generate_embeddings(query)
        scored_records = []
        for index, record in enumerate(collection):
            record.similarity = RagService.cosine_similarity(
                record.embedding, query_emb
            )
            scored_records.append((index, record))

        top_records = sorted(
            scored_records,
            key=lambda item: item[1].similarity,
            reverse=True,
        )[:top_k]

        return [record for _, record in top_records]

    @staticmethod
    @track_stats
    def similarity_search(
        query: str,
        collection: list[Chunk],
        top_k: int = 5,
    ) -> str:
        retrieved_chunks = RagService.retrieve_chunks(query, collection, top_k)
        return "\n\n".join([record.text for record in retrieved_chunks])
