from dataclasses import dataclass

import numpy as np

from app.embeddings import embedding_model


@dataclass
class StoredChunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    vector: list[float]


class InMemoryVectorStore:
    """Small local vector store used for the initial implementation.

    The interface is deliberately compatible with replacing it by pgvector/Qdrant.
    """

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def add(self, chunk: StoredChunk) -> None:
        self._chunks.append(chunk)

    def search(self, query: str, top_k: int) -> list[tuple[StoredChunk, float]]:
        if not self._chunks:
            return []
        q = np.asarray(embedding_model.embed(query), dtype=np.float32)
        scored = []
        for chunk in self._chunks:
            v = np.asarray(chunk.vector, dtype=np.float32)
            score = float(np.dot(q, v))
            scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def count(self) -> int:
        return len(self._chunks)


vector_store = InMemoryVectorStore()
