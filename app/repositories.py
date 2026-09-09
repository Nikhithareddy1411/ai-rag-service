from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import embedding_model
from app.models import Document, DocumentChunk


@dataclass
class StoredChunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    vector: list[float]


class DocumentRepository(Protocol):
    def add_document(self, document_id: str, filename: str, chunks: list[StoredChunk]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[StoredChunk, float]]: ...
    def count(self) -> int: ...


class InMemoryDocumentRepository:
    """Deterministic repository used by unit/API tests and local fallback tests."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def add_document(self, document_id: str, filename: str, chunks: list[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[StoredChunk, float]]:
        if not self._chunks:
            return []
        q = np.asarray(embedding_model.embed(query), dtype=np.float32)
        scored = []
        for chunk in self._chunks:
            v = np.asarray(chunk.vector, dtype=np.float32)
            scored.append((chunk, float(np.dot(q, v))))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def count(self) -> int:
        return len(self._chunks)


class PostgresDocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_document(self, document_id: str, filename: str, chunks: list[StoredChunk]) -> None:
        document = Document(id=document_id, filename=filename)
        document.chunks = [
            DocumentChunk(
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=chunk.vector,
            )
            for chunk in chunks
        ]
        self.session.add(document)
        self.session.commit()

    def search(self, query: str, top_k: int) -> list[tuple[StoredChunk, float]]:
        query_vector = embedding_model.embed(query)
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        statement = (
            select(DocumentChunk, Document.filename, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .order_by(distance)
            .limit(top_k)
        )
        rows = self.session.execute(statement).all()
        return [
            (
                StoredChunk(
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    vector=chunk.embedding,
                ),
                1.0 - float(row_distance),
            )
            for chunk, filename, row_distance in rows
        ]

    def count(self) -> int:
        return self.session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
