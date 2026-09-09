from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.repositories import DocumentRepository, InMemoryDocumentRepository, PostgresDocumentRepository, StoredChunk


# Kept for backwards-compatible unit tests and local test fixtures.
vector_store = InMemoryDocumentRepository()


def get_repository(db: Session | None = None) -> DocumentRepository:
    """Return the persistence implementation used by the API.

    Production requests use PostgreSQL/pgvector. Tests can inject the in-memory
    implementation without requiring a database server.
    """
    if db is not None:
        return PostgresDocumentRepository(db)
    return PostgresDocumentRepository(SessionLocal())


__all__ = ["DocumentRepository", "InMemoryDocumentRepository", "PostgresDocumentRepository", "StoredChunk", "get_repository", "vector_store"]
