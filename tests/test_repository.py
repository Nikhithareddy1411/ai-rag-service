import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.models import Document, DocumentChunk
from app.repositories import InMemoryDocumentRepository, PostgresDocumentRepository, StoredChunk


def sample_chunk(document_id: str = "doc-1") -> StoredChunk:
    return StoredChunk(
        document_id=document_id,
        filename="policy.txt",
        chunk_index=0,
        text="Compliance reporting requires timely data validation.",
        vector=[0.1] * 384,
    )


def test_in_memory_repository_contract():
    repo = InMemoryDocumentRepository()
    chunk = sample_chunk()
    repo.add_document(chunk.document_id, chunk.filename, [chunk])

    results = repo.search("compliance reporting", top_k=1)

    assert repo.count() == 1
    assert len(results) == 1
    assert results[0][0].document_id == "doc-1"


@pytest.mark.integration
def test_postgres_repository_persists_and_retrieves():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Document.__table__.create(connection, checkfirst=True)
        DocumentChunk.__table__.create(connection, checkfirst=True)

    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        repo = PostgresDocumentRepository(session)
        chunk = sample_chunk("integration-doc")
        repo.add_document(chunk.document_id, chunk.filename, [chunk])
        results = repo.search("compliance reporting", top_k=1)
        assert repo.count() == 1
        assert results[0][0].document_id == "integration-doc"

    with engine.begin() as connection:
        DocumentChunk.__table__.drop(connection, checkfirst=True)
        Document.__table__.drop(connection, checkfirst=True)
        connection.execute(text("DROP EXTENSION IF EXISTS vector"))
