# AI RAG Service

Production-oriented Retrieval-Augmented Generation API built with FastAPI. This increment adds PostgreSQL + pgvector persistence while keeping the application storage behind a repository abstraction.

## Architecture

```text
Client
  |
  v
FastAPI
  |-- JWT Authentication
  |-- Document Ingestion -> Chunking -> Embeddings
  |                         |
  |                         v
  |                    PostgreSQL + pgvector
  |
  `-- Search -> cosine similarity -> Top-K chunks
```

## Repository Structure

```text
ai-rag-service/
├── app/
│   ├── chunking.py
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── repositories.py
│   ├── schemas.py
│   ├── security.py
│   └── store.py
├── alembic/
│   ├── versions/0001_initial_pgvector.py
│   ├── env.py
│   └── script.py.mako
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_chunking.py
│   └── test_repository.py
├── .env.example
├── alembic.ini
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Persistence

- PostgreSQL is the system of record for documents and chunks.
- pgvector stores 384-dimensional embeddings using the `vector` type.
- SQLAlchemy 2.x provides ORM models and sessions.
- Alembic manages schema migrations.
- An HNSW index uses cosine distance for scalable nearest-neighbor search.
- `DocumentRepository` isolates persistence from the FastAPI routes.
- `InMemoryDocumentRepository` remains available for fast unit/API tests.

## Database Setup

The included Compose stack starts `pgvector/pgvector:pg16`, creates the database, waits for PostgreSQL health, then runs `alembic upgrade head` before starting FastAPI.

```bash
docker compose up --build
```

The database is persisted in the `postgres_data` Docker volume.

For local Python execution, configure:

```bash
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/ragdb
```

Run migrations manually with:

```bash
alembic upgrade head
```

Rollback the latest migration with:

```bash
alembic downgrade -1
```

## API

```http
GET  /health
GET  /ready
POST /api/v1/auth/token
POST /api/v1/documents
POST /api/v1/search
```

`/api/v1/documents` and `/api/v1/search` require a bearer JWT.

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@policy.txt"
```

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the compliance reporting process?","top_k":5}'
```

Search now executes cosine-distance retrieval directly in PostgreSQL rather than scanning an in-memory list.

## Configuration

- `APP_NAME`
- `ENVIRONMENT`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `JWT_EXPIRE_MINUTES`
- `EMBEDDING_DIMENSION`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `DATABASE_URL`

Never commit `.env` or production secrets.

## Tests

```bash
pytest -q
```

The default suite remains database-independent by injecting the in-memory repository. The PostgreSQL integration test runs when `TEST_DATABASE_URL` is set and verifies persistence plus vector retrieval against a real pgvector database.

## Technology

Python 3.11, FastAPI, SQLAlchemy, PostgreSQL, pgvector, Psycopg 3, Alembic, Pydantic Settings, PyJWT, NumPy, Pytest, Docker, and Docker Compose.

## Next Steps

1. Replace deterministic local embeddings with a real transformer embedding model.
2. Add PDF/DOCX ingestion and metadata filtering.
3. Add reranking and grounded LLM generation.
4. Add RAG evaluation metrics and observability.
5. Deploy the service and database with Kubernetes.
