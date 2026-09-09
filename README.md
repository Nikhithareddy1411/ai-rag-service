# AI RAG Service

Production-oriented Retrieval-Augmented Generation API built with FastAPI, PostgreSQL + pgvector, local transformer embeddings, and a configurable local LLM.

## Architecture

```text
Client
  |
  v
FastAPI
  |-- JWT Authentication
  |-- Document Ingestion -> Chunking -> Local Transformer Embeddings
  |                         |                    |
  |                         |                    v
  |                         |             all-MiniLM-L6-v2
  |                         v
  |                    PostgreSQL + pgvector
  |
  |-- Search -> cosine similarity -> Top-K chunks
  |
  `-- RAG Query -> Retrieve -> Citation-aware Prompt -> Local LLM -> Grounded Answer
                                                        |
                                                        v
                                             Qwen2.5-0.5B-Instruct
```

## Embeddings

The service uses `sentence-transformers/all-MiniLM-L6-v2` by default. It runs locally after its first model download and produces 384-dimensional embeddings for semantic retrieval.

```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

## Grounded Answer Generation

`POST /api/v1/rag/query` retrieves the top relevant chunks from pgvector, places each chunk behind an explicit `[Source N]` marker, and sends a citation-aware prompt to the configured local LLM. The API returns both the generated answer and structured citations.

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the compliance reporting process?","top_k":5}'
```

Example response shape:

```json
{
  "answer": "Compliance reporting requires timely validation. [Source 1]",
  "citations": [
    {"source_id":"Source 1","filename":"policy.md","chunk_index":2,"score":0.91}
  ]
}
```

The prompt explicitly instructs the model to use only retrieved sources, avoid unsupported facts, and cite factual claims. If retrieval returns no chunks, the endpoint returns a no-information response without calling the LLM.

## Local LLM

The default generator is `Qwen/Qwen2.5-0.5B-Instruct`, loaded locally through Hugging Face Transformers. Transformers supports local `text-generation` pipelines and configurable generation parameters. citeturn0search2turn0search4

```env
LLM_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
LLM_DEVICE=-1
LLM_MAX_NEW_TOKENS=256
LLM_TEMPERATURE=0.2
LLM_DO_SAMPLE=false
```

`LLM_DEVICE=-1` uses CPU. Set a supported device value for a local accelerator when available. The LLM is loaded lazily, so importing the application does not download model weights.

No hosted LLM API is required for this configuration. The first generation request downloads the selected model from Hugging Face and subsequent requests use the local cache.

## Repository Structure

```text
ai-rag-service/
├── app/
│   ├── chunking.py
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── generation.py
│   ├── main.py
│   ├── models.py
│   ├── repositories.py
│   ├── schemas.py
│   ├── security.py
│   └── store.py
├── alembic/
│   └── versions/0001_initial_pgvector.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_generation.py
│   ├── test_rag_api.py
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
- HNSW indexes support cosine-distance retrieval.
- `DocumentRepository` isolates persistence from the FastAPI routes.
- `InMemoryDocumentRepository` remains available for fast tests.

## Database Setup

```bash
docker compose up --build
```

The Compose stack starts PostgreSQL/pgvector, waits for database health, runs `alembic upgrade head`, and then starts FastAPI. PostgreSQL data is persisted in the `postgres_data` Docker volume.

## API

```http
GET  /health
GET  /ready
POST /api/v1/auth/token
POST /api/v1/documents
POST /api/v1/search
POST /api/v1/rag/query
```

Protected endpoints require a bearer JWT.

## Tests

```bash
pytest -q
```

Generation tests use a fake generator and therefore do not download LLM weights. The RAG API test verifies retrieval, citation-aware prompting, generated answers, and structured citations. Embedding tests similarly avoid model downloads. The PostgreSQL integration test runs when `TEST_DATABASE_URL` is set.

## Technology

Python 3.11, FastAPI, Sentence Transformers, Transformers, PyTorch, SQLAlchemy, PostgreSQL, pgvector, Psycopg 3, Alembic, Pydantic Settings, PyJWT, NumPy, Pytest, Docker, and Docker Compose.

## Next Steps

1. Add PDF/DOCX ingestion and metadata filtering.
2. Add reranking and citation validation.
3. Add RAG evaluation metrics, hallucination/faithfulness checks, and observability.
4. Deploy the service and database with Kubernetes.
