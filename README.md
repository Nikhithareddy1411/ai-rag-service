# AI RAG Service

Initial production-oriented Retrieval-Augmented Generation API built with FastAPI. This first increment implements secure JWT-protected document ingestion, text chunking, deterministic embeddings, in-memory vector retrieval, automated tests, and containerization. The storage and embedding layers are deliberately abstracted so they can be replaced by pgvector/Qdrant and a transformer or hosted embedding provider in the next increment.

## Architecture

```text
Client
  |
  v
FastAPI
  |-- JWT Authentication
  |-- Document Ingestion -> Chunking -> Embeddings
  `-- Search -> In-Memory Vector Store -> Top-K Results
```

## Repository Structure

```text
ai-rag-service/
├── app/
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── main.py
│   ├── schemas.py
│   ├── security.py
│   └── store.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_chunking.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Implemented API

```http
GET  /health
GET  /ready
POST /api/v1/auth/token
POST /api/v1/documents
POST /api/v1/search
```

`/api/v1/documents` and `/api/v1/search` require a bearer JWT.

### Get a development token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo","role":"user"}'
```

The token endpoint is a local demonstration issuer only. Production deployments should validate tokens issued by an enterprise identity provider rather than issuing credentials from the application.

### Ingest a document

The initial version accepts UTF-8 `text/plain`, `text/markdown`, and `application/json` uploads up to 2 MB.

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

The response returns document ID, filename, chunk index, chunk text, and similarity score.

## Configuration

Configuration is managed with `pydantic-settings` and can be supplied through `.env` or environment variables:

- `APP_NAME`
- `ENVIRONMENT`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `JWT_EXPIRE_MINUTES`
- `EMBEDDING_DIMENSION`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`

Never commit `.env` or production secrets.

## Embeddings and Retrieval

`LocalHashEmbedding` provides deterministic, dependency-light vector embeddings for the initial implementation. Retrieval uses vector dot-product similarity over normalized embeddings.

The embedding provider and vector store are isolated behind small interfaces/classes so the application can move to a production embedding model and PostgreSQL/pgvector or Qdrant without changing the API contract.

## Security

- JWT signature and expiration validation
- Bearer authentication on protected endpoints
- Role field included in authenticated claims
- Request schema validation with Pydantic
- 2 MB upload limit
- MIME-type allowlist
- No credentials committed to source control
- Non-root Docker runtime user

The demo token issuer must not be exposed as an authentication system in production. Use an external identity provider, rotate secrets, enforce RBAC, add rate limiting, TLS, audit logging, document-level authorization, and prompt-injection defenses as the platform grows.

## Local Development

Requirements: Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

OpenAPI documentation is available at `/docs` during development.

## Tests

```bash
pytest -q
```

The test suite covers health checks, authentication enforcement, token creation, document ingestion, retrieval, chunk overlap, and invalid chunk configuration.

## Docker

```bash
docker build -t ai-rag-service .
docker run --rm -p 8000:8000 \
  -e JWT_SECRET_KEY='replace-with-a-long-random-secret' \
  ai-rag-service
```

Or:

```bash
docker compose up --build
```

The image runs as a non-root user and includes a container health check through Docker Compose.

## Roadmap

1. PostgreSQL + pgvector/Qdrant persistence
2. Transformer/hosted embedding provider
3. PDF/DOCX ingestion
4. Metadata filtering and document authorization
5. Reranking
6. LLM grounded generation and `/api/v1/rag/query`
7. RAG evaluation metrics
8. Prometheus/Grafana observability
9. Kubernetes manifests and CI/CD

## Technology

Python 3.11, FastAPI, Pydantic Settings, PyJWT, NumPy, Pytest, Docker, and Docker Compose.

## License

For portfolio and demonstration purposes.
