# AI RAG Service

Production-oriented Retrieval-Augmented Generation service for document ingestion, embeddings, semantic retrieval, reranking, grounded generation, and RAG evaluation.

## Architecture

```text
Client -> API Gateway -> RAG API
                         |-> Ingestion -> Chunking -> Embeddings -> Vector Store
                         |-> Retrieval -> Reranking -> Context Builder -> LLM
                         |-> Evaluation
```

## Responsibilities

- Document ingestion and preprocessing
- Chunking and metadata management
- Embedding generation and vector indexing
- Semantic search and metadata filtering
- Grounded LLM responses with source attribution
- Retrieval and generation evaluation
- Secure API access and production observability

## API Boundaries

The service owns document processing, retrieval, vector indexing, RAG context construction, generation, and RAG evaluation. It does not own user identity storage, agent orchestration, ML model training, or infrastructure provisioning.

### Endpoints

```http
GET  /health
GET  /ready
POST /api/v1/documents
POST /api/v1/search
POST /api/v1/rag/query
```

Example query:

```json
{"question":"What caused the compliance reporting delay?","top_k":5}
```

Example response:

```json
{"answer":"The primary cause was delayed source-system ingestion.","sources":[{"document_id":"doc-123","score":0.91}]}
```

## Security

Protected APIs require `Authorization: Bearer <JWT>`. Validate JWTs before protected handlers and enforce role-based permissions for ingestion, search, and evaluation operations.

Secrets such as `LLM_API_KEY`, `DATABASE_URL`, `VECTOR_DB_URL`, and JWT configuration must be supplied through environment variables or a production secret manager. Never commit credentials, tokens, or sensitive document contents.

Additional controls include input validation, request limits, rate limiting, TLS, audit logging, prompt-injection defenses, document-level authorization, and redaction of sensitive values from logs.

## Local Setup

Requirements: Python 3.11+, Docker, PostgreSQL with pgvector or Qdrant, and an LLM/embedding provider.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

API docs are available at `/docs` during local development.

## Testing

```bash
pytest
pytest --cov=app --cov-report=term-missing
```

Evaluation should measure retrieval precision/recall, context relevance, faithfulness, answer correctness, latency, and token usage.

## Deployment

Build the container with:

```bash
docker build -t ai-rag-service .
```

Production Kubernetes should use a Deployment, Service, ConfigMap, Secret references, readiness/liveness probes, resource requests and limits, HPA, NetworkPolicy, and ServiceMonitor.

```bash
kubectl apply -f k8s/
```

## Observability

Recommended metrics include request rate, request latency, retrieval latency, embedding latency, LLM latency, token usage, retrieval result counts, and error counts. Export structured logs with request IDs and never log secrets or raw sensitive document content.

## Technology

Python, FastAPI, Pydantic, PostgreSQL/pgvector or Qdrant, configurable embedding/LLM providers, Pytest, Docker, Kubernetes, Prometheus, Grafana, and GitHub Actions.

## License

For portfolio and demonstration purposes.
