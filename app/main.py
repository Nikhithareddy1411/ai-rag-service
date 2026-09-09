from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.chunking import chunk_text
from app.config import settings
from app.embeddings import embedding_model
from app.schemas import DocumentResponse, SearchRequest, SearchResponse, SearchResult, TokenRequest, TokenResponse
from app.security import create_access_token, get_current_user
from app.store import StoredChunk, vector_store

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str | int]:
    return {"status": "ready", "indexed_chunks": vector_store.count()}


@app.post("/api/v1/auth/token", response_model=TokenResponse)
def issue_token(request: TokenRequest) -> TokenResponse:
    # Demo token issuer only. Replace with an enterprise IdP in production.
    return TokenResponse(access_token=create_access_token(request.username, request.role))


@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> DocumentResponse:
    if user.get("role") not in {"admin", "user"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    if file.content_type not in {"text/plain", "text/markdown", "application/json"}:
        raise HTTPException(status_code=415, detail="Initial version supports text, markdown, and JSON files")

    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="File exceeds 2 MB limit")
    text = raw.decode("utf-8", errors="strict")
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    document_id = str(uuid4())
    for chunk in chunks:
        vector_store.add(
            StoredChunk(
                document_id=document_id,
                filename=file.filename or "unknown",
                chunk_index=chunk.index,
                text=chunk.text,
                vector=embedding_model.embed(chunk.text),
            )
        )
    return DocumentResponse(document_id=document_id, filename=file.filename or "unknown", chunks_created=len(chunks))


@app.post("/api/v1/search", response_model=SearchResponse)
def search(request: SearchRequest, _: dict = Depends(get_current_user)) -> SearchResponse:
    matches = vector_store.search(request.question, request.top_k)
    return SearchResponse(
        results=[
            SearchResult(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=round(score, 6),
            )
            for chunk, score in matches
        ]
    )
