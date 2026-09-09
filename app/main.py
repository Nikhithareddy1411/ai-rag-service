from re import findall
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.chunking import chunk_text
from app.config import settings
from app.database import get_db
from app.embeddings import embedding_model
from app.generation import build_grounded_prompt, get_llm
from app.repositories import DocumentRepository, StoredChunk
from app.schemas import Citation, DocumentResponse, RAGRequest, RAGResponse, SearchRequest, SearchResponse, SearchResult, TokenRequest, TokenResponse
from app.security import create_access_token, get_current_user
from app.store import get_repository

app = FastAPI(title=settings.app_name, version="0.3.0")


def repository(db: Session = Depends(get_db)) -> DocumentRepository:
    return get_repository(db)


def parse_source_markers(answer: str, source_count: int) -> list[int]:
    markers = [int(value) for value in findall(r"\[Source\s+(\d+)\]", answer)]
    invalid = sorted({marker for marker in markers if marker < 1 or marker > source_count})
    if invalid:
        raise HTTPException(
            status_code=502,
            detail=f"Generated answer contains unsupported source markers: {invalid}",
        )
    return list(dict.fromkeys(markers))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(repo: DocumentRepository = Depends(repository)) -> dict[str, str | int]:
    return {"status": "ready", "indexed_chunks": repo.count()}


@app.post("/api/v1/auth/token", response_model=TokenResponse)
def issue_token(request: TokenRequest) -> TokenResponse:
    return TokenResponse(access_token=create_access_token(request.username, request.role))


@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...), user: dict = Depends(get_current_user), repo: DocumentRepository = Depends(repository)) -> DocumentResponse:
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
    stored_chunks = [StoredChunk(document_id=document_id, filename=file.filename or "unknown", chunk_index=chunk.index, text=chunk.text, vector=embedding_model.embed(chunk.text)) for chunk in chunks]
    repo.add_document(document_id, file.filename or "unknown", stored_chunks)
    return DocumentResponse(document_id=document_id, filename=file.filename or "unknown", chunks_created=len(chunks))


@app.post("/api/v1/search", response_model=SearchResponse)
def search(request: SearchRequest, _: dict = Depends(get_current_user), repo: DocumentRepository = Depends(repository)) -> SearchResponse:
    matches = repo.search(request.question, request.top_k)
    return SearchResponse(results=[SearchResult(document_id=chunk.document_id, filename=chunk.filename, chunk_index=chunk.chunk_index, text=chunk.text, score=round(score, 6)) for chunk, score in matches])


@app.post("/api/v1/rag/query", response_model=RAGResponse)
def rag_query(request: RAGRequest, _: dict = Depends(get_current_user), repo: DocumentRepository = Depends(repository)) -> RAGResponse:
    matches = repo.search(request.question, request.top_k)
    if not matches:
        return RAGResponse(answer="I don't have enough information in the provided sources.", citations=[])
    prompt = build_grounded_prompt(request.question, [(chunk.filename, chunk.text, chunk.chunk_index) for chunk, _ in matches])
    answer = get_llm().generate(prompt)
    cited_source_indexes = parse_source_markers(answer, len(matches))
    citations = [
        Citation(source_id=f"Source {index}", filename=matches[index - 1][0].filename, chunk_index=matches[index - 1][0].chunk_index, score=round(matches[index - 1][1], 6))
        for index in cited_source_indexes
    ]
    return RAGResponse(answer=answer, citations=citations)
