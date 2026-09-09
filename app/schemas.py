from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    role: str = Field(default="user", pattern="^(user|admin)$")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
