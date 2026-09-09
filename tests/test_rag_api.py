from app.main import app
from app.repositories import InMemoryDocumentRepository, StoredChunk
from app.store import get_repository


def test_rag_query_returns_grounded_answer_and_citations(client, token, monkeypatch):
    repo = InMemoryDocumentRepository()
    repo.add_document(
        "doc-1",
        "policy.md",
        [StoredChunk("doc-1", "policy.md", 0, "Retention is seven years.", [1.0] * 384)],
    )
    app.dependency_overrides[get_repository] = lambda db: repo

    class FakeLLM:
        def generate(self, prompt):
            assert "[Source 1]" in prompt
            assert "Retention is seven years." in prompt
            return "Records are retained for seven years. [Source 1]"

    monkeypatch.setattr("app.main.get_llm", lambda: FakeLLM())
    response = client.post(
        "/api/v1/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "How long are records retained?", "top_k": 1},
    )
    app.dependency_overrides.pop(get_repository, None)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[Source 1]")
    assert body["citations"][0]["source_id"] == "Source 1"
    assert body["citations"][0]["filename"] == "policy.md"


def test_rag_query_requires_authentication(client):
    response = client.post("/api/v1/rag/query", json={"question": "test"})
    assert response.status_code == 401
