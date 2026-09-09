from app.main import app, repository
from app.repositories import InMemoryDocumentRepository, StoredChunk


def test_rag_query_returns_grounded_answer_and_citations(client, token, monkeypatch):
    repo = InMemoryDocumentRepository()
    repo.add_document(
        "doc-1",
        "policy.md",
        [StoredChunk("doc-1", "policy.md", 0, "Retention is seven years.", [1.0] * 384)],
    )
    app.dependency_overrides[repository] = lambda: repo

    generated = {}

    class FakeLLM:
        def generate(self, prompt):
            generated["prompt"] = prompt
            assert "[Source 1]" in prompt
            assert "Retention is seven years." in prompt
            return "Records are retained for seven years. [Source 1]"

    monkeypatch.setattr("app.main.get_llm", lambda: FakeLLM())
    try:
        response = client.post(
            "/api/v1/rag/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "How long are records retained?", "top_k": 1},
        )
    finally:
        app.dependency_overrides.pop(repository, None)

    assert response.status_code == 200
    assert generated["prompt"]

    body = response.json()
    assert body["answer"] == "Records are retained for seven years. [Source 1]"
    assert body["citations"] == [
        {
            "source_id": "Source 1",
            "filename": "policy.md",
            "chunk_index": 0,
            "score": 1.0,
        }
    ]


def test_rag_query_requires_authentication(client):
    response = client.post(
        "/api/v1/rag/query",
        json={"question": "How long are records retained?"},
    )

    assert response.status_code == 401


def test_rag_query_returns_no_results_fallback_without_generating(client, token, monkeypatch):
    repo = InMemoryDocumentRepository()
    app.dependency_overrides[repository] = lambda: repo

    def fail_if_called():
        raise AssertionError("LLM should not be called when retrieval returns no results")

    monkeypatch.setattr("app.main.get_llm", fail_if_called)
    try:
        response = client.post(
            "/api/v1/rag/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "What is not in the knowledge base?", "top_k": 1},
        )
    finally:
        app.dependency_overrides.pop(repository, None)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't have enough information in the provided sources.",
        "citations": [],
    }
