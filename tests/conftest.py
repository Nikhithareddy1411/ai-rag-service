import pytest
from fastapi.testclient import TestClient

from app.main import app, repository
from app.store import vector_store


@pytest.fixture(autouse=True)
def clear_store():
    vector_store._chunks.clear()
    app.dependency_overrides[repository] = lambda: vector_store
    yield
    vector_store._chunks.clear()
    app.dependency_overrides.pop(repository, None)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def token(client):
    response = client.post("/api/v1/auth/token", json={"username": "tester", "role": "user"})
    return response.json()["access_token"]
