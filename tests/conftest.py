import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import vector_store


@pytest.fixture(autouse=True)
def clear_store():
    vector_store._chunks.clear()
    yield
    vector_store._chunks.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def token(client):
    response = client.post("/api/v1/auth/token", json={"username": "tester", "role": "user"})
    return response.json()["access_token"]
