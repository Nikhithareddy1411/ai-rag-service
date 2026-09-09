def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_requires_authentication(client):
    response = client.post("/api/v1/search", json={"question": "test"})
    assert response.status_code == 401


def test_token_and_ingestion_and_search(client, token):
    headers = {"Authorization": f"Bearer {token}"}
    upload = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("policy.txt", b"Compliance reporting requires timely data validation and review.", "text/plain")},
    )
    assert upload.status_code == 201
    assert upload.json()["chunks_created"] == 1

    response = client.post(
        "/api/v1/search",
        headers=headers,
        json={"question": "How does compliance reporting work?", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json()["results"]
