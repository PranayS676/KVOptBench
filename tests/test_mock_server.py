from fastapi.testclient import TestClient

from kvoptbench.mock_server.main import create_app


def test_mock_server_models_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "mock-frontier-model"


def test_mock_server_non_streaming_chat_completion() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-frontier-model",
            "messages": [{"role": "user", "content": "Return alpha-123"}],
            "stream": False,
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0


def test_mock_server_streaming_chat_completion() -> None:
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-frontier-model",
            "messages": [{"role": "user", "content": "Return alpha-123"}],
            "stream": True,
            "max_tokens": 8,
        },
    ) as response:
        chunks = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert any("data:" in chunk for chunk in chunks)
    assert any("[DONE]" in chunk for chunk in chunks)


def test_mock_server_metrics_endpoint() -> None:
    client = TestClient(create_app())
    client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-frontier-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["total_requests"] == 1

