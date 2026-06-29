import asyncio
import json

import httpx

from kvoptbench.client.openai_compat import OpenAICompatClient
from kvoptbench.schemas import ExperimentConfig, WorkloadItem


def _config(**overrides) -> ExperimentConfig:
    data = {
        "experiment_id": "client_test",
        "provider": "local",
        "engine": "vllm",
        "model_id": "real-model",
        "strategy": "baseline",
        "base_url": "http://testserver/v1",
        "workload_file": "workloads/generated/test.jsonl",
        "output_file": "results/raw/test.jsonl",
        "endpoint_type": "vllm",
        "healthcheck_path": "/v1/models",
        "retries": 0,
        "retry_backoff_seconds": 0,
        "request_timeout_seconds": 5,
        "stream": False,
    }
    data.update(overrides)
    return ExperimentConfig.model_validate(data)


def _item() -> WorkloadItem:
    return WorkloadItem(
        task_id="task-1",
        workload="shared_prefix_long_doc",
        category="prefix_cache",
        prompt="Return EXPECTED_ANSWER: endpoint-ok",
        expected_answer="endpoint-ok",
        target_input_tokens=100,
        target_output_tokens=10,
    )


def test_healthcheck_captures_models_and_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "real-model"}, {"id": "draft-model"}]},
            headers={"server": "uvicorn"},
        )

    client = OpenAICompatClient(_config(capture_response_headers=True), transport=httpx.MockTransport(handler))

    health = asyncio.run(client.healthcheck())

    assert health.ok is True
    assert health.status_code == 200
    assert health.model_ids == ["real-model", "draft-model"]
    assert health.headers["server"] == "uvicorn"


def test_healthcheck_reports_failure_without_exception() -> None:
    client = OpenAICompatClient(
        _config(),
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "down"})),
    )

    health = asyncio.run(client.healthcheck())

    assert health.ok is False
    assert health.status_code == 503
    assert "HTTP 503" in health.error_message


def test_non_streaming_chat_retries_transient_status() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if request.url.path.endswith("/chat/completions") and calls["count"] == 1:
            return httpx.Response(503, json={"error": "try again"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "endpoint-ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    client = OpenAICompatClient(_config(retries=1), transport=httpx.MockTransport(handler))

    response = asyncio.run(client.chat(_item()))

    assert response.success is True
    assert response.content == "endpoint-ok"
    assert response.response_metadata["retry_count"] == 1
    assert calls["count"] == 2


def test_streaming_chat_parses_openai_sse_chunks() -> None:
    first = {
        "choices": [{"delta": {"content": "endpoint"}, "finish_reason": None}],
    }
    second = {
        "choices": [{"delta": {"content": "-ok"}, "finish_reason": None}],
    }
    content = (
        f"data: {json.dumps(first)}\n\n"
        f"data: {json.dumps(second)}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = OpenAICompatClient(
        _config(stream=True),
        transport=httpx.MockTransport(handler),
    )

    response = asyncio.run(client.chat(_item()))

    assert response.success is True
    assert response.content == "endpoint-ok"
    assert response.ttft_ms is not None
    assert response.response_metadata["retry_count"] == 0
