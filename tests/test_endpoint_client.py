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


def test_non_streaming_chat_captures_reasoning_metadata_without_storing_trace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "I should compute this before answering.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                    "total_tokens": 12,
                },
            },
        )

    client = OpenAICompatClient(_config(), transport=httpx.MockTransport(handler))

    response = asyncio.run(client.chat(_item()))

    assert response.success is True
    assert response.content == ""
    assert response.output_tokens == 0
    assert response.provider_completion_tokens == 7
    assert response.reasoning_content_present is True
    assert response.reasoning_content is None
    assert response.reasoning_tokens == 7
    assert response.visible_answer_missing is True
    assert response.finish_reason == "stop"


def test_non_streaming_chat_can_opt_in_to_capture_reasoning_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "endpoint-ok",
                            "reasoning_content": "Short local reasoning trace.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        )

    client = OpenAICompatClient(
        _config(capture_reasoning_content=True),
        transport=httpx.MockTransport(handler),
    )

    response = asyncio.run(client.chat(_item()))

    assert response.content == "endpoint-ok"
    assert response.reasoning_content_present is True
    assert response.reasoning_content == "Short local reasoning trace."
    assert response.visible_answer_missing is False


def test_non_streaming_chat_sends_configured_tools_and_parses_tool_calls() -> None:
    item = _item().model_copy(
        update={
            "eval_type": "tool_calling",
            "metadata": {
                "openai_tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "description": "Look up an order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"][0]["function"]["name"] == "lookup_order"
        assert payload["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_order",
                                        "arguments": '{"order_id": "A123"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    client = OpenAICompatClient(_config(), transport=httpx.MockTransport(handler))

    response = asyncio.run(client.chat(item))

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "lookup_order"
    assert response.tool_calls[0].arguments == {"order_id": "A123"}
    assert response.tool_calls[0].arguments_json == '{"order_id": "A123"}'
    assert response.visible_answer_missing is False


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


def test_streaming_chat_captures_reasoning_timing_without_visible_ttft() -> None:
    first = {
        "choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}],
    }
    second = {
        "choices": [{"delta": {"reasoning_content": " more"}, "finish_reason": None}],
    }
    done = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    content = (
        f"data: {json.dumps(first)}\n\n"
        f"data: {json.dumps(second)}\n\n"
        f"data: {json.dumps(done)}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = OpenAICompatClient(_config(stream=True), transport=httpx.MockTransport(handler))

    response = asyncio.run(client.chat(_item()))

    assert response.success is True
    assert response.content == ""
    assert response.output_tokens == 0
    assert response.ttft_ms is None
    assert response.first_reasoning_token_ms is not None
    assert response.reasoning_content_present is True
    assert response.reasoning_content is None
    assert response.reasoning_tokens > 0
    assert response.visible_answer_missing is True
    assert response.finish_reason == "stop"


def test_streaming_chat_reassembles_tool_call_fragments() -> None:
    first = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup_order", "arguments": '{"order_id"'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    second = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": ': "A123"}'}}
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    done = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    content = (
        f"data: {json.dumps(first)}\n\n"
        f"data: {json.dumps(second)}\n\n"
        f"data: {json.dumps(done)}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    client = OpenAICompatClient(_config(stream=True), transport=httpx.MockTransport(handler))

    response = asyncio.run(client.chat(_item()))

    assert response.content == ""
    assert response.visible_answer_missing is False
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "lookup_order"
    assert response.tool_calls[0].arguments == {"order_id": "A123"}
