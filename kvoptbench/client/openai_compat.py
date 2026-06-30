"""Minimal OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from kvoptbench.client.timing import (
    average_inter_token_latency_ms,
    estimate_tokens,
    milliseconds,
)
from kvoptbench.schemas import (
    EndpointHealth,
    ExperimentConfig,
    TimedResponse,
    ToolCallRecord,
    WorkloadItem,
)

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatClient:
    """Small client for OpenAI-compatible `/chat/completions` endpoints."""

    def __init__(self, config: ExperimentConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.timeout = httpx.Timeout(config.request_timeout_seconds or config.timeout_seconds)
        self.transport = transport
        self.headers: dict[str, str] = {}
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if api_key:
                self.headers["Authorization"] = f"Bearer {api_key}"

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        normalized = "/" + path.lstrip("/")
        if self.base_url.endswith("/v1") and normalized.startswith("/v1/"):
            normalized = normalized.removeprefix("/v1")
        return f"{self.base_url}{normalized}"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            transport=self.transport,
        )

    async def healthcheck(self) -> EndpointHealth:
        """Probe the configured endpoint and capture safe metadata."""
        url = self._url(self.config.healthcheck_path)
        try:
            async with self._client() as client:
                response = await client.get(url)
        except httpx.HTTPError as exc:
            return EndpointHealth(ok=False, url=url, error_message=str(exc))

        headers = {}
        if self.config.capture_response_headers:
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"server", "x-request-id", "x-runtime", "date"}
            }
        model_ids: list[str] = []
        if response.status_code < 400:
            try:
                body = response.json()
                data = body.get("data", [])
                if isinstance(data, list):
                    model_ids = [
                        str(item["id"])
                        for item in data
                        if isinstance(item, dict) and item.get("id") is not None
                    ]
            except json.JSONDecodeError:
                pass
        return EndpointHealth(
            ok=response.status_code < 400,
            url=url,
            status_code=response.status_code,
            error_message=None if response.status_code < 400 else f"HTTP {response.status_code}",
            model_ids=model_ids,
            headers=headers,
            metadata={
                "endpoint_type": self.config.endpoint_type,
                "configured_metadata": self.config.endpoint_metadata,
            },
        )

    async def chat(self, item: WorkloadItem) -> TimedResponse:
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": item.prompt}],
            "stream": self.config.stream,
            "max_tokens": self.config.max_output_tokens,
            "kvoptbench_metadata": {
                "task_id": item.task_id,
                "workload": item.workload,
                "prefix_group_id": item.prefix_group_id,
                "shared_prefix_tokens": item.shared_prefix_tokens,
            },
        }
        payload.update(_tool_request_payload(item))

        last_response: TimedResponse | None = None
        for attempt in range(self.config.retries + 1):
            async with self._client() as client:
                if self.config.stream:
                    response = await self._streaming_chat(client, payload, item)
                else:
                    response = await self._non_streaming_chat(client, payload, item)
            response.response_metadata["retry_count"] = attempt
            if response.success or not _should_retry(response):
                return response
            last_response = response
            if attempt < self.config.retries and self.config.retry_backoff_seconds:
                await asyncio_sleep(self.config.retry_backoff_seconds)
        return last_response or self._error_response(time.perf_counter(), item, "RetryError", "no attempts")

    async def _streaming_chat(
        self, client: httpx.AsyncClient, payload: dict[str, Any], item: WorkloadItem
    ) -> TimedResponse:
        started = time.perf_counter()
        first_token_at: float | None = None
        first_reasoning_token_at: float | None = None
        token_timestamps: list[float] = []
        output_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_builders: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        provider_completion_tokens: int | None = None
        usage_reasoning_tokens: int | None = None
        metadata: dict[str, Any] = {}
        try:
            async with client.stream("POST", self._url("/chat/completions"), json=payload) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    return self._error_response(
                        started,
                        item,
                        f"HTTP {resp.status_code}",
                        text.decode(),
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("kvoptbench_mock"):
                        metadata.update(chunk["kvoptbench_mock"])
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        provider_completion_tokens = (
                            _int_or_none(usage.get("completion_tokens"))
                            or provider_completion_tokens
                        )
                        usage_reasoning_tokens = (
                            _reasoning_token_count(usage, "") or usage_reasoning_tokens
                        )
                    choice = _first_choice(chunk)
                    finish_reason = _finish_reason(choice) or finish_reason
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    content = _normalize_text(delta.get("content")) if isinstance(delta, dict) else ""
                    if content:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        token_timestamps.append(now)
                        output_parts.append(content)
                    reasoning_content = _normalize_text(
                        delta.get("reasoning_content") or delta.get("reasoning")
                    ) if isinstance(delta, dict) else ""
                    if reasoning_content:
                        now = time.perf_counter()
                        if first_reasoning_token_at is None:
                            first_reasoning_token_at = now
                        reasoning_parts.append(reasoning_content)
                    if self.config.capture_tool_calls and isinstance(delta, dict):
                        _update_tool_call_builders(tool_call_builders, delta.get("tool_calls"))
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return self._error_response(started, item, type(exc).__name__, str(exc))

        ended = time.perf_counter()
        output = "".join(output_parts)
        reasoning_text = "".join(reasoning_parts)
        tool_calls = _tool_call_records_from_builders(tool_call_builders)
        output_tokens = estimate_tokens(output)
        e2e_ms = milliseconds(ended - started)
        ttft_ms = milliseconds(first_token_at - started) if first_token_at is not None else None
        first_reasoning_ms = (
            milliseconds(first_reasoning_token_at - started)
            if first_reasoning_token_at is not None
            else None
        )
        itl_ms = average_inter_token_latency_ms(token_timestamps)
        tpot_ms = itl_ms
        return TimedResponse(
            content=output,
            input_tokens=estimate_tokens(item.prompt),
            output_tokens=output_tokens,
            provider_completion_tokens=provider_completion_tokens,
            reasoning_content=reasoning_text if self.config.capture_reasoning_content and reasoning_text else None,
            reasoning_content_present=bool(reasoning_text),
            reasoning_tokens=usage_reasoning_tokens
            or (estimate_tokens(reasoning_text) if reasoning_text else None),
            first_reasoning_token_ms=first_reasoning_ms,
            visible_answer_missing=not output.strip() and not tool_calls,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            itl_ms=itl_ms,
            e2e_latency_ms=e2e_ms,
            success=True,
            response_metadata=metadata,
        )

    async def _non_streaming_chat(
        self, client: httpx.AsyncClient, payload: dict[str, Any], item: WorkloadItem
    ) -> TimedResponse:
        started = time.perf_counter()
        try:
            resp = await client.post(self._url("/chat/completions"), json=payload)
        except httpx.HTTPError as exc:
            return self._error_response(started, item, type(exc).__name__, str(exc))
        ended = time.perf_counter()
        if resp.status_code >= 400:
            return self._error_response(
                started,
                item,
                f"HTTP {resp.status_code}",
                resp.text,
                status_code=resp.status_code,
            )
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            return self._error_response(started, item, type(exc).__name__, str(exc))

        choice = _first_choice(body)
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = _normalize_text(message.get("content")) if isinstance(message, dict) else ""
        reasoning_text = _normalize_text(
            message.get("reasoning_content") or message.get("reasoning")
        ) if isinstance(message, dict) else ""
        raw_tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        tool_calls = (
            [_parse_tool_call(raw) for raw in raw_tool_calls if isinstance(raw, dict)]
            if self.config.capture_tool_calls and isinstance(raw_tool_calls, list)
            else []
        )
        finish_reason = _finish_reason(choice)
        usage = body.get("usage") or {}
        metadata = body.get("kvoptbench_mock") or {}
        ttft_ms = metadata.get("simulated_ttft_ms")
        decode_ms = metadata.get("simulated_decode_ms")
        provider_completion_tokens = _int_or_none(usage.get("completion_tokens"))
        reasoning_tokens = _reasoning_token_count(usage, reasoning_text)
        return TimedResponse(
            content=content,
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(item.prompt)),
            output_tokens=estimate_tokens(content),
            provider_completion_tokens=provider_completion_tokens,
            reasoning_content=reasoning_text if self.config.capture_reasoning_content and reasoning_text else None,
            reasoning_content_present=bool(reasoning_text),
            reasoning_tokens=reasoning_tokens,
            visible_answer_missing=not content.strip() and not tool_calls,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            ttft_ms=round(float(ttft_ms), 3) if ttft_ms is not None else None,
            tpot_ms=round(float(decode_ms), 3) if decode_ms is not None else None,
            itl_ms=round(float(decode_ms), 3) if decode_ms is not None else None,
            e2e_latency_ms=milliseconds(ended - started),
            success=True,
            response_metadata=metadata,
        )

    @staticmethod
    def _error_response(
        started: float,
        item: WorkloadItem,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
    ) -> TimedResponse:
        return TimedResponse(
            content="",
            input_tokens=estimate_tokens(item.prompt),
            output_tokens=0,
            e2e_latency_ms=milliseconds(time.perf_counter() - started),
            success=False,
            error_type=error_type,
            error_message=error_message,
            response_metadata={"status_code": status_code} if status_code is not None else {},
        )


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _should_retry(response: TimedResponse) -> bool:
    status_code = response.response_metadata.get("status_code")
    return status_code in TRANSIENT_STATUS_CODES


def _tool_request_payload(item: WorkloadItem) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    tools = item.metadata.get("openai_tools") or item.metadata.get("tools")
    if tools:
        payload["tools"] = tools
    if "tool_choice" in item.metadata:
        payload["tool_choice"] = item.metadata["tool_choice"]
    if "parallel_tool_calls" in item.metadata:
        payload["parallel_tool_calls"] = item.metadata["parallel_tool_calls"]
    return payload


def _first_choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _finish_reason(choice: dict[str, Any]) -> str | None:
    value = choice.get("finish_reason") if isinstance(choice, dict) else None
    if value is None:
        return None
    return str(value)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reasoning_token_count(usage: dict[str, Any], reasoning_text: str) -> int | None:
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            value = _int_or_none(details.get("reasoning_tokens"))
            if value is not None:
                return value
    return estimate_tokens(reasoning_text) if reasoning_text else None


def _parse_tool_call(raw: dict[str, Any]) -> ToolCallRecord:
    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    arguments_raw = function.get("arguments")
    arguments_json: str | None = None
    arguments: dict[str, Any] | list[Any] | str | None = None
    parse_error: str | None = None
    if isinstance(arguments_raw, str):
        arguments_json = arguments_raw
        if arguments_raw.strip():
            try:
                parsed = json.loads(arguments_raw)
                if isinstance(parsed, (dict, list, str)):
                    arguments = parsed
                else:
                    arguments = str(parsed)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                arguments = arguments_raw
        else:
            arguments = {}
    elif isinstance(arguments_raw, (dict, list)):
        arguments = arguments_raw
        arguments_json = json.dumps(arguments_raw, ensure_ascii=False)
    elif arguments_raw is not None:
        arguments = str(arguments_raw)
        arguments_json = str(arguments_raw)

    return ToolCallRecord(
        id=str(raw["id"]) if raw.get("id") is not None else None,
        type=str(raw.get("type") or "function"),
        name=str(function["name"]) if function.get("name") is not None else None,
        arguments=arguments,
        arguments_json=arguments_json,
        arguments_parse_error=parse_error,
        index=_int_or_none(raw.get("index")),
    )


def _update_tool_call_builders(
    builders: dict[int, dict[str, Any]], tool_calls_delta: Any
) -> None:
    if not isinstance(tool_calls_delta, list):
        return
    for fallback_index, delta in enumerate(tool_calls_delta):
        if not isinstance(delta, dict):
            continue
        index = _int_or_none(delta.get("index"))
        if index is None:
            index = fallback_index
        builder = builders.setdefault(
            index,
            {"index": index, "id": None, "type": "function", "name": None, "arguments": ""},
        )
        if delta.get("id") is not None:
            builder["id"] = str(delta["id"])
        if delta.get("type") is not None:
            builder["type"] = str(delta["type"])
        function = delta.get("function")
        if isinstance(function, dict):
            if function.get("name") is not None:
                builder["name"] = (builder.get("name") or "") + str(function["name"])
            if function.get("arguments") is not None:
                builder["arguments"] = (builder.get("arguments") or "") + str(
                    function["arguments"]
                )


def _tool_call_records_from_builders(builders: dict[int, dict[str, Any]]) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []
    for index in sorted(builders):
        builder = builders[index]
        records.append(
            _parse_tool_call(
                {
                    "id": builder.get("id"),
                    "type": builder.get("type") or "function",
                    "index": index,
                    "function": {
                        "name": builder.get("name"),
                        "arguments": builder.get("arguments"),
                    },
                }
            )
        )
    return records

