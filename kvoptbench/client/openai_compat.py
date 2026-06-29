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
from kvoptbench.schemas import EndpointHealth, ExperimentConfig, TimedResponse, WorkloadItem

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
        token_timestamps: list[float] = []
        output_parts: list[str] = []
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
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        token_timestamps.append(now)
                        output_parts.append(content)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return self._error_response(started, item, type(exc).__name__, str(exc))

        ended = time.perf_counter()
        output = "".join(output_parts)
        output_tokens = estimate_tokens(output)
        e2e_ms = milliseconds(ended - started)
        ttft_ms = milliseconds(first_token_at - started) if first_token_at is not None else None
        itl_ms = average_inter_token_latency_ms(token_timestamps)
        tpot_ms = itl_ms
        return TimedResponse(
            content=output,
            input_tokens=estimate_tokens(item.prompt),
            output_tokens=output_tokens,
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

        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage") or {}
        metadata = body.get("kvoptbench_mock") or {}
        ttft_ms = metadata.get("simulated_ttft_ms")
        decode_ms = metadata.get("simulated_decode_ms")
        return TimedResponse(
            content=content,
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(item.prompt)),
            output_tokens=int(usage.get("completion_tokens") or estimate_tokens(content)),
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

