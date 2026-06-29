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
from kvoptbench.schemas import ExperimentConfig, TimedResponse, WorkloadItem


class OpenAICompatClient:
    """Small client for OpenAI-compatible `/chat/completions` endpoints."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.timeout = httpx.Timeout(config.timeout_seconds)
        self.headers: dict[str, str] = {}
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if api_key:
                self.headers["Authorization"] = f"Bearer {api_key}"

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

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            if self.config.stream:
                return await self._streaming_chat(client, payload, item)
            return await self._non_streaming_chat(client, payload, item)

    async def _streaming_chat(
        self, client: httpx.AsyncClient, payload: dict[str, Any], item: WorkloadItem
    ) -> TimedResponse:
        started = time.perf_counter()
        first_token_at: float | None = None
        token_timestamps: list[float] = []
        output_parts: list[str] = []
        metadata: dict[str, Any] = {}
        try:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    return self._error_response(started, item, f"HTTP {resp.status_code}", text.decode())
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
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            return self._error_response(started, item, type(exc).__name__, str(exc))
        ended = time.perf_counter()
        if resp.status_code >= 400:
            return self._error_response(started, item, f"HTTP {resp.status_code}", resp.text)
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
        started: float, item: WorkloadItem, error_type: str, error_message: str
    ) -> TimedResponse:
        return TimedResponse(
            content="",
            input_tokens=estimate_tokens(item.prompt),
            output_tokens=0,
            e2e_latency_ms=milliseconds(time.perf_counter() - started),
            success=False,
            error_type=error_type,
            error_message=error_message,
        )

