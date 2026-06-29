"""Local mock OpenAI-compatible server for Milestone 1."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kvoptbench.schemas import MockMetrics

EXPECTED_RE = re.compile(r"EXPECTED_ANSWER:\s*(?P<answer>[^\n\r]+)")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _message_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(str(chunk.get("text", "")))
    return "\n".join(parts)


def _extract_expected(prompt: str) -> str:
    match = EXPECTED_RE.search(prompt)
    if match:
        return match.group("answer").strip()
    return "mock-answer"


def _completion_text(prompt: str, max_tokens: int) -> str:
    expected = _extract_expected(prompt)
    words = ["Mock", "response", "for", expected]
    while len(words) < max(1, min(max_tokens, 64)):
        words.extend(["cache", "aware", "latency", "measurement"])
    return " ".join(words[: max(1, min(max_tokens, len(words)))])


def _cache_info(app: FastAPI, prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
    prefix_group_id = metadata.get("prefix_group_id")
    shared_prefix_tokens = int(metadata.get("shared_prefix_tokens") or 0)
    cache_key = prefix_group_id or prompt[:256]
    cache_state = "na"
    cache_hit_rate: float | None = None
    cache_miss_penalty_ms: float | None = None

    if shared_prefix_tokens > 0 or prefix_group_id:
        if cache_key in app.state.prefix_cache:
            cache_state = "warm"
            cache_hit_rate = 0.9
            app.state.metrics.cache_hits += 1
        else:
            cache_state = "cold"
            cache_hit_rate = 0.0
            cache_miss_penalty_ms = min(5000.0, max(5.0, shared_prefix_tokens * 0.02))
            app.state.prefix_cache.add(cache_key)
            app.state.metrics.cache_misses += 1
            app.state.metrics.warmed_prefixes = len(app.state.prefix_cache)

    return {
        "cache_state": cache_state,
        "cache_hit_rate": cache_hit_rate,
        "cache_miss_penalty_ms": cache_miss_penalty_ms,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="KVOptBench Mock Server")
    app.state.metrics = MockMetrics()
    app.state.prefix_cache = set()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "mock-frontier-model",
                    "object": "model",
                    "created": 0,
                    "owned_by": "kvoptbench",
                }
            ],
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return app.state.metrics.model_dump()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or 16)
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        error_rate = float(body.get("mock_error_rate") or 0.0)
        if error_rate > 0 and random.random() < error_rate:
            app.state.metrics.total_requests += 1
            app.state.metrics.error_requests += 1
            raise HTTPException(status_code=503, detail="simulated mock error")

        prompt = _message_text(messages)
        metadata = body.get("kvoptbench_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        cache = _cache_info(app, prompt, metadata)
        ttft_ms = float(body.get("mock_ttft_ms") or 5.0)
        decode_ms = float(body.get("mock_decode_ms") or 1.0)
        if cache["cache_state"] == "cold" and cache["cache_miss_penalty_ms"]:
            ttft_ms += float(cache["cache_miss_penalty_ms"])
        elif cache["cache_state"] == "warm":
            ttft_ms = max(1.0, ttft_ms * 0.4)

        content = _completion_text(prompt, max_tokens)
        tokens = content.split()
        created = int(time.time())
        request_id = f"chatcmpl-mock-{created}"
        app.state.metrics.total_requests += 1

        mock_meta = {
            **cache,
            "simulated_ttft_ms": ttft_ms,
            "simulated_decode_ms": decode_ms,
        }

        if not stream:
            app.state.metrics.non_streaming_requests += 1
            await asyncio.sleep(ttft_ms / 1000)
            await asyncio.sleep(max(0, len(tokens) - 1) * decode_ms / 1000)
            return JSONResponse(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": body.get("model", "mock-frontier-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": _estimate_tokens(prompt),
                        "completion_tokens": len(tokens),
                        "total_tokens": _estimate_tokens(prompt) + len(tokens),
                    },
                    "kvoptbench_mock": mock_meta,
                }
            )

        app.state.metrics.streaming_requests += 1

        async def event_stream():
            await asyncio.sleep(ttft_ms / 1000)
            for index, token in enumerate(tokens):
                if index > 0:
                    await asyncio.sleep(decode_ms / 1000)
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.get("model", "mock-frontier-model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token + (" " if index < len(tokens) - 1 else "")},
                            "finish_reason": None,
                        }
                    ],
                    "kvoptbench_mock": mock_meta if index == 0 else None,
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.get("model", "mock-frontier-model"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start KVOptBench mock server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

