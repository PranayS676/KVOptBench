# AGENTS.md

This file defines rules for AI coding agents working on KVOptBench.

## Project Identity

- Project name: KVOptBench
- License: MIT
- Purpose: cache-aware frontier LLM inference benchmark and strategy optimizer
- Primary language: Python
- Initial scope: local/mock benchmark harness before RunPod execution

## Core Rules

1. Do not turn this project into an inference engine.
2. Do not build a UI unless explicitly requested.
3. Do not add Kubernetes in early milestones.
4. Do not add fine-tuning.
5. Do not add a vector database.
6. Do not require RunPod for local tests.
7. Do not require a GPU for Milestone 1.
8. Do not download frontier model weights during tests.
9. Do not commit secrets, API keys, Hugging Face tokens, RunPod tokens, or `.env` files.
10. Do not fake real engine metrics. If a metric is unavailable, store `null` and explain why.
11. Do not hardcode model names, engines, strategies, or workloads inside runner logic.
12. All experiments must be config-driven.
13. Every benchmark result must include enough metadata to reproduce the run.
14. Every new feature must include tests.
15. Keep dependencies minimal.
16. Prefer clear code over clever abstractions.

## Architecture Rules

KVOptBench sits above engines such as:

- vLLM
- SGLang
- LMCache
- Mooncake
- llm-d
- future KV/disaggregated serving systems

The project should use adapter interfaces for engine-specific behavior.

Engine-specific logic must not leak into generic runner code.

## Required Metadata for Results

Every result row should include:

- run_id
- experiment_id
- model_id
- engine
- strategy
- workload
- task_id
- provider
- gpu_type
- gpu_count
- concurrency
- input_tokens
- output_tokens
- target_input_tokens
- target_output_tokens
- shared_prefix_tokens
- cache_state
- cache_hit_rate or null
- cache_hit_proxy or null
- TTFT
- TPOT
- total latency
- success/error fields
- quality fields
- timestamp

## Milestone 1 Priority

Milestone 1 must implement:

- mock OpenAI-compatible server
- workload generation
- experiment runner
- JSONL result writing
- summary CSV generation
- markdown report generation
- basic evaluators
- tests

Do not implement RunPod automation before Milestone 1 is complete.

## Testing Rules

- `pytest` must pass.
- Unit tests should not require a GPU.
- Unit tests should not call external APIs.
- Unit tests should not require model downloads.
- Mock server tests must validate streaming and non-streaming responses.
- Schema tests must validate required result fields.

## Documentation Rules

Update docs when adding features.

Every new engine adapter should document:

- supported strategies
- unsupported strategies
- required server flags
- exposed metrics
- known limitations

Every new workload generator should document:

- workload purpose
- expected bottleneck
- required evaluation method
- expected metrics

## Safety and Secrets

Never print or store:

- API keys
- HF tokens
- RunPod tokens
- private endpoint URLs unless user explicitly asks
- private workload data

Generated reports should redact secrets and endpoint credentials.

## Style

- Use Python 3.11+.
- Use type hints where practical.
- Use pydantic for schemas where appropriate.
- Use structured errors.
- Prefer small modules.
- Keep CLI commands simple.
- Keep README examples copy-pasteable.
