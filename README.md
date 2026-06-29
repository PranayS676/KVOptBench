# KVOptBench

KVOptBench is a cache-aware LLM inference benchmark and strategy advisor.

It does not replace serving engines such as vLLM, SGLang, LMCache, Mooncake, or
llm-d. KVOptBench sits above those systems, sends controlled workloads to an
OpenAI-compatible endpoint, records request-level metrics, and turns comparison
results into practical strategy recommendations.

Use it to answer questions such as:

- When does prefix or radix caching actually reduce TTFT?
- How expensive is a cache miss?
- Is a workload prefill-bound, decode-bound, or mixed?
- Does long context create memory pressure or quality risk?
- Does KV quantization improve capacity without hurting quality?
- Does KV offload help or move the bottleneck to host memory and transfer?
- Does speculative decoding help decode-heavy work?
- Is prefill/decode disaggregation worth testing for mixed traffic?

## What KVOptBench Does

- Generates controlled benchmark workloads.
- Runs against mock, vLLM, SGLang, or generic OpenAI-compatible endpoints.
- Measures TTFT, TPOT, ITL, end-to-end latency, throughput, success rate, and quality fields.
- Preserves unavailable backend metrics as `null` and records them in `missing_metrics`.
- Produces JSONL request results, summary CSVs, comparison CSVs, Markdown reports, and strategy-advisor outputs.

## What KVOptBench Does Not Do

- It does not serve models.
- It does not provision GPUs.
- It does not download frontier model weights in tests.
- It does not manage Kubernetes or production orchestration.
- It does not fabricate engine internals when a backend does not expose them.

Users bring any reachable endpoint: local GPU, RunPod, Lambda Cloud, another cloud GPU provider,
bare metal, or an internal OpenAI-compatible serving stack.

## Status

The project currently includes:

- local mock OpenAI-compatible server
- YAML-driven experiment runner
- streaming and non-streaming timing capture
- workload generators for cache, long-context, decode-heavy, RAG, tool-calling, and agentic patterns
- vLLM and SGLang command previews
- real endpoint health checks and runner support
- cache, prefix-overlap, prefill/decode, long-context, KV quantization, KV offload, speculative decoding, and disaggregation comparisons
- public example bundle and report templates
- evidence-based strategy advisor

Real endpoint result collection is the next major validation step.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## Local Mock Quickstart

The mock path validates the benchmark harness without a GPU, RunPod, model weights, or external API keys.

Start the mock server:

```bash
python -m kvoptbench.mock_server --port 8000
```

In another terminal:

```bash
kvoptbench generate-workload --profile shared_prefix --out workloads/generated/shared_prefix_32k.jsonl
kvoptbench run --config examples/example_experiment_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/mock_report.md
```

Generated workloads, raw results, summaries, and reports are ignored by git by default.

## Bring Your Own Endpoint

KVOptBench expects a running OpenAI-compatible HTTP endpoint. The endpoint can be local or remote.

Example configs:

| Environment | Config |
|---|---|
| local vLLM | `examples/vllm_openai_compatible_config.yaml` |
| local SGLang | `examples/sglang_openai_compatible_config.yaml` |
| RunPod vLLM | `examples/runpod_vllm_openai_compatible_config.yaml` |
| RunPod SGLang | `examples/runpod_sglang_openai_compatible_config.yaml` |
| Lambda Cloud vLLM | `examples/lambda_cloud_vllm_openai_compatible_config.yaml` |
| generic OpenAI-compatible endpoint | `examples/generic_openai_compatible_config.yaml` |

Edit the config fields that match your server:

- `provider`
- `engine`
- `endpoint_type`
- `base_url`
- `model_id`
- `api_key_env`, only when authentication is required
- `workload_file`
- `output_file`

Then run:

```bash
kvoptbench validate-config --config examples/vllm_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/vllm_openai_compatible_config.yaml
kvoptbench run --config examples/vllm_openai_compatible_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/real_endpoint_report.md
```

## Strategy Experiments

The CLI can generate config plans and comparison CSVs for common inference-strategy tests:

- `cache-plan`, `cache-run`, `cache-compare`
- `prefix-sweep-compare`
- `prefill-decode-plan`, `prefill-decode-run`, `prefill-decode-compare`
- `long-context-plan`, `long-context-run`, `long-context-compare`
- `kv-quant-plan`, `kv-quant-run`, `kv-quant-compare`
- `kv-offload-plan`, `kv-offload-run`, `kv-offload-compare`
- `spec-decoding-plan`, `spec-decoding-run`, `spec-decoding-compare`
- `disagg-plan`, `disagg-run`, `disagg-compare`
- `strategy-recommend`

Command previews document how a compatible server may be started. They do not launch servers:

```bash
kvoptbench engine-command --engine vllm --strategy cache_on --model-id your/model
kvoptbench engine-command --engine sglang --strategy cache_off --model-id your/model
```

For official real endpoint results, record the exact backend command, engine version, model revision,
GPU type, workload hash, config hash, and `missing_metrics`.

## Public Example Bundle

`examples/public_release/` contains deterministic fixture CSVs, a mock report, and strategy-advisor
outputs. These examples prove the reporting pipeline works. They are not real vLLM, SGLang, LMCache,
Mooncake, or llm-d performance claims.

Regenerate the advisor and combined report with:

```bash
kvoptbench strategy-recommend \
  --summary examples/public_release/summary.csv \
  --cache-input examples/public_release/cache_summary.csv \
  --prefix-sweep-input examples/public_release/prefix_sweep.csv \
  --prefill-decode-input examples/public_release/prefill_decode.csv \
  --long-context-input examples/public_release/long_context.csv \
  --kv-quant-input examples/public_release/kv_quantization.csv \
  --kv-offload-input examples/public_release/kv_offload.csv \
  --spec-decoding-input examples/public_release/speculative_decoding.csv \
  --disagg-input examples/public_release/disaggregation.csv \
  --json-output reports/outputs/strategy_advisor.json \
  --markdown-output reports/outputs/strategy_advisor.md

kvoptbench report \
  --input examples/public_release/summary.csv \
  --cache-input examples/public_release/cache_summary.csv \
  --prefix-sweep-input examples/public_release/prefix_sweep.csv \
  --prefill-decode-input examples/public_release/prefill_decode.csv \
  --long-context-input examples/public_release/long_context.csv \
  --kv-quant-input examples/public_release/kv_quantization.csv \
  --kv-offload-input examples/public_release/kv_offload.csv \
  --spec-decoding-input examples/public_release/speculative_decoding.csv \
  --disagg-input examples/public_release/disaggregation.csv \
  --strategy-input reports/outputs/strategy_advisor.json \
  --output reports/outputs/mock_benchmark_report.md
```

## Guides

- `guides/reproducibility.md`
- `guides/real_endpoint_vllm_sglang.md`
- `guides/runpod.md`
- `guides/first_real_benchmark.md`
- `examples/public_release/result_template.md`
- `examples/public_release/blog_report_template.md`

## Contributing

Good contributions improve reproducibility, workload coverage, engine support, metric parsing,
quality evaluation, reporting, or safe public examples. See `CONTRIBUTING.md` and `SECURITY.md`.

## License

MIT License.
