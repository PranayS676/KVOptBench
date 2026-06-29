# Real Endpoint Guide: vLLM And SGLang

This guide explains how to run KVOptBench against already-running vLLM and SGLang OpenAI-compatible endpoints. KVOptBench does not start, stop, deploy, or manage these servers. It records benchmark data and preserves missing telemetry honestly through `missing_metrics`.

Primary upstream references:

- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
- vLLM serve CLI docs: https://docs.vllm.ai/en/stable/cli/serve.html
- SGLang server arguments: https://docs.sglang.io/references/server_arguments.html
- SGLang OpenAI-compatible API docs: https://docs.sglang.io/backend/openai_api_completions.html

## Repo Entry Points

Use the public-safe configs as the starting point:

- `examples/vllm_openai_compatible_config.yaml`
- `examples/sglang_openai_compatible_config.yaml`

Generate command previews from the checked-in engine profiles:

```bash
kvoptbench engine-command --engine vllm --strategy baseline --model-id your/model
kvoptbench engine-command --engine vllm --strategy cache_on --model-id your/model
kvoptbench engine-command --engine sglang --strategy baseline --model-id your/model
kvoptbench engine-command --engine sglang --strategy cache_off --model-id your/model
```

These previews are documentation. They are not process launchers.

## Endpoint Contract

KVOptBench expects an OpenAI-compatible HTTP endpoint:

- `GET /v1/models`
- `POST /v1/chat/completions`
- optional streaming responses
- optional `/metrics` or other backend telemetry endpoint

The config fields that must match the running server are:

- `base_url`
- `model_id`
- `endpoint_type`
- `engine`
- `strategy`
- `api_key_env`, only when the endpoint requires authentication
- `workload_file`
- `output_file`

Validate before running:

```bash
kvoptbench validate-config --config examples/vllm_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/vllm_openai_compatible_config.yaml
```

## vLLM Baseline

A typical vLLM endpoint shape is:

```bash
vllm serve your/model --host 0.0.0.0 --port 8000
```

Then set:

```yaml
engine: vllm
endpoint_type: vllm
base_url: http://127.0.0.1:8000/v1
model_id: your/model
strategy: baseline
```

For prefix-cache experiments, run a cache-on variant with `--enable-prefix-caching` and compare it to a cache-disabled or baseline control appropriate for the installed vLLM version.

```bash
vllm serve your/model --host 0.0.0.0 --port 8000 --enable-prefix-caching
```

KV cache quantization should be treated as engine-version-specific. The built-in command preview uses `--kv-cache-dtype fp8`, but official runs should confirm the supported dtype values for the installed vLLM build before publishing.

Speculative decoding, KV offload, and prefill/decode disaggregation are also version- and deployment-specific. Replace placeholder flags from `kvoptbench engine-command` with the exact flags validated against the installed backend docs and record those flags in the public result template.

## SGLang Baseline

A typical SGLang endpoint shape is:

```bash
python -m sglang.launch_server --model-path your/model --host 0.0.0.0 --port 30000
```

Then set:

```yaml
engine: sglang
endpoint_type: sglang
base_url: http://127.0.0.1:30000/v1
model_id: your/model
strategy: baseline
```

SGLang uses radix caching as the normal cache-on condition. For a cache-off control, the checked-in profile uses `--disable-radix-cache`:

```bash
python -m sglang.launch_server --model-path your/model --host 0.0.0.0 --port 30000 --disable-radix-cache
```

KV cache dtype, speculative decoding, and disaggregation options should be confirmed against the installed SGLang server arguments before official publication. Record the exact flags rather than relying on shorthand names such as `kv_fp8` or `speculative_decoding`.

## Running A Smoke Test

Generate a small workload:

```bash
kvoptbench generate-workload --profile shared_prefix --count 5 --out workloads/generated/shared_prefix_smoke.jsonl
```

Edit the matching real endpoint config so `workload_file` points to that file and `output_file` points under `results/raw/`.

Run:

```bash
kvoptbench validate-config --config examples/vllm_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/vllm_openai_compatible_config.yaml
kvoptbench run --config examples/vllm_openai_compatible_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/outputs/real_endpoint_report.md
```

Repeat the same flow with `examples/sglang_openai_compatible_config.yaml` for SGLang.

## Strategy Experiment Order

Use this order for a public result set:

1. Baseline smoke run.
2. Shared-prefix cache ablation with random-prefix controls.
3. Prefix-overlap sweep.
4. Prefill/decode grid.
5. Long-context pressure sweep.
6. KV quantization comparison.
7. KV offload comparison, only when memory telemetry is available or explicitly listed as missing.
8. Speculative decoding comparison for decode-heavy workloads.
9. Prefill/decode disaggregation comparison for mixed prefill/decode workloads.
10. Strategy Advisor report over all comparison CSVs.

## Required Publication Metadata

Record these fields in `examples/public_release/result_template.md` before publishing:

- model id and model revision
- engine and engine version
- server command and all flags
- GPU type and count
- driver/CUDA/runtime image
- workload file hash
- config file hash
- raw JSONL result location
- summary CSV hash
- comparison CSV hashes
- strategy advisor JSON hash
- `missing_metrics`
- whether the run is exploratory or official

## Missing Metrics Rules

Do not infer unavailable telemetry. If an endpoint does not expose GPU memory, KV cache hit rate, speculative acceptance rate, or engine internals, leave those fields null and list them in `missing_metrics`.

A result can still be useful with missing metrics, but the Strategy Advisor may mark a strategy as `inconclusive` until the missing telemetry is captured.

## Publication Rule

Never publish mock metrics as real endpoint results. Mock runs validate the benchmark harness shape only.
