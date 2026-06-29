# KVOptBench

KVOptBench is a cache-aware frontier LLM inference benchmark and strategy optimizer.

It does **not** replace inference engines such as vLLM, SGLang, LMCache, Mooncake, or llm-d. Instead, it sits above them as an experiment and decision layer.

KVOptBench helps users measure and understand:

- KV cache behavior
- prefix caching
- radix caching
- cache hit rate
- cache miss penalty
- prefill vs decode bottlenecks
- long-context pressure
- KV cache quantization
- KV offload / hierarchical cache
- speculative decoding
- prefill/decode disaggregation
- quality vs latency tradeoffs
- automatic inference strategy selection

## Why KVOptBench exists

Most inference benchmarks report basic metrics such as requests/sec, tokens/sec, TTFT, and p95 latency. Those metrics are useful, but they do not explain **why** a model or engine behaves the way it does.

KVOptBench is designed to answer deeper questions:

- When does prefix caching actually help?
- How expensive is a cache miss?
- Is this workload prefill-bound or decode-bound?
- Does KV quantization improve capacity without hurting quality?
- Does KV offload help or just move the bottleneck to CPU/PCIe?
- Does speculative decoding improve long-output workloads?
- Does prefill/decode disaggregation improve mixed traffic?
- Which inference strategy should I choose for my workload?

## Status

Early project. The local/mock benchmark harness, real OpenAI-compatible endpoint runner, engine profiles, cache experiment planning, cache comparison reporting, prefix-overlap sweep analysis, prefill/decode decomposition, long-context pressure analysis, KV cache quantization comparison, KV offload experiment support, speculative decoding sweep support, and prefill/decode disaggregation experiment support are in place.

## Local Mock Quickstart

The local mock path runs entirely on a developer machine. It does not require RunPod, a GPU, model weights, or any external API.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

### Run the mock harness

In one terminal, start the mock OpenAI-compatible server:

```bash
python -m kvoptbench.mock_server --port 8000
```

In another terminal, generate a workload and run the example experiment:

```bash
python -m kvoptbench.workloads.generate --profile shared_prefix --out workloads/generated/shared_prefix_32k.jsonl
python -m kvoptbench.runner --config examples/example_experiment_config.yaml
python -m kvoptbench.analysis.summarize --input results/raw --output results/summary.csv
python -m kvoptbench.reports.generate --input results/summary.csv --output reports/mock_report.md
```

The local run writes:

- request-level JSONL results under `results/raw/`
- a summary CSV at `results/summary.csv`
- a markdown report at `reports/mock_report.md`

Generated workloads, raw results, summaries, and reports are ignored by git by default.

### CLI shortcuts

The installed console command exposes the same workflow:

```bash
kvoptbench validate-config --config examples/example_experiment_config.yaml
kvoptbench generate-workload --profile shared_prefix --out workloads/generated/shared_prefix_32k.jsonl
kvoptbench run --config examples/example_experiment_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/mock_report.md
```

## Real Endpoint Mode

Real endpoint mode can run against an existing OpenAI-compatible endpoint, including vLLM and SGLang servers that are already running. KVOptBench does not start, deploy, or manage those servers.

Public-safe example configs are included:

- `examples/vllm_openai_compatible_config.yaml`
- `examples/sglang_openai_compatible_config.yaml`

Before running, edit the example config for your local endpoint:

- `base_url`
- `model_id`
- `api_key_env`, only if the endpoint requires auth
- `workload_file`
- `output_file`

Then validate and check the endpoint:

```bash
kvoptbench validate-config --config examples/vllm_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/vllm_openai_compatible_config.yaml
```

Run the benchmark with the same runner:

```bash
kvoptbench run --config examples/vllm_openai_compatible_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/real_endpoint_report.md
```

Real endpoint mode records unavailable engine/GPU metrics as `null` and lists them in `missing_metrics`. Do not treat missing telemetry as zero.

## Engine Profiles And Cache Planning

KVOptBench includes command previews for vLLM and SGLang strategies so users can document how a server should be started without letting the benchmark harness manage the server process.

```bash
kvoptbench engine-command --engine vllm --strategy cache_on --model-id your/model
kvoptbench engine-command --engine sglang --strategy cache_on --model-id your/model
```

Cache experiment helpers build a cold/warm ablation matrix with shared-prefix workloads and random-prefix controls. The benchmark still runs through normal YAML configs and records unavailable engine telemetry as missing rather than inventing metrics.

```bash
kvoptbench generate-workload --profile shared_prefix --out workloads/generated/shared_prefix_32k.jsonl
kvoptbench generate-workload --profile random_prefix --out workloads/generated/random_prefix_32k.jsonl

kvoptbench cache-plan \
  --plan-dir configs/cache_plan \
  --experiment-prefix cache_exp \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --shared-workload-file workloads/generated/shared_prefix_32k.jsonl \
  --random-workload-file workloads/generated/random_prefix_32k.jsonl \
  --output-dir results/raw

kvoptbench cache-run --plan-dir configs/cache_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench cache-compare --input results/raw --output results/cache_summary.csv
kvoptbench report --input results/summary.csv --cache-input results/cache_summary.csv --output reports/cache_report.md
```

To test where prefix caching starts to pay off, generate a partial-prefix workload and add the prefix sweep comparison output to the report:

```bash
kvoptbench generate-workload --profile partial_prefix --count 6 --out workloads/generated/partial_prefix_sweep.jsonl

kvoptbench cache-plan \
  --plan-dir configs/prefix_sweep_plan \
  --experiment-prefix prefix_sweep \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --shared-workload-file workloads/generated/partial_prefix_sweep.jsonl \
  --random-workload-file workloads/generated/random_prefix_32k.jsonl \
  --output-dir results/raw

kvoptbench cache-run --plan-dir configs/prefix_sweep_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench prefix-sweep-compare --input results/raw --output results/prefix_sweep.csv
kvoptbench report --input results/summary.csv --prefix-sweep-input results/prefix_sweep.csv --output reports/prefix_sweep_report.md
```

## Prefill Vs Decode Decomposition

Prefill/decode helpers run a controlled input/output grid and infer bottleneck pressure from request-level timing metrics. TTFT is treated as a prefill-pressure signal, while TPOT, ITL, and output-token throughput are treated as decode-pressure signals. Missing metrics remain unavailable rather than being invented.

```bash
kvoptbench generate-workload --profile prefill_decode_grid --count 12 --out workloads/generated/prefill_decode_grid.jsonl

kvoptbench prefill-decode-plan \
  --plan-dir configs/prefill_decode_plan \
  --experiment-prefix prefill_decode \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --workload-file workloads/generated/prefill_decode_grid.jsonl \
  --output-dir results/raw

kvoptbench prefill-decode-run --plan-dir configs/prefill_decode_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench prefill-decode-compare --input results/raw --output results/prefill_decode.csv
kvoptbench report --input results/summary.csv --prefill-decode-input results/prefill_decode.csv --output reports/prefill_decode_report.md
```

## Long-Context Pressure

Long-context helpers run the same prompt shape across increasing context buckets and classify whether the observed request-level behavior is stable, prefill-latency dominated, throughput degraded, failed under pressure, or missing enough signal to classify. The default buckets are `4096`, `16384`, `32768`, `65536`, and `131072` tokens. Larger buckets such as `262144`, `524288`, or `1000000` can be supplied for endpoints that support frontier-scale context windows.

```bash
kvoptbench generate-workload --profile long_context_pressure --count 5 --out workloads/generated/long_context_pressure.jsonl

kvoptbench long-context-plan \
  --plan-dir configs/long_context_plan \
  --experiment-prefix long_context \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --workload-file workloads/generated/long_context_pressure.jsonl \
  --output-dir results/raw

kvoptbench long-context-run --plan-dir configs/long_context_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench long-context-compare --input results/raw --output results/long_context.csv
kvoptbench report --input results/summary.csv --long-context-input results/long_context.csv --output reports/long_context_report.md
```

For a larger real endpoint sweep, provide explicit buckets:

```bash
kvoptbench generate-workload \
  --profile long_context_pressure \
  --count 6 \
  --context-buckets 4096,32768,131072,262144,524288,1000000 \
  --out workloads/generated/long_context_frontier.jsonl
```

Mock timings validate harness wiring only. Use real vLLM and SGLang endpoint runs before making engine-level claims, and treat unavailable engine/GPU metrics as missing rather than zero.

## KV Cache Quantization

KV quantization helpers compare a baseline run against a quantized KV cache strategy, currently `kv_fp8`, using the same workload and endpoint shape. The default path uses `long_context_pressure` because KV cache precision tradeoffs matter most when context length creates memory pressure. KVOptBench reports latency, throughput, success rate, quality, and memory deltas when real memory telemetry is available.

```bash
kvoptbench generate-workload --profile long_context_pressure --count 5 --out workloads/generated/long_context_pressure.jsonl

kvoptbench kv-quant-plan \
  --plan-dir configs/kv_quant_plan \
  --experiment-prefix kv_quant \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --workload-file workloads/generated/long_context_pressure.jsonl \
  --output-dir results/raw

kvoptbench kv-quant-run --plan-dir configs/kv_quant_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench kv-quant-compare --input results/raw --output results/kv_quantization.csv
kvoptbench report --input results/summary.csv --kv-quant-input results/kv_quantization.csv --output reports/kv_quantization_report.md
```

Mock runs validate the comparison shape only. For real vLLM and SGLang runs, verify the engine flags, model support, and any reported GPU/KV memory telemetry before interpreting capacity benefits. Missing memory telemetry remains unavailable rather than being treated as zero.

## KV Offload

KV offload helpers compare a baseline run against a `kv_offload` strategy using the same long-context workload shape. This is benchmark-layer support only: KVOptBench writes configs, runs the endpoint, compares latency/throughput/quality/memory deltas, and preserves missing telemetry. It does not implement offload or manage backend server lifecycle.

The built-in vLLM and SGLang `kv_offload` engine profiles are placeholders. Replace `<kv-offload-flags>` with flags supported by the installed backend version before official real-endpoint runs.

```bash
kvoptbench generate-workload --profile long_context_pressure --count 5 --out workloads/generated/long_context_pressure.jsonl

kvoptbench kv-offload-plan \
  --plan-dir configs/kv_offload_plan \
  --experiment-prefix kv_offload \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --workload-file workloads/generated/long_context_pressure.jsonl \
  --output-dir results/raw

kvoptbench kv-offload-run --plan-dir configs/kv_offload_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench kv-offload-compare --input results/raw --output results/kv_offload.csv
kvoptbench report --input results/summary.csv --kv-offload-input results/kv_offload.csv --output reports/kv_offload_report.md
```

Mock runs validate the comparison shape only. For real vLLM and SGLang runs, verify offload flags, model support, host/device memory telemetry, and transfer bottleneck visibility before making capacity or latency claims. Missing memory telemetry remains unavailable rather than being treated as zero.

## Speculative Decoding

Speculative decoding helpers compare a baseline run against `speculative_decoding` on decode-heavy workloads. KVOptBench measures latency, output throughput, success rate, quality, and missing backend-specific telemetry without claiming draft-model acceptance metrics unless the endpoint exposes them.

The built-in vLLM and SGLang speculative decoding profiles are placeholders. Replace `<draft-model>` and any algorithm flags with a supported backend setup before official real-endpoint runs.

```bash
kvoptbench generate-workload --profile decode_heavy --count 5 --out workloads/generated/decode_heavy.jsonl

kvoptbench spec-decoding-plan \
  --plan-dir configs/spec_decoding_plan \
  --experiment-prefix spec_decode \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --workload-file workloads/generated/decode_heavy.jsonl \
  --output-dir results/raw

kvoptbench spec-decoding-run --plan-dir configs/spec_decoding_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench spec-decoding-compare --input results/raw --output results/speculative_decoding.csv
kvoptbench report --input results/summary.csv --spec-decoding-input results/speculative_decoding.csv --output reports/speculative_decoding_report.md
```

Mock runs validate the comparison shape only. For real vLLM and SGLang runs, verify draft-model compatibility, speculative algorithm flags, and any exposed acceptance-rate telemetry before interpreting performance gains.

## Prefill/Decode Disaggregation

Prefill/decode disaggregation helpers compare a baseline run against `prefill_decode_disaggregation` on the prefill/decode grid workload. KVOptBench measures TTFT, TPOT, ITL, E2E latency, output throughput, quality, success rate, and missing telemetry by input/output bucket. It does not launch, coordinate, or implement disaggregated serving.

The built-in vLLM and SGLang disaggregation profiles are placeholders. Replace `<prefill-decode-disaggregation-flags>` with the backend-specific multi-process or disaggregated serving setup before official real-endpoint runs.

```bash
kvoptbench generate-workload --profile prefill_decode_grid --count 12 --out workloads/generated/prefill_decode_grid.jsonl

kvoptbench disagg-plan \
  --plan-dir configs/disagg_plan \
  --experiment-prefix disagg \
  --provider mock \
  --engine vllm \
  --model-id mock-frontier-model \
  --base-url http://127.0.0.1:8000/v1 \
  --workload-file workloads/generated/prefill_decode_grid.jsonl \
  --output-dir results/raw

kvoptbench disagg-run --plan-dir configs/disagg_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench disagg-compare --input results/raw --output results/disaggregation.csv
kvoptbench report --input results/summary.csv --disagg-input results/disaggregation.csv --output reports/disaggregation_report.md
```

Mock runs validate the comparison shape only. For real vLLM and SGLang runs, verify the disaggregated deployment topology, routing behavior, model revision, and available backend telemetry before making architecture claims.

## License

MIT License.
