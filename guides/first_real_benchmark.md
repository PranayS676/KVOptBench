# First Real Benchmark

This guide defines the first real endpoint result package for KVOptBench.

The goal is not to prove that one engine is always better. The goal is to produce a small,
reproducible result that shows whether cache-aware benchmarking is wired correctly against a
real OpenAI-compatible endpoint.

## Recommended Scope

Start with one reachable vLLM or SGLang endpoint.

Good first target:

- model: a model that fits comfortably on the selected GPU
- endpoint: OpenAI-compatible `/v1/chat/completions`
- workload: shared-prefix document QA plus a random-prefix control
- strategies: baseline/cache-off and cache-on
- concurrency: `1` first, then `4` and `8` if the endpoint is stable
- output: raw JSONL, summary CSV, cache comparison CSV, report, and strategy advisor

Use RunPod, Lambda Cloud, local GPU, bare metal, or another provider. The provider only needs to
expose a normal HTTP endpoint that KVOptBench can reach.

## Preflight

Before spending GPU time, run the local validation path:

```bash
python -m pytest
kvoptbench validate-config --config examples/example_experiment_config.yaml
```

Generate a small smoke workload:

```bash
kvoptbench generate-workload \
  --profile shared_prefix \
  --count 5 \
  --out workloads/generated/shared_prefix_smoke.jsonl
```

Update the real endpoint config with:

- `provider`
- `engine`
- `endpoint_type`
- `base_url`
- `model_id`
- `api_key_env`, if authentication is required
- `workload_file`
- `output_file`

## Smoke Run

Check the endpoint before sending benchmark traffic:

```bash
kvoptbench validate-config --config examples/vllm_openai_compatible_config.yaml
kvoptbench endpoint-check --config examples/vllm_openai_compatible_config.yaml
```

Run the benchmark:

```bash
kvoptbench run --config examples/vllm_openai_compatible_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/outputs/real_endpoint_report.md
```

If the endpoint check fails, fix server reachability, auth, `base_url`, or `model_id` before
running larger workloads.

## Cache Comparison

Generate matched shared-prefix and random-prefix workloads:

```bash
kvoptbench generate-workload \
  --profile shared_prefix \
  --out workloads/generated/shared_prefix_32k.jsonl

kvoptbench generate-workload \
  --profile random_prefix \
  --out workloads/generated/random_prefix_32k.jsonl
```

Write and run a cache experiment plan:

```bash
kvoptbench cache-plan \
  --plan-dir configs/cache_plan \
  --experiment-prefix first_cache_run \
  --provider local \
  --engine vllm \
  --model-id your/model \
  --base-url http://127.0.0.1:8000/v1 \
  --shared-workload-file workloads/generated/shared_prefix_32k.jsonl \
  --random-workload-file workloads/generated/random_prefix_32k.jsonl \
  --output-dir results/raw

kvoptbench cache-run --plan-dir configs/cache_plan
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench cache-compare --input results/raw --output results/cache_summary.csv
kvoptbench report \
  --input results/summary.csv \
  --cache-input results/cache_summary.csv \
  --output reports/outputs/cache_report.md
```

For SGLang, set `--engine sglang` and use the SGLang endpoint URL. For RunPod or Lambda Cloud,
set `--provider` accordingly and use the reachable proxy or instance URL.

## Strategy Advisor

Once comparison CSVs exist, generate advisor outputs:

```bash
kvoptbench strategy-recommend \
  --summary results/summary.csv \
  --cache-input results/cache_summary.csv \
  --json-output reports/outputs/strategy_advisor.json \
  --markdown-output reports/outputs/strategy_advisor.md
```

The advisor should recommend only from observed evidence. If memory, cache-hit, or engine-internal
telemetry is unavailable, it should report the missing data instead of inferring it.

## Result Package

Before publishing, fill `examples/public_release/result_template.md` with:

- commit hash
- provider
- GPU type and count
- model id and revision
- engine and engine version
- exact server command
- exact config path and hash
- workload file path and hash
- raw JSONL result path
- summary and comparison CSV paths
- report path
- `missing_metrics`
- known failures or caveats

Do not publish mock metrics as real endpoint results. Do not publish secrets, private endpoint
credentials, private prompts, or private model outputs.
