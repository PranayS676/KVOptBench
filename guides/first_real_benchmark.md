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
- workload: QASPER shared-prefix document QA plus a QASPER random-prefix control
- strategies: baseline/cache-off and cache-on
- context buckets: 8K and 32K
- concurrency: `1` first, then `4` and `8` if the endpoint is stable
- passes: cold and warm
- items: 50-100 shared-prefix and 50-100 random-prefix rows
- output: raw JSONL, summary CSV, cache comparison CSV, report, and strategy advisor

Use RunPod, Lambda Cloud, local GPU, bare metal, or another provider. The provider only needs to
expose a normal HTTP endpoint that KVOptBench can reach.

Do not compare vLLM and SGLang in the first public result unless the model revision,
tokenizer, dataset manifests, prompt template, hardware class, concurrency, output settings,
request order, timeout policy, and retry policy all match. A single-engine result with
strong controls is the better first public claim.

## Preflight

Before spending GPU time, run the local validation path:

```bash
python -m pytest
kvoptbench validate-config --config examples/example_experiment_config.yaml
```

Generate a small synthetic smoke workload:

```bash
kvoptbench generate-workload \
  --profile shared_prefix \
  --count 5 \
  --out workloads/generated/shared_prefix_smoke.jsonl
```

Synthetic workloads validate the harness shape. Before publishing real endpoint
claims, prepare public dataset workloads from `guides/frontier_dataset_pack.md` and
record adapter manifests as described in `guides/dataset_adapter_contract.md`.

Install dataset download dependencies when using `--download` for Hugging Face-backed
adapters:

```bash
python -m pip install -e ".[data]"
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

For a smoke run, generate matched synthetic shared-prefix and random-prefix workloads:

```bash
kvoptbench generate-workload \
  --profile shared_prefix \
  --out workloads/generated/shared_prefix_32k.jsonl

kvoptbench generate-workload \
  --profile random_prefix \
  --out workloads/generated/random_prefix_32k.jsonl
```

For a publishable real-data run, use the QASPER shared-prefix and random-prefix packs
from `guides/frontier_dataset_pack.md` instead.

Prepare 8K workloads:

```bash
kvoptbench dataset prepare \
  --source qasper \
  --mode shared_prefix \
  --download \
  --cache-dir data/raw \
  --split validation \
  --target-input-tokens 8192 \
  --target-output-tokens 256 \
  --max-items 100 \
  --out workloads/generated/qasper_shared_prefix_8k.jsonl \
  --manifest workloads/generated/qasper_shared_prefix_8k_manifest.json

kvoptbench dataset prepare \
  --source qasper \
  --mode random_prefix \
  --download \
  --cache-dir data/raw \
  --split validation \
  --target-input-tokens 8192 \
  --target-output-tokens 256 \
  --max-items 100 \
  --out workloads/generated/qasper_random_prefix_8k.jsonl \
  --manifest workloads/generated/qasper_random_prefix_8k_manifest.json
```

Prepare 32K workloads:

```bash
kvoptbench dataset prepare \
  --source qasper \
  --mode shared_prefix \
  --download \
  --cache-dir data/raw \
  --split validation \
  --target-input-tokens 32768 \
  --target-output-tokens 256 \
  --max-items 100 \
  --out workloads/generated/qasper_shared_prefix_32k.jsonl \
  --manifest workloads/generated/qasper_shared_prefix_32k_manifest.json

kvoptbench dataset prepare \
  --source qasper \
  --mode random_prefix \
  --download \
  --cache-dir data/raw \
  --split validation \
  --target-input-tokens 32768 \
  --target-output-tokens 256 \
  --max-items 100 \
  --out workloads/generated/qasper_random_prefix_32k.jsonl \
  --manifest workloads/generated/qasper_random_prefix_32k_manifest.json
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
  --shared-workload-file workloads/generated/qasper_shared_prefix_32k.jsonl \
  --random-workload-file workloads/generated/qasper_random_prefix_32k.jsonl \
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

- `README_result.md`
- run manifest path and hash
- commit hash
- provider
- GPU type and count
- model id and revision
- engine and engine version
- exact server command
- exact config path and hash
- workload file path and hash
- workload sample path
- dataset manifest path and hash
- dataset source URL, split, license, and adapter version
- license review status and redistribution policy
- prompt template hash
- tokenizer ID, tokenizer revision, and token count method
- truncation policy and truncation count
- raw JSONL result path
- summary and comparison CSV paths
- report path
- plots directory, if generated
- `engine_reported_cache_hit_rate`, when exposed by the backend
- `cache_hit_proxy`, when derived by KVOptBench
- `missing_metrics`
- `missing_metrics.json`
- known failures or caveats
- known limitations file

Do not publish mock metrics as real endpoint results. Do not publish secrets, private endpoint
credentials, private prompts, or private model outputs.

Failed requests, timeouts, unavailable telemetry, and quality failures should stay in the
result package. Filtering them out makes the benchmark less useful.
