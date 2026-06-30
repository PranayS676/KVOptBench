# Public Benchmark Result Template

Use this template for real endpoint publications. Do not publish mock metrics as real endpoint results.

## Run Identity

- Run name:
- Date:
- Operator:
- KVOptBench version:
- Repository commit:
- Official run: yes/no
- Provider: local / RunPod / other
- Endpoint type: vLLM / SGLang / OpenAI-compatible
- Run manifest:
- Run manifest hash:

## Environment

- Model id:
- Model Revision:
- Engine:
- Engine version:
- Runtime image:
- GPU type:
- GPU count:
- Driver/CUDA/runtime details:
- Host region or deployment location:

## Backend Launch

- Server command:
- Strategy:
- Strategy flags:
- Endpoint base URL, redacted if needed:
- Auth method, without secret values:
- Metrics endpoint:

## Workloads

- Workload profiles:
- Workload Hash:
- Workload sample:
- Dataset source URL:
- Dataset split / revision / dump date:
- Dataset license or rights note:
- License review status:
- Redistribution policy:
- Dataset adapter:
- Dataset adapter version:
- Dataset manifest:
- Dataset manifest hash:
- Prompt template:
- Prompt template hash:
- Tokenizer id:
- Tokenizer revision:
- Token count method:
- Prefix group count:
- Truncation policy:
- Truncation count:
- Input token buckets:
- Output token buckets:
- Concurrency:
- Request rate:
- Max tasks:
- Streamed responses: yes/no
- Cold/warm pass order:
- Randomization seed:

## Artifacts

- README_result.md:
- Run manifest JSON:
- Dataset manifest shared JSON:
- Dataset manifest random JSON:
- Workload sample JSONL:
- Raw JSONL:
- Summary CSV:
- Cache comparison CSV:
- Prefix sweep CSV:
- Prefill/decode CSV:
- Long-context CSV:
- KV quantization CSV:
- KV offload CSV:
- Speculative decoding CSV:
- Disaggregation CSV:
- Strategy Advisor JSON:
- Combined report:
- Plots directory:
- Server command file:
- Config file:
- Version snapshot:
- Missing metrics JSON:
- Known limitations:

## Metrics Summary

- TTFT p50/p95/p99:
- TPOT p50/p95/p99:
- ITL:
- E2E latency p50/p95/p99:
- Requests/sec:
- Input tokens/sec:
- Output tokens/sec:
- Engine-reported cache hit rate:
- Cache hit proxy:
- Cache miss penalty:
- GPU memory peak:
- Success rate:
- Error rate:
- Quality score:
- `missing_metrics`:

## Strategy Advisor

- Overall recommendation:
- Recommended strategies:
- Strategies to consider:
- Inconclusive strategies:
- Caveats:
- Next experiments:

## Quality And Safety

- Evaluators used:
- Quality failures:
- Error rate:
- Retry behavior:
- Known unsupported metrics:
- Secrets scan result:

## Reproduction Commands

```bash
kvoptbench dataset prepare --source qasper --mode shared_prefix --download --cache-dir data/raw --split validation --target-input-tokens 32768 --target-output-tokens 256 --max-items 100 --out workloads/generated/qasper_shared_prefix_32k.jsonl --manifest workloads/generated/qasper_shared_prefix_manifest.json
kvoptbench dataset prepare --source qasper --mode random_prefix --download --cache-dir data/raw --split validation --target-input-tokens 32768 --target-output-tokens 256 --max-items 100 --out workloads/generated/qasper_random_prefix_32k.jsonl --manifest workloads/generated/qasper_random_prefix_manifest.json
kvoptbench validate-config --config path/to/config.yaml
kvoptbench endpoint-check --config path/to/config.yaml
kvoptbench run --config path/to/config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench strategy-recommend --summary results/summary.csv --json-output reports/outputs/strategy_advisor.json --markdown-output reports/outputs/strategy_advisor.md
kvoptbench report --input results/summary.csv --strategy-input reports/outputs/strategy_advisor.json --output reports/outputs/real_endpoint_report.md
```

## Publication Notes

- Do not publish mock metrics as real endpoint results.
- Do not treat missing telemetry as zero.
- Do not present `cache_hit_proxy` as an engine-reported cache metric.
- Do not publish credentials, private endpoint URLs, private model paths, or private user data.
- Do not publish real dataset results without dataset source, adapter, manifest, and hash metadata.
- Do not compare runs directly unless model revision, engine version, tokenizer, prompt template,
  dataset revision, context bucket, strategy flags, hardware, sampling parameters, request order,
  timeout policy, and retry policy match. Label any other comparison exploratory.
