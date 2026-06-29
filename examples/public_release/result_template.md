# Public Benchmark Result Template

Use this template for real endpoint publications. Do not publish mock metrics as real endpoint results.

## Run Identity

- Run name:
- Date:
- Operator:
- Repository commit:
- Official run: yes/no
- Provider: local / RunPod / other
- Endpoint type: vLLM / SGLang / OpenAI-compatible

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
- Input token buckets:
- Output token buckets:
- Concurrency:
- Request rate:
- Max tasks:
- Streamed responses: yes/no

## Artifacts

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

## Metrics Summary

- TTFT:
- TPOT:
- ITL:
- E2E latency:
- Requests/sec:
- Output tokens/sec:
- Cache hit rate:
- Cache miss penalty:
- GPU memory peak:
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
- Do not publish credentials, private endpoint URLs, private model paths, or private user data.
