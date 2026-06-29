# Blog-Ready Benchmark Report Template

Use this after a real endpoint run has passed the result-template checklist. Do not publish mock metrics as real endpoint results.

## Headline

`KVOptBench: <workload> on <engine> with <strategy finding>`

## TL;DR

- Workload:
- Engine:
- Model:
- Main finding:
- Strategy Advisor recommendation:
- Biggest caveat:

## Setup

- Provider:
- GPU:
- Engine version:
- Model Revision:
- Launch command:
- Workload Hash:
- Config hash:

## Findings

| Question | Evidence | Interpretation |
|---|---|---|
| Did prefix caching help? |  |  |
| Was the workload prefill-bound or decode-bound? |  |  |
| Did long context create memory pressure? |  |  |
| Did KV quantization help? |  |  |
| Did KV offload help? |  |  |
| Did speculative decoding help? |  |  |
| Did prefill/decode disaggregation help? |  |  |

## Charts Or Tables

Add compact tables or charts for:

- TTFT by strategy
- TPOT or ITL by output bucket
- E2E latency by context bucket
- output tokens/sec by strategy
- cache hit rate and cache miss penalty
- quality score by strategy
- `missing_metrics`

## Strategy Advisor

- Overall recommendation:
- Why:
- Caveats:
- Next experiments:

## Quality And Caveats

- Evaluators:
- Quality regressions:
- Error rate:
- Unavailable telemetry:
- Why unavailable telemetry does or does not change the conclusion:

## Reproduction

Include the exact commands used:

```bash
kvoptbench validate-config --config path/to/config.yaml
kvoptbench endpoint-check --config path/to/config.yaml
kvoptbench run --config path/to/config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --strategy-input reports/outputs/strategy_advisor.json --output reports/outputs/real_endpoint_report.md
```

## Appendix

- Raw artifact hashes:
- Engine command:
- Environment details:
- Known limitations:
- Follow-up experiments:
