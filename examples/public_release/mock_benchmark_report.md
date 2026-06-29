# KVOptBench Mock Benchmark Report

This checked-in report is a deterministic public example generated from the CSV fixtures in this directory. Mock timings validate benchmark wiring only; use real endpoint runs for engine claims.

## Run Summary

- Summary source: `examples/public_release/summary.csv`
- Experiment groups: 8
- Total requests: 46
- Mean success rate: `1.000`

## Workload Summary

- `shared_prefix_long_doc` on `vllm`/`cache_on`: 8 requests
- `partial_prefix_reuse` on `vllm`/`cache_on`: 6 requests
- `prefill_decode_grid` on `vllm`/`baseline`: 6 requests
- `long_context_pressure` on `vllm`/`baseline`: 5 requests
- `long_context_pressure` on `vllm`/`kv_fp8`: 5 requests
- `long_context_pressure` on `vllm`/`kv_offload`: 5 requests
- `decode_heavy` on `vllm`/`speculative_decoding`: 5 requests
- `prefill_decode_grid` on `sglang`/`prefill_decode_disaggregation`: 6 requests

## Cache Comparison

| engine | strategy | shared cold TTFT ms | shared warm TTFT ms | random penalty ms | control-adjusted gain ms | interpretation |
|---|---|---:|---:|---:|---:|---|
| vllm | cache_on | 320.000 | 110.000 | 15.000 | 195.000 | credible_cache_reuse_signal |

## Prefix Overlap Sweep

First meaningful cache gain appears at shared-prefix ratio `0.250`.

| engine | strategy | shared prefix ratio | cache gain ms | interpretation |
|---|---|---:|---:|---|
| vllm | cache_on | 0.000 | 1.000 | no_prefix_overlap |
| vllm | cache_on | 0.250 | 45.000 | meaningful_prefix_cache_gain |
| vllm | cache_on | 0.500 | 140.000 | meaningful_prefix_cache_gain |

## Advanced Strategy Comparisons

| strategy | key signal | interpretation |
|---|---|---|
| KV quantization | 39.583% lower GPU memory and 17.241% higher output throughput | quantization_promising |
| KV offload | GPU memory telemetry unavailable | memory_telemetry_missing |
| Speculative decoding | 24.000% lower E2E latency and 33.333% higher throughput | speculative_decoding_promising |
| Prefill/decode disaggregation | 22.222% lower TTFT with no decode regression | disaggregation_promising |

## Strategy Advisor

Overall recommendation: `prefix_caching`.

| rank | strategy | decision | confidence | source |
|---:|---|---|---|---|
| 1 | `prefix_caching` | `recommend` | `high` | cache comparison CSV |
| 2 | `kv_quantization` | `recommend` | `high` | KV quantization comparison CSV |
| 3 | `speculative_decoding` | `recommend` | `medium` | speculative decoding comparison CSV |
| 4 | `prefill_decode_disaggregation` | `consider` | `high` | prefill/decode disaggregation comparison CSV |
| 5 | `kv_offload` | `inconclusive` | `low` | KV offload comparison CSV |

## Missing Metrics Warning

The following metrics were unavailable or intentionally null in this run: `gpu_memory_peak_gb`, `speculative_acceptance_rate`.

## Next Steps

- Do not treat mock metrics as real engine benchmark results.
- Reproduce this report with the commands in `guides/reproducibility.md`.
- For real endpoint runs, verify engine flags, model revision, workload hash, and missing telemetry before publishing.
