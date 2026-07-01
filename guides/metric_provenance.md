# Metric Provenance

Metric provenance explains where each value came from and how much confidence a
reader should place in it. KVOptBench should keep measured, reported, imported,
derived, and estimated values separate.

## Source Types

Use these source types when documenting or implementing metric provenance:

| source_type | Meaning | Examples |
|---|---|---|
| `client_observed` | Measured by the KVOptBench client around request execution. | TTFT for streaming responses, E2E latency, client-side request success. |
| `provider_reported` | Returned by the OpenAI-compatible endpoint response. | `provider_completion_tokens`, finish reason, provider usage fields. |
| `engine_reported` | Exposed by serving engine metrics or logs. | `engine_reported_cache_hit_rate`, queue time, speculative acceptance rate. |
| `gpu_reported` | Captured from GPU telemetry tools. | `gpu_memory_peak_gb`, utilization, power draw from `nvidia-smi` or DCGM. |
| `imported` | Loaded from another benchmark or profiler output. | vLLM bench summaries, GenAI-Perf or AIPerf exports. |
| `derived` | Computed from other observed or reported fields. | cache miss penalty, cache-on vs cache-off speedup, quality-adjusted latency. |
| `estimated` | Approximated by KVOptBench when exact backend values are unavailable. | character-based token estimates, KV memory estimate. |

Every important metric should eventually expose:

- `source_type`
- `measurement_method`
- `source_name`
- `source_version`, when available
- `available`
- `missing_reason`, when unavailable
- `confidence`, when used by the advisor

Use `metric_provenance` in result packages and reports to group this information.

## Current Result Fields

Request-level JSONL rows include:

- `metric_provenance`: per-metric source labels, measurement methods, availability,
  units, provider fields when applicable, and missing reasons.
- `telemetry_run_id`, `telemetry_summary_path`, and `telemetry_snapshots_path` when
  live telemetry is enabled.
- `environment`: a reproducibility snapshot with Python version, platform, KVOptBench
  version, git commit, branch, dirty-state flag, and selected package versions.

Summary CSVs include compact provenance columns:

- `metric_provenance`: grouped JSON summary by metric.
- `metric_source_types`: readable `metric:source_type` pairs.
- `unavailable_metric_reasons`: readable missing-reason pairs.

Result packages include `metric_provenance.json` in addition to
`missing_metrics.json`. Use `missing_metrics.json` to see what was unavailable,
and use `metric_provenance.json` to see whether available values were observed,
provider-reported, engine-reported, GPU-reported, imported, derived, or estimated.

## Token Count Provenance

Token counts are especially easy to misread. KVOptBench should keep these fields
separate:

- `estimated_input_tokens`: estimated by KVOptBench when tokenizer-native counts
  are unavailable.
- `estimated_output_tokens`: estimated visible answer tokens from response text.
- `provider_completion_tokens`: completion token count reported by the endpoint,
  when available.
- `reasoning_tokens`: reasoning-token estimate or provider-reported reasoning
  count when the endpoint exposes it.

Do not treat estimated token counts as equivalent to provider or tokenizer
counts. Reports should state the token count method.

## Timing Provenance

Client-observed timing is valuable, but it is not the same as engine-internal
scheduling telemetry.

| Metric | Preferred source | Caveat |
|---|---|---|
| TTFT | `client_observed` from streaming responses | Non-streaming endpoints may not expose true first-token timing. |
| TPOT | `client_observed` or derived from streamed token intervals | Token estimation affects TPOT when provider token counts are missing. |
| ITL | `client_observed` streaming intervals | Not always available for non-streaming responses. |
| E2E latency | `client_observed` | Includes network and client overhead. |
| Queue time | `engine_reported` | Requires backend telemetry. |

If TTFT, TPOT, or ITL cannot be measured for a response mode, keep the field null
or label it as estimated. Do not fill it with E2E latency.

## Cache Metric Provenance

Cache behavior needs explicit source labels.

| Field | Source type | Meaning |
|---|---|---|
| `engine_reported_cache_hit_rate` | `engine_reported` | Cache hit rate exposed by vLLM, SGLang, LMCache, or another backend. |
| `cache_hit_proxy` | `derived` | KVOptBench-derived signal based on workload structure and observed timing. |
| `cache_miss_penalty_ms` | `derived` | Difference between matched cold and warm TTFT. |
| `shared_prefix_tokens` | `estimated` or dataset-derived | Workload-level prefix reuse opportunity. |

`cache_hit_proxy` must not be presented as an engine-reported cache metric.

## GPU And Memory Provenance

GPU and memory metrics should identify the tool that captured them.

Common sources:

- `nvidia-smi`
- DCGM
- Prometheus exporter
- engine metrics endpoint
- cloud provider metrics

If GPU memory telemetry is missing, keep `gpu_memory_peak_gb` null and list
`gpu_memory_peak_gb` in `missing_metrics`. This is especially important for KV
quantization and KV offload decisions.

Live telemetry collection writes `telemetry_snapshots.jsonl` and
`telemetry_summary.json` under `results/telemetry/<run_id>/` by default. Request
rows reference these files and copy run-level metrics such as `gpu_memory_peak_gb`
into the normal metric fields when available. If a sampler times out, is missing,
or does not expose an expected metric, keep the metric null and preserve the
missing reason.

## LMCache Telemetry

LMCache telemetry is treated as backend evidence, not as KVOptBench-managed cache
state. The first supported formats are:

- Prometheus-compatible text endpoints.
- structured JSON or JSONL metric exports.

Normalized LMCache fields include:

- `lmcache_cache_hits`
- `lmcache_cache_misses`
- `lmcache_cache_hit_rate`
- `lmcache_cache_loads`
- `lmcache_cache_stores`
- `lmcache_kv_transfer_bytes`
- `lmcache_kv_transfer_ms`
- `lmcache_offload_ms`
- `lmcache_load_ms`

Unknown structured metrics should be preserved as telemetry records. Free-form log
parsing should wait until there is a stable documented format.

## Imported Metrics

KVOptBench may eventually import results from tools such as vLLM bench,
GenAI-Perf, AIPerf, or other profiler outputs. Imported metrics should preserve:

- original tool name
- original tool version
- original field name
- normalized KVOptBench field name
- units
- lost information or mapping caveats

Imported evidence should be usable by summaries and the strategy advisor, but it
should remain auditable.

KVOptBench includes an initial vLLM bench importer that maps local JSON, JSONL,
or CSV artifacts into KVOptBench-like rows. Imported rows preserve unavailable
fields as null, list missing metrics, and keep the original field names in
metadata without embedding absolute local paths.

## Advisor Confidence

Strategy recommendations should use provenance. Confidence should fall when:

- a metric is estimated instead of measured
- key cache, GPU, or queue telemetry is missing
- sample count is low
- condition order was not randomized
- repeated trials were not run
- quality evaluator coverage is missing

The advisor should prefer an inconclusive recommendation over a confident answer
from weak evidence.
