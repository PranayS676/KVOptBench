# LMCache and SCBench Extension Architecture

## Purpose

KVOptBench can evaluate systems that expose cache and lifecycle telemetry, but
it should not become a KV cache implementation or an inference engine. This
document defines an implementation-facing architecture for LMCache telemetry
imports and SCBench-inspired workload modes while preserving KVOptBench's role
as an experiment and decision layer.

The extension should let KVOptBench:

- import or observe LMCache-related telemetry when a backend exposes it
- record cache hit, load, store, offload, and transfer metrics with provenance
- model lifecycle-oriented workloads inspired by SCBench
- express lifecycle behavior above endpoints through request sequences,
  evaluators, and metric requirements
- keep missing metrics explicit instead of fabricating backend details

## Design Principles

- Observe and import telemetry; do not manage KV cache internals.
- KVOptBench does not implement cache storage, eviction, offload, transfer, or
  serving lifecycle control.
- Keep adapters narrow and optional.
- Treat metric availability as backend-dependent.
- Represent lifecycle behavior as workload and evaluation modes above
  endpoints.
- Preserve public-safe output: no secrets, private endpoint details, or local
  machine paths.
- Make every imported metric traceable to a run, request, backend source, and
  adapter version.

## Non-Goals

- Implementing KV cache allocation, eviction, compression, offload, loading, or
  transfer logic.
- Starting, stopping, tuning, or reconfiguring LMCache or model servers.
- Claiming that KVOptBench is an SCBench implementation or a drop-in substitute.
- Requiring every backend to expose cache telemetry.
- Inferring hidden KV cache behavior from latency alone.
- Mutating endpoint state to force cache hits, evictions, or transfer behavior.

## LMCache Telemetry Contract

The adapter should normalize available LMCache-related telemetry into a stable
contract. Fields can be absent when the backend does not expose them; absence
must be represented in `missing_metric_ids`.

```yaml
lmcache_telemetry:
  telemetry_source:
    source_type: observed | imported
    adapter_id: string
    adapter_version: string
    backend_label: string
  scope:
    run_id: string
    request_id: optional string
    candidate_id: string
    workload_profile: string
  cache:
    cache_lookup_count: optional integer
    cache_hit_count: optional integer
    cache_miss_count: optional integer
    cache_hit_rate: optional number
    cache_key_reuse_count: optional integer
  load_store:
    cache_load_count: optional integer
    cache_load_ms: optional number
    cache_load_bytes: optional integer
    cache_store_count: optional integer
    cache_store_ms: optional number
    cache_store_bytes: optional integer
  offload:
    offload_count: optional integer
    offload_ms: optional number
    offload_bytes: optional integer
    reload_count: optional integer
    reload_ms: optional number
    reload_bytes: optional integer
  transfer:
    transfer_count: optional integer
    transfer_ms: optional number
    transfer_bytes: optional integer
    transfer_source: optional string
    transfer_destination: optional string
  reliability:
    cache_error_count: optional integer
    cache_timeout_count: optional integer
    cache_warning_count: optional integer
  missing_metric_ids:
    - string
```

The contract should support both request-level and run-level metrics. When only
aggregate run-level telemetry is available, the adapter should mark request
scope as absent instead of distributing aggregate values across requests.

## Adapter Boundaries

Adapters are responsible for translation and provenance, not control.

Allowed adapter responsibilities:

- read metrics emitted by an endpoint, log, trace, metrics endpoint, or result
  export
- normalize metric names and units into the telemetry contract
- attach adapter version, source type, and source timestamp when available
- mark unavailable metrics as missing
- reject ambiguous units or unsupported schemas with explicit warnings
- preserve raw metric references when public-safe

Disallowed adapter responsibilities:

- changing cache policy
- triggering cache eviction or compression directly
- moving KV blocks between devices or hosts
- assuming a cache hit occurred only because latency improved
- hiding missing metrics behind derived placeholders
- requiring privileged server access for a normal benchmark import

The adapter interface should be shaped so multiple telemetry sources can be
merged without losing provenance:

```yaml
telemetry_import_result:
  adapter_id: lmcache
  source_type: observed
  records:
    - lmcache_telemetry
  warnings:
    - code: string
      message: string
  missing_metric_ids:
    - string
```

## SCBench-Inspired Lifecycle Modes

SCBench highlights that KV cache behavior is workload-lifecycle dependent.
KVOptBench can adopt that insight as workload modes without implementing
SCBench internals.

Supported lifecycle concepts:

- `kv_generation`: create a reusable context or prefix and measure the cost of
  generating cacheable state.
- `compression`: evaluate workloads where the backend may expose compressed KV
  or reduced-memory behavior, using only telemetry the backend reports.
- `retrieval`: reuse prior context or prefix material and measure whether
  retrieval-sensitive workloads preserve quality and latency.
- `loading`: evaluate workloads where the backend may load or reload cached
  state and expose load-related metrics.
- `multi_turn`: model conversational reuse across turns in one logical session.
- `multi_request`: model repeated or related requests across a request group.

These are workload modes, not backend commands. KVOptBench should generate
request sequences and evaluation expectations, send them to configured
endpoints, then record observed behavior and telemetry.

## Workload Mode Contract

Lifecycle modes should be expressed above endpoints with explicit request
groups, reuse hints, metric requirements, and evaluators:

```yaml
workload_mode:
  mode_id: string
  lifecycle_pattern: kv_generation | compression | retrieval | loading | multi_turn | multi_request
  workload_profile: rag | long_context_qa | tool_calling | agentic_coding | decode_heavy
  request_group:
    group_id: string
    ordering: fixed | randomized_with_constraints
    session_boundary: single_session | multiple_sessions | request_group
    reuse_hint: none | shared_prefix | prior_turns | related_requests
  endpoint_contract:
    endpoint_group_id: string
    required_capabilities:
      - openai_compatible_chat
    optional_telemetry:
      - cache_hit_rate
      - cache_load_ms
      - cache_store_ms
      - offload_bytes
      - transfer_bytes
  evaluation:
    required_evaluators:
      - evaluator_id: string
    required_metrics:
      - latency_ms
      - error_rate
    recommended_metrics:
      - cache_hit_rate
      - cache_load_ms
      - transfer_bytes
```

The `reuse_hint` field tells KVOptBench how to group requests for evaluation. It
must not imply that KVOptBench can force a backend to reuse or expose KV state.

## Example Mode Semantics

### KV Generation

Goal: measure the cost of creating reusable context.

Expected behavior:

- send an initial long-prefix or context-building request
- record latency, token counts, error rate, and any store telemetry
- require a quality or validity evaluator appropriate to the workload
- mark cache-store metrics missing when the backend does not expose them

### Compression

Goal: compare outcomes when a backend reports compressed or reduced-memory KV
behavior.

Expected behavior:

- import compression-related telemetry only if exposed by the backend
- compare quality and latency against the same workload profile
- avoid claiming compression occurred without explicit telemetry
- treat missing compression metrics as a confidence penalty, not a failure of
  the benchmark framework

### Retrieval

Goal: evaluate reuse-sensitive workloads such as RAG or long-context QA.

Expected behavior:

- group requests that share prior context or retrieval material
- require answer-quality evaluators
- record hit, load, latency, and transfer metrics when exposed
- keep quality gates stronger than cache-hit metrics

### Loading

Goal: measure workloads where cached state may be loaded or reloaded.

Expected behavior:

- structure request groups so loading behavior can be observed if exposed
- record load count, load latency, load bytes, reload count, and reload latency
- avoid direct cache loading commands
- mark load metrics missing when telemetry is not available

### Multi-Turn and Multi-Request

Goal: represent repeated interactions where previous context may matter.

Expected behavior:

- preserve logical session or request-group identifiers
- randomize candidate order without breaking required turn order
- evaluate task success or answer quality across the whole group
- record per-request and aggregate telemetry when available

## Metric Interpretation Rules

- Cache telemetry can explain a benchmark result, but it should not replace
  quality gates.
- Cache-hit improvements should not produce a recommendation when error rate or
  quality regresses beyond the workload threshold.
- Missing cache telemetry should reduce confidence for cache-sensitive modes,
  but it should not invalidate ordinary latency and quality benchmarks.
- Imported telemetry should carry source and adapter provenance.
- Aggregate telemetry should remain aggregate unless request-level attribution
  is explicitly available.

## Public Caveats

- LMCache metric names and availability can vary by deployment and version.
- Some endpoints may expose no cache telemetry at all.
- SCBench-inspired modes are workload patterns for evaluation, not a claim of
  benchmark equivalence.
- KVOptBench can compare evidence from configured endpoints, but it cannot
  guarantee that two endpoints use identical KV cache policies.
- Cache-related recommendations are only as strong as the exposed telemetry,
  sample design, and quality evaluator coverage.

## Acceptance Criteria

- LMCache adapter output conforms to the telemetry contract and records missing
  metric identifiers explicitly.
- Adapter code can import observed or exported metrics without changing backend
  cache behavior.
- Lifecycle workload modes represent `kv_generation`, `compression`,
  `retrieval`, `loading`, `multi_turn`, and `multi_request`.
- Workload modes are expressed as request groups, reuse hints, evaluators, and
  metric requirements above endpoints.
- Advisor input can consume cache telemetry without treating absent telemetry
  as fabricated zero values.
- Public result packages can include cache metrics, missing metric notes, and
  caveats without exposing secrets or private endpoint details.
- Tests cover missing telemetry, aggregate-only telemetry, request-level
  telemetry, and ambiguous-unit rejection.

## Testing Plan

Implementation should include tests that do not require GPUs, external APIs,
model weights, LMCache, or live services.

- Fixture tests for LMCache telemetry imports with full, partial, and absent
  metric coverage.
- Unit tests for unit normalization, missing metric handling, and provenance
  preservation.
- Tests that reject ambiguous units or unsupported metric schemas with explicit
  warnings.
- Workload-mode tests for lifecycle request grouping and ordering constraints.
- Advisor integration fixtures showing how cache telemetry affects confidence
  without bypassing quality gates.
- Public-safety scans for local paths, secrets, and private endpoint details in
  packaged telemetry output.
- Regression tests proving adapters do not require backend mutation or server
  lifecycle control.

## Implementation Notes

- Keep telemetry adapters optional so ordinary KVOptBench runs remain usable
  without cache-specific dependencies.
- Prefer additive metric fields over backend-specific branching in advisor
  logic.
- Version adapter contracts so result packages can explain how telemetry was
  interpreted.
- Treat backend-specific documentation as adapter documentation, not as a core
  benchmark requirement.
