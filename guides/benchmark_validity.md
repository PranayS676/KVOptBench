# Benchmark Validity

KVOptBench is useful only when its results are honest about what was measured,
what was estimated, and what was unavailable. This guide defines the public
validity rules for interpreting KVOptBench outputs.

## What KVOptBench Can Claim

KVOptBench can make claims about a completed run when the package includes the
workload, config, endpoint metadata, raw results, summaries, reports, and
missing-metric notes needed to audit the run.

Valid claims include:

- client-observed latency and throughput for the configured endpoint
- quality results from the configured evaluators
- cache miss penalty derived from matched cold and warm controls
- cache-hit proxy values derived from workload structure and observed timing
- strategy recommendations based on packaged evidence and documented caveats
- whether a result is official or exploratory

The strongest KVOptBench claims come from real endpoint runs with public dataset
workloads, random-prefix controls, randomized condition order, repeated trials,
and confidence intervals.

## What KVOptBench Refuses To Claim

KVOptBench does not claim that a backend internal metric exists unless the
backend reports it or an explicit telemetry adapter captures it.

KVOptBench refuses to claim:

- mock latency is real serving latency
- synthetic-only smoke tests prove production performance
- estimated token counts equal tokenizer-native or provider-reported counts
- `cache_hit_proxy` is the same as engine-reported cache hit rate
- missing GPU memory or cache telemetry is zero
- one engine is better than another when model revision, workload, hardware,
  request order, sampling parameters, timeout policy, and retry policy differ
- a speedup is useful when quality, tool-call correctness, or reliability drops

## Exploratory vs Publishable Results

Label a run as exploratory when any major validity requirement is missing.

Exploratory results include:

- local mock runs
- debug runs with very small workloads
- single-pass runs without repeated trials
- synthetic workloads without public dataset manifests
- real endpoint runs missing key environment metadata
- comparisons without random-prefix controls
- runs with unavailable cache or GPU telemetry that materially affects the
  strategy decision

Publishable results should include:

- public dataset workload manifests and workload hashes
- matched shared-prefix and random-prefix controls for prefix-cache claims
- cold and warm passes
- randomized condition order with the seed recorded
- repeated trials with run count recorded
- p50 and p95 summary metrics
- confidence intervals or an explicit note that they are not yet available
- effect size or practical delta for each recommendation
- failed requests, timeouts, and quality failures preserved in the result
  package
- environment capture: model revision, engine version, launch command, GPU
  type/count, runtime image, CUDA/driver details when available

## Mock And Synthetic Result Rules

Mock results validate the pipeline. They are useful for testing config loading,
workload generation, request execution, result writing, summaries, reports, and
result-package generation.

Mock results validate the pipeline, not real inference performance.

Synthetic workloads are smoke tests. They can exercise cache, decode, RAG,
tool-calling, and long-context code paths, but public performance claims should
use dataset-backed workloads with manifests and hashes.

## Controls And Ordering

Prefix-cache claims require random-prefix controls. Warm runs can improve for
reasons unrelated to prefix reuse, so a matched random-prefix control is needed
to separate cache benefit from normal warmup, queue, or thermal effects.

Public runs should use randomized condition order. A simple sequential order can
bias results because cache state, GPU temperature, queue depth, and background
load can change over time.

Public runs should use repeated trials. Single runs are fragile. Repetitions let
the report show run count, p50, p95, confidence intervals, and effect size.

## Failed Runs And Missing Metrics

Failed requests stay in the result package. Filtering failures out can make a
strategy look better than it is.

Unavailable telemetry stays unavailable. Store nulls and list the metric in
`missing_metrics`. A missing metric is not zero.

If a recommendation depends on a missing metric, the advisor should reduce
confidence or mark the result inconclusive until the missing evidence is
captured.

## Quality Gates

Latency and throughput are not enough. A strategy should not be recommended when
quality, JSON validity, tool-call correctness, RAG grounding, or long-context
answer correctness regresses beyond the configured tolerance.

When evaluator coverage is incomplete, label the recommendation as exploratory
or lower confidence.

## Result Package Standard

Every publishable result should include a `kvoptbench result-package` output
with:

- `run_manifest.json`
- `missing_metrics.json`
- `README_result.md`
- raw JSONL samples
- workload samples
- dataset manifest copies
- redacted config snapshots
- summary and comparison CSVs
- report and strategy-advisor outputs
- known limitations
- reproduction commands

Use the result package as the unit of review, not a single CSV or screenshot.
