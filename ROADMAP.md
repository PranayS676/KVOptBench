# Roadmap

## Done: Local Mock Harness

- Mock OpenAI-compatible server
- Workload generators
- Config-driven experiment runner
- Streaming timing metrics
- JSONL result writing
- Summary CSV generation
- Markdown report generation
- Basic quality evaluators
- Tests and CI

## Done: Real Endpoint Benchmarking

- OpenAI-compatible endpoint runner
- vLLM endpoint support
- SGLang endpoint support
- Timeout and retry handling
- Server metadata capture

## Done: Cache Experiments

- Prefix cache ablation
- Radix cache ablation
- Cold/warm cache testing
- Cache miss penalty calculation
- Shared-prefix ratio sweep

## Done: Prefill Vs Decode Decomposition

- Prefill vs decode decomposition

## Done: Long-Context Pressure

- Configurable long-context workload buckets
- Long-context experiment planning and execution
- Context-bucket latency and throughput comparison
- Pressure classification for stable, prefill-growth, throughput-degraded, failed, or insufficient-signal runs
- Markdown report integration

## Done: KV Cache Quantization

- Baseline vs quantized KV cache planning
- vLLM and SGLang `kv_fp8` strategy comparison
- Latency, throughput, quality, success-rate, and memory delta analysis
- Missing telemetry preservation for unavailable memory metrics
- Markdown report integration

## Done: KV Offload Experiment Support

- Baseline vs KV offload planning
- vLLM and SGLang `kv_offload` placeholder strategy comparison
- Latency, throughput, quality, success-rate, and memory delta analysis
- Missing telemetry preservation for unavailable memory metrics
- Markdown report integration

## Done: Advanced Inference Experiments

- Baseline vs speculative decoding planning and comparison
- Decode-heavy workload sweep support for speculative decoding
- Baseline vs prefill/decode disaggregation planning and comparison
- Prefill/decode grid comparison for disaggregation behavior
- vLLM and SGLang placeholder strategy profiles for backend-specific setups
- Markdown report integration

## Done: Strategy Advisor

- Evidence-based recommendation rules
- Ranked recommendations for prefix caching, KV quantization, KV offload, speculative decoding, and prefill/decode disaggregation
- JSON and markdown advisor outputs
- Missing-telemetry caveats and follow-up experiment suggestions
- Compatibility wrapper for basic strategy selection

## Done: Public Example Bundle

- Deterministic public CSV fixtures for benchmark summary and comparison outputs
- Checked-in mock benchmark report and strategy advisor outputs
- Fresh-clone reproducibility guide
- Combined report support for embedded strategy advisor output

## Done: Public Research Release Guides

- RunPod runbook
- vLLM/SGLang engine guides
- First real benchmark guide
- Provider-neutral endpoint example configs
- Public dataset selection guide
- Dataset adapter contract reference
- Frontier dataset pack recipe
- Public benchmark result templates
- Blog-ready report format

## Done: Public Dataset Preparation

- Dataset preparation CLI
- Workload and dataset manifest schemas
- QASPER shared-prefix and random-prefix adapter
- QASPER partial-prefix sweep adapter
- Project Gutenberg long-context needle adapter
- Project Gutenberg no-needle, multi-needle, and conflicting-needle controls
- LongBench long-context adapter
- BEIR SciFact RAG adapter
- BFCL tool-calling adapter
- Optional download/cache support for public sources
- Dataset manifest writer with workload hashes
- Tiny adapter fixtures that do not require network access
- Offline adapter tests and gated real-download smoke test
- Documentation tests for dataset sources, manifests, and adapter contract fields

## Done: Result Package Generation

- `kvoptbench result-package` command for completed benchmark artifacts
- Package manifest with package-relative paths, hashes, summary metadata, and artifact inventory
- Missing-metric JSON with explicit unavailable-telemetry explanations
- Workload samples and dataset manifest copies for provenance review
- Redacted config snapshots to avoid publishing endpoint URLs or secret-bearing fields
- README and reproducibility guide updates for local package generation

## Done: Benchmark Methodology Documentation

- Public benchmark validity guide for mock, synthetic, exploratory, and publishable results
- Public metric provenance guide for client, provider, engine, GPU, imported, derived, and estimated metrics
- First real benchmark guidance for randomized condition order, repeated trials, and confidence intervals
- Real endpoint telemetry expectations for Prometheus and GPU memory capture
- Result template fields for run order, repetitions, effect size, advisor confidence, and metric provenance
- Public architecture notes for live telemetry, environment capture, import adapters, strategy plan/run,
  workload-aware advisor confidence, and LMCache/SCBench-style extensions

## Done: Non-GPU Evidence Foundations

- Request-level metric provenance schema and runner population
- Environment snapshots for reproducible result rows
- Summary and report sections for metric source types and unavailable metric reasons
- Result package `metric_provenance.json`
- Deterministic repeated-run scheduling helpers
- Repeated-trial statistical aggregation and comparison helpers
- Offline Prometheus, DCGM, and `nvidia-smi` telemetry parsing foundations
- vLLM bench import foundations
- Strategy advisor confidence rationale, quality guardrails, and prioritized next experiments

## Next: Real Endpoint Result Collection

- First public cache run on one real vLLM or SGLang endpoint
- QASPER shared-prefix vs random-prefix controls at 8K and 32K
- Concurrency 1, 4, and 8 with cold and warm passes
- Cache and partial-prefix sweep run on a real endpoint
- Long-context pressure run with memory telemetry
- RAG and tool-calling smoke runs on BEIR SciFact and BFCL
- Publish one complete result package using the public templates

## Next: Public Result Packaging

- Optional plot generation for packaged benchmark outputs
- Example real endpoint package layout
- Stronger RAG citation evaluator
- Stronger BFCL tool-call evaluator
- Dataset download provenance snapshots for published runs
