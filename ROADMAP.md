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
- Public benchmark result templates
- Blog-ready report format

## Next: Real Endpoint Result Collection

- First public vLLM real endpoint smoke run
- First public SGLang real endpoint smoke run
- Cache and prefix sweep run on a real endpoint
- Long-context pressure run with memory telemetry
- Publish one complete result package using the public templates
