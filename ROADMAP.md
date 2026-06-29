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

## Next: Cache Experiments

- Prefix cache ablation
- Radix cache ablation
- Cold/warm cache testing
- Cache miss penalty calculation
- Shared-prefix ratio sweep

## Later: Advanced Inference Experiments

- Prefill vs decode decomposition
- Long-context pressure sweep
- KV cache quantization sweep
- KV offload placeholder/integration
- Speculative decoding sweep
- Prefill/decode disaggregation placeholder/integration

## Later: Strategy Optimizer

- Workload feature extraction
- Strategy recommendation rules
- Quality-adjusted throughput
- Pareto analysis
- Automatic strategy selector evaluation

## Later: Public Research Release

- Example reports
- Reproducibility guide
- RunPod runbook
- vLLM/SGLang engine guides
- Public benchmark result templates
- Blog-ready report format
