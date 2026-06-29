# KVOptBench

KVOptBench is a cache-aware frontier LLM inference benchmark and strategy optimizer.

It does **not** replace inference engines such as vLLM, SGLang, LMCache, Mooncake, or llm-d. Instead, it sits above them as an experiment and decision layer.

KVOptBench helps users measure and understand:

- KV cache behavior
- prefix caching
- radix caching
- cache hit rate
- cache miss penalty
- prefill vs decode bottlenecks
- long-context pressure
- KV cache quantization
- KV offload / hierarchical cache
- speculative decoding
- prefill/decode disaggregation
- quality vs latency tradeoffs
- automatic inference strategy selection

## Why KVOptBench exists

Most inference benchmarks report basic metrics such as requests/sec, tokens/sec, TTFT, and p95 latency. Those metrics are useful, but they do not explain **why** a model or engine behaves the way it does.

KVOptBench is designed to answer deeper questions:

- When does prefix caching actually help?
- How expensive is a cache miss?
- Is this workload prefill-bound or decode-bound?
- Does KV quantization improve capacity without hurting quality?
- Does KV offload help or just move the bottleneck to CPU/PCIe?
- Does speculative decoding improve long-output workloads?
- Does prefill/decode disaggregation improve mixed traffic?
- Which inference strategy should I choose for my workload?

## Status

Early project. Milestone 1 focuses on a local/mock benchmark harness. Real vLLM/SGLang/RunPod experiments come after the local harness is stable.

## License

MIT License.
