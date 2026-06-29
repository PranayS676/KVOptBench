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

## Milestone 1 Quickstart

Milestone 1 runs entirely locally. It does not require RunPod, a GPU, model weights, or any external API.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

### Run the mock harness

In one terminal, start the mock OpenAI-compatible server:

```bash
python -m kvoptbench.mock_server --port 8000
```

In another terminal, generate a workload and run the example experiment:

```bash
python -m kvoptbench.workloads.generate --profile shared_prefix --out workloads/generated/shared_prefix_32k.jsonl
python -m kvoptbench.runner --config examples/example_experiment_config.yaml
python -m kvoptbench.analysis.summarize --input results/raw --output results/summary.csv
python -m kvoptbench.reports.generate --input results/summary.csv --output reports/mock_report.md
```

The local run writes:

- request-level JSONL results under `results/raw/`
- a summary CSV at `results/summary.csv`
- a markdown report at `reports/mock_report.md`

Generated workloads, raw results, summaries, and reports are ignored by git by default.

### CLI shortcuts

The installed console command exposes the same workflow:

```bash
kvoptbench validate-config --config examples/example_experiment_config.yaml
kvoptbench generate-workload --profile shared_prefix --out workloads/generated/shared_prefix_32k.jsonl
kvoptbench run --config examples/example_experiment_config.yaml
kvoptbench summarize --input results/raw --output results/summary.csv
kvoptbench report --input results/summary.csv --output reports/mock_report.md
```

## License

MIT License.
