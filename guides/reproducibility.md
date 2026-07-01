# Reproducibility Guide

This guide shows how to reproduce the public example bundle from a fresh clone without RunPod, a GPU, model weights, or external API credentials.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## Public Example Bundle

The deterministic public example lives in `examples/public_release/`. It contains small CSV fixtures for:

- summary results
- cache comparison
- prefix-overlap sweep
- prefill/decode decomposition
- long-context pressure
- KV quantization
- KV offload
- speculative decoding
- prefill/decode disaggregation

The checked-in example outputs are:

- `examples/public_release/mock_benchmark_report.md`
- `examples/public_release/strategy_advisor.json`
- `examples/public_release/strategy_advisor.md`

Regenerate the strategy advisor from the fixture CSVs:

```bash
kvoptbench strategy-recommend \
  --summary examples/public_release/summary.csv \
  --cache-input examples/public_release/cache_summary.csv \
  --prefix-sweep-input examples/public_release/prefix_sweep.csv \
  --prefill-decode-input examples/public_release/prefill_decode.csv \
  --long-context-input examples/public_release/long_context.csv \
  --kv-quant-input examples/public_release/kv_quantization.csv \
  --kv-offload-input examples/public_release/kv_offload.csv \
  --spec-decoding-input examples/public_release/speculative_decoding.csv \
  --disagg-input examples/public_release/disaggregation.csv \
  --json-output reports/outputs/strategy_advisor.json \
  --markdown-output reports/outputs/strategy_advisor.md
```

Regenerate the combined benchmark report with the advisor section embedded:

```bash
kvoptbench report \
  --input examples/public_release/summary.csv \
  --cache-input examples/public_release/cache_summary.csv \
  --prefix-sweep-input examples/public_release/prefix_sweep.csv \
  --prefill-decode-input examples/public_release/prefill_decode.csv \
  --long-context-input examples/public_release/long_context.csv \
  --kv-quant-input examples/public_release/kv_quantization.csv \
  --kv-offload-input examples/public_release/kv_offload.csv \
  --spec-decoding-input examples/public_release/speculative_decoding.csv \
  --disagg-input examples/public_release/disaggregation.csv \
  --strategy-input reports/outputs/strategy_advisor.json \
  --output reports/outputs/mock_benchmark_report.md
```

Build a local result package from the generated report and fixture CSVs:

```bash
kvoptbench result-package \
  --summary examples/public_release/summary.csv \
  --report reports/outputs/mock_benchmark_report.md \
  --artifact examples/public_release/cache_summary.csv \
  --artifact examples/public_release/prefix_sweep.csv \
  --artifact examples/public_release/prefill_decode.csv \
  --artifact examples/public_release/long_context.csv \
  --artifact examples/public_release/kv_quantization.csv \
  --artifact examples/public_release/kv_offload.csv \
  --artifact examples/public_release/speculative_decoding.csv \
  --artifact examples/public_release/disaggregation.csv \
  --artifact reports/outputs/strategy_advisor.json \
  --output-dir results/packages/mock_public_example
```

The package contains `run_manifest.json`, `missing_metrics.json`, `README_result.md`,
artifact hashes, and copied inputs. It is local generated output and should not be committed.

## Interpreting The Example

The example is intentionally a mock release artifact. Treat it as proof that the workload, comparison, reporting, and advisor pipeline is wired correctly. Do not treat mock metrics as real vLLM, SGLang, LMCache, Mooncake, or llm-d performance claims.

Unavailable telemetry is represented through `missing_metrics`. A missing metric is not zero. For example, if `gpu_memory_peak_gb` is missing, the advisor may mark KV offload as `inconclusive` even when latency and quality fields are present.

## Real Endpoint Reproducibility Checklist

Before publishing real endpoint results, record:

- engine and engine version
- model id and model revision
- exact engine launch flags
- GPU type and GPU count
- endpoint URL shape, without secrets
- workload file hash
- config file hash
- summary CSV and comparison CSV hashes
- `missing_metrics` values
- whether the run is official or exploratory
- metric provenance for measured, reported, imported, derived, and estimated values
- randomized condition order and seed
- repeated trials and run count
- confidence intervals or a documented reason they are unavailable
- effect size for each strategy comparison

## Official Or Exploratory Results

Every public package should state whether the run is official or exploratory.

Official results should include public dataset manifests, random-prefix controls,
environment capture, metric provenance, randomized condition order, repeated trials,
quality results, failed requests, and missing telemetry notes.

Exploratory results are still useful, but they should be labeled clearly. Examples include
mock runs, synthetic smoke tests, single-run endpoint checks, runs without repeated trials,
or runs where important engine or GPU telemetry is missing.

Failed requests, timeouts, unavailable telemetry, and quality failures should stay in the
result package. Removing them makes the run harder to reproduce and can make a strategy
look safer than it is.

RunPod is not required for the public example bundle. Use `guides/first_real_benchmark.md`
when you are ready to collect the first real endpoint result package. Use `guides/runpod.md`
only when the selected endpoint is hosted on RunPod.
