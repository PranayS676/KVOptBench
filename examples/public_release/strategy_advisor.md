# Strategy Advisor

- Generated at: `2026-06-29T00:00:00+00:00`
- Overall recommendation: `prefix_caching`

## Recommended

1. `prefix_caching` - `recommend` (confidence: `high`)
   Evidence:
   - Observed control-adjusted cache gain of 195.000 ms.
   - Meaningful cache gain appeared at shared-prefix ratio 0.250.

2. `kv_quantization` - `recommend` (confidence: `high`)
   Evidence:
   - End-to-end latency improved by 8.036%.
   - Output throughput improved by 17.241%.
   - GPU memory improved by 39.583%.
   Next experiments:
   - Repeat KV quantization across larger context buckets and quality-sensitive tasks.

3. `speculative_decoding` - `recommend` (confidence: `medium`)
   Evidence:
   - End-to-end latency improved by 24.000%.
   - Output throughput improved by 33.333%.
   Caveats:
   - Missing metrics: speculative_acceptance_rate.
   - Speculative acceptance telemetry is missing.
   Next experiments:
   - Retest with acceptance-rate telemetry and representative long-output prompts.

## Consider

4. `prefill_decode_disaggregation` - `consider` (confidence: `high`)
   Evidence:
   - TTFT improved by 22.222%.
   - End-to-end latency improved by 14.754%.
   Next experiments:
   - Retest under mixed prefill/decode traffic with production-like routing.

## Inconclusive

5. `kv_offload` - `inconclusive` (confidence: `low`)
   Evidence:
   - End-to-end latency regressed by 3.571%.
   Caveats:
   - KV offload memory telemetry is missing.
   - Missing metrics: gpu_memory_peak_gb.
   Next experiments:
   - Rerun KV offload with GPU memory telemetry enabled.
