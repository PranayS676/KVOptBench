# Advisor Confidence Architecture

## Purpose

KVOptBench should explain why it recommends one benchmark outcome over another.
The advisor confidence layer turns run evidence into a deterministic,
transparent recommendation record. It should make strong recommendations only
when the workload profile, metric coverage, quality evaluators, and sample
design support that conclusion.

This document defines the target architecture for:

- workload-aware decision thresholds
- quality and evaluator coverage gates
- confidence penalties for weak evidence
- next-experiment command generation for inconclusive recommendations
- acceptance criteria and testing expectations

The advisor remains an experiment and decision layer. It does not infer hidden
backend state, fabricate missing telemetry, or manage serving infrastructure.

## Design Principles

- Deterministic: the same inputs must produce the same score, reasons, penalty
  ordering, and next-experiment plan.
- Transparent: every recommendation must include the evidence used, the gates
  passed or failed, and the penalties applied.
- Conservative: incomplete telemetry, narrow evaluator coverage, low sample
  counts, and non-randomized runs reduce confidence.
- Workload-aware: the threshold for a recommendation depends on what the user
  is trying to optimize.
- Public-safe: advisor records must not include secrets, private endpoints, or
  local machine paths.

## Advisor Input Contract

The advisor should consume normalized run summaries, not raw backend internals.
The minimum input shape is:

```yaml
advisor_input:
  run_id: string
  comparison_id: string
  workload_profile: rag | long_context_qa | tool_calling | agentic_coding | decode_heavy
  candidates:
    - candidate_id: string
      role: baseline | variant
  sample_design:
    sample_count: integer
    randomized_order: boolean
    repeated_trials: integer
    imported_only: boolean
    mock_only: boolean
  metrics:
    latency_ms: optional number
    time_to_first_token_ms: optional number
    output_tokens_per_second: optional number
    total_tokens_per_second: optional number
    cost_per_1k_tokens: optional number
    error_rate: optional number
    timeout_rate: optional number
    cache_hit_rate: optional number
    cache_load_ms: optional number
    cache_store_ms: optional number
    transfer_bytes: optional number
  quality:
    evaluators:
      - evaluator_id: string
        evaluator_type: string
        coverage_ratio: number
        pass_rate: optional number
        score_delta: optional number
  provenance:
    metrics_source: observed | imported | mixed | mock
    missing_metric_ids:
      - string
    warnings:
      - string
```

Implementation can add fields, but the advisor should preserve these concepts:
profile, sample design, metrics, quality coverage, provenance, and explicit
missing metric identifiers.

## Advisor Output Contract

The advisor should emit a structured record that can be printed, packaged, and
tested with golden fixtures:

```yaml
advisor_recommendation:
  recommendation_id: string
  recommendation: prefer_baseline | prefer_variant | inconclusive | reject_run
  confidence_score: number
  confidence_band: high | medium | low | blocked
  workload_profile: string
  primary_metric: string
  quality_gate_status: pass | warn | fail
  reasons:
    - code: string
      severity: info | warn | block
      message: string
      evidence:
        metric_id: optional string
        candidate_id: optional string
        value: optional number
  penalties:
    - code: string
      points: number
      message: string
  missing_metrics:
    - metric_id: string
      impact: advisory | confidence_penalty | blocking
  next_experiment:
    needed: boolean
    reason: optional string
    command_plan: optional object
```

Reason and penalty ordering should be stable. A recommended ordering is:
blocking gate failures, primary metric deltas, quality evidence, telemetry
coverage, sample design, then secondary metrics.

## Workload-Aware Thresholds

Each workload profile should define the minimum evidence needed before the
advisor can emit `prefer_baseline` or `prefer_variant`.

| Workload profile | Primary decision focus | Required quality coverage | Required telemetry | Threshold posture |
| --- | --- | --- | --- | --- |
| `rag` | answer quality, retrieval-sensitive latency, cost | answer relevance or factuality evaluator on the compared sample set | latency, error rate, token counts, cost when available | Favor quality preservation. Latency or cost wins should not override quality regressions. |
| `long_context_qa` | long-input reliability, time to first token, end-to-end latency | answer correctness or citation/factuality evaluator on long-context samples | prompt tokens, completion tokens, latency, timeout rate | Require stronger sample coverage because prompt length variance can dominate results. |
| `tool_calling` | valid tool selection, argument correctness, latency | tool-call validity evaluator and task success evaluator | latency, error rate, retry or invalid-call rate when available | Block confident wins if tool correctness drops materially. |
| `agentic_coding` | task success, patch quality, latency, cost | task-level success evaluator, test outcome import, or human-reviewed score import | latency, cost, error rate, token counts | Treat quality evidence as mandatory. Performance-only wins are advisory at most. |
| `decode_heavy` | output throughput, latency stability, error rate | lightweight output validity evaluator or task pass check | output tokens per second, latency, error rate, token counts | Throughput can be primary only when error rate and quality gates pass. |

Profile thresholds should be data-driven configuration, not hard-coded branch
logic. A future implementation can represent them as:

```yaml
workload_threshold:
  workload_profile: rag
  minimum_samples: 30
  minimum_repeated_trials: 2
  required_quality_evaluators:
    - answer_relevance
  blocking_quality_regression: true
  required_metrics:
    - latency_ms
    - error_rate
    - total_tokens_per_second
  recommended_metrics:
    - cost_per_1k_tokens
    - cache_hit_rate
  high_confidence_min_score: 85
  medium_confidence_min_score: 65
```

The exact numeric defaults should be implemented in versioned advisor
configuration and covered by fixtures.

## Quality Gates

Quality gates determine whether the advisor is allowed to recommend a winner.
They are separate from confidence penalties because some quality failures should
block a recommendation outright.

Required gate states:

- `pass`: required evaluators are present, coverage meets the workload
  threshold, and no material quality regression is detected.
- `warn`: quality evidence exists but is partial, noisy, imported, or below the
  recommended coverage level.
- `fail`: required quality evidence is absent, evaluator coverage is below the
  minimum, or the favored candidate regresses on a blocking quality metric.

Evaluator coverage should be calculated against the same sample set used for
the performance comparison. Coverage from a different sample set can be
included as context, but it should not satisfy the gate unless provenance links
it to the compared run.

Minimum evaluator expectations by profile:

| Workload profile | Minimum evaluator coverage |
| --- | --- |
| `rag` | answer relevance, answer correctness, citation grounding, or factuality on compared samples |
| `long_context_qa` | correctness or factuality on long-context examples, with timeout and truncation visibility |
| `tool_calling` | tool selection validity and argument validity, plus task success when available |
| `agentic_coding` | task success, test result import, patch validation, or reviewed score import |
| `decode_heavy` | output validity, format validity, or task pass check sufficient to catch invalid high-throughput output |

The advisor should report evaluator coverage as both a ratio and a reason. For
example, partial coverage should produce a warning reason even if it does not
block the recommendation.

## Confidence Penalties

Confidence starts from the workload threshold result, then applies deterministic
penalties. The implementation should keep penalty values in a visible config or
constant table and include each applied penalty in the output.

| Penalty code | Trigger | Recommended impact |
| --- | --- | --- |
| `missing_required_metric` | A required metric for the workload profile is absent | Large penalty or blocking impact, depending on profile |
| `missing_recommended_metric` | A recommended metric is absent | Small to medium penalty |
| `low_sample_count` | Sample count is below the profile minimum | Medium to large penalty |
| `single_trial` | Only one trial is available for a noisy comparison | Medium penalty |
| `mock_only_evidence` | Evidence comes only from mock runs | Large penalty and no high-confidence recommendation |
| `imported_only_evidence` | Evidence is imported without direct run provenance | Medium penalty unless provenance is strong |
| `non_randomized_order` | Candidate order was fixed for the compared run | Medium penalty |
| `quality_coverage_partial` | Evaluator coverage is below the recommended target but above the blocking minimum | Medium penalty |
| `quality_gate_failed` | Required quality gate failed | Blocking impact |
| `wide_variance` | Repeated trials disagree beyond the profile tolerance | Medium to large penalty |

Penalty behavior should be monotonic: adding missing metrics or weakening
sample design should never increase confidence. Adding valid evidence can raise
confidence only when the new evidence satisfies a known gate or removes a known
penalty.

## Recommendation Logic

The advisor should produce one of four outcomes:

- `prefer_variant`: the variant clears quality gates and beats the baseline on
  the workload's primary decision focus with sufficient confidence.
- `prefer_baseline`: the baseline remains preferable because the variant fails
  quality, reliability, cost, or performance thresholds.
- `inconclusive`: the available evidence is directionally useful but not enough
  for a recommendation.
- `reject_run`: the run is not suitable for recommendation because critical
  evidence is invalid, contradictory, or missing.

Performance deltas should be interpreted within workload context. A lower
latency result should not win a `rag`, `tool_calling`, or `agentic_coding`
comparison if the favored candidate fails required quality gates.

## Next-Experiment Command Generation

When the advisor emits `inconclusive`, it should produce a concrete
next-experiment plan. The plan should be schema-oriented so implementation can
render a command only when all required bindings are available.

Target shape:

```yaml
next_experiment:
  needed: true
  reason: insufficient_quality_coverage
  objective: collect_required_quality_and_latency_evidence
  command_plan:
    command: kvoptbench run
    required_bindings:
      workload_profile: rag
      baseline_id: string
      variant_id: string
      sample_set_id: string
      endpoint_group_id: string
      evaluator_ids:
        - answer_relevance
    options:
      min_samples: 30
      repeated_trials: 2
      randomize_order: true
      required_metrics:
        - latency_ms
        - error_rate
        - total_tokens_per_second
      recommended_metrics:
        - cost_per_1k_tokens
        - cache_hit_rate
      output_format: result_package_ready
```

Rendered command template:

```text
kvoptbench run --workload-profile <workload_profile> --baseline <baseline_id> --variant <variant_id> --sample-set <sample_set_id> --endpoint-group <endpoint_group_id> --min-samples <min_samples> --repeated-trials <repeated_trials> --randomize-order --require-evaluator <evaluator_id> --require-metric <metric_id>
```

Rules:

- Do not invent config file names, endpoint identifiers, or sample set paths.
- Include unresolved bindings explicitly so a caller can fill them.
- Prefer the smallest experiment that removes the blocking uncertainty.
- Carry forward the reason code that triggered the plan.
- Include required metrics and evaluator requirements, not just sample count.
- Keep generated argument ordering stable for testability.

## Acceptance Criteria

- Advisor output includes confidence score, confidence band, reasons,
  penalties, missing metrics, quality gate status, and next-experiment plan
  when needed.
- Workload profiles have separate threshold configuration for `rag`,
  `long_context_qa`, `tool_calling`, `agentic_coding`, and `decode_heavy`.
- Required quality gates can block performance-only recommendations.
- Missing telemetry, low samples, mock-only evidence, imported-only evidence,
  and non-randomized order reduce confidence in deterministic ways.
- Inconclusive recommendations produce a schema-oriented command plan without
  inventing paths or config files.
- The same input fixture produces byte-stable advisor output after sorting.
- Public outputs contain no secrets, private endpoint details, or local paths.

## Testing Plan

Implementation should include tests that do not require GPUs, external APIs,
model weights, or live services.

- Golden fixture tests for each workload profile and recommendation outcome.
- Unit tests for quality gate pass, warn, fail, and blocking behavior.
- Unit tests for each penalty trigger and for monotonic confidence behavior.
- Determinism tests that shuffle input metric and reason ordering, then assert
  stable output ordering.
- Command-plan tests that verify unresolved bindings remain placeholders and
  no config paths are invented.
- Public-safety tests that scan advisor records for local paths, secrets, and
  forbidden private identifiers.
- Regression tests for mock-only, imported-only, low-sample, and
  non-randomized comparisons.

## Open Implementation Notes

- Confidence bands should be versioned with the advisor configuration so older
  result packages remain interpretable.
- The advisor should expose enough detail for a user to disagree with the
  recommendation without reading source code.
- Workload thresholds should be easy to extend without changing the core
  scoring flow.
