# Architecture Notes

This directory contains implementation-facing design notes for KVOptBench features
that extend benchmark credibility without turning the project into a serving
engine or infrastructure orchestrator.

Read these documents in this order:

1. `telemetry_lifecycle.md`
2. `environment_capture.md`
3. `import_adapters.md`
4. `strategy_plan_run.md`
5. `advisor_confidence.md`
6. `lmcache_scbench_extensions.md`

The design rule across all documents is the same: KVOptBench records, imports,
normalizes, packages, and recommends from evidence. It does not provision GPUs,
manage model-serving processes, implement a KV cache, or fabricate backend
telemetry when a system does not expose it.

Every implementation slice should include:

- config-driven behavior
- request-level or run-level provenance
- explicit missing metric handling
- tests that do not require a GPU, model weights, external APIs, or live services
- public-safe documentation with no secrets or private endpoint details
