# KVOptBench Artifact Schema Contracts

KVOptBench publishes JSON Schema snapshots for artifacts that users may store,
compare, package, or validate outside the Python process.

Current contract version: `1`.

Version policy:

- Do not change the meaning or type of an existing field within the same schema version.
- New optional or nullable fields may be added to version `1` when older artifacts remain valid.
- Breaking changes require a new `schemas/vN/` directory and validator support for both old and new supported versions.
- Validators must reject unknown future `schema_version` values instead of silently accepting them.
- Every intentional schema change must update the committed snapshots and tests.

Check committed snapshots with:

```bash
kvoptbench schema export --output-dir schemas/v1 --check
```

Validate publishable artifacts with:

```bash
kvoptbench validate-results --input results/raw
kvoptbench validate-package --path results/packages/golden
```
