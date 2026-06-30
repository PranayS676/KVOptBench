# Dataset Adapter Contract

This document defines the public contract for dataset adapters in KVOptBench.
Adapters convert public benchmark datasets and corpora into KVOptBench workload JSONL
without making the generic runner dataset-specific.

The current implementation includes QASPER, Project Gutenberg, LongBench, BEIR SciFact,
and BFCL. Future adapters should follow this same contract.

## Goals

Dataset adapters must:

- produce `WorkloadItem`-compatible JSONL
- produce a manifest JSON for reproducibility
- preserve dataset provenance and license metadata
- support tiny fixture mode for tests
- avoid hardcoding model, engine, provider, or strategy choices
- avoid committing large raw or generated datasets
- keep downloads optional in default tests

Dataset adapters must not:

- call a model endpoint
- run experiments
- write results under `results/raw/`
- infer missing labels or expected answers without marking them as derived
- silently truncate prompts without recording truncation metadata
- download data during normal unit tests

## CLI Usage

The top-level command is:

```bash
kvoptbench dataset prepare \
  --source qasper \
  --mode shared_prefix \
  --split validation \
  --source-path path/to/qasper.json \
  --out workloads/generated/qasper_shared_prefix_32k.jsonl \
  --manifest workloads/generated/qasper_manifest.json
```

Adapter-specific options should be explicit:

```bash
kvoptbench dataset prepare \
  --source qasper \
  --split validation \
  --mode shared_prefix \
  --target-input-tokens 32768 \
  --target-output-tokens 256 \
  --max-items 100 \
  --out workloads/generated/qasper_shared_prefix_32k.jsonl \
  --manifest workloads/generated/qasper_manifest.json
```

Long-context or needle workloads should support context buckets:

```bash
kvoptbench dataset prepare \
  --source gutenberg \
  --mode needle \
  --download \
  --cache-dir data/raw \
  --book-ids 1342,84,2701 \
  --context-buckets 8192,32768,65536,131072 \
  --out workloads/generated/gutenberg_needle_8k_128k.jsonl \
  --manifest workloads/generated/gutenberg_manifest.json
```

Hugging Face-backed downloads require optional data dependencies:

```bash
python -m pip install -e ".[data]"
```

Generic options:

| Option | Required | Meaning |
|---|---:|---|
| `--source` | yes | Dataset adapter name. |
| `--mode` | yes | Workload mode, such as `shared_prefix`, `random_prefix`, `needle`, or `rag`. |
| `--split` | adapter-specific | Dataset split or subset. |
| `--out` | yes | Workload JSONL output path. |
| `--manifest` | yes | Manifest JSON output path. |
| `--max-items` | no | Cap generated rows for smoke runs. |
| `--seed` | no | Deterministic sampling seed. |
| `--target-input-tokens` | no | Approximate input token target. |
| `--target-output-tokens` | no | Approximate generation target. |
| `--context-buckets` | no | Comma-separated long-context buckets. |
| `--book-ids` | no | Comma-separated Project Gutenberg book IDs. |
| `--cache-dir` | no | Local dataset cache path. |
| `--download` | no | Allow the adapter to download source data. |
| `--dataset-revision` | no | Optional upstream dataset revision or ref. |
| `--subset` | no | Comma-separated dataset subset names or BFCL source files. |
| `--force` | no | Refresh cached source files instead of reusing them. |
| `--source-path` | no | Use an existing local dataset path instead of downloading. |

Default behavior uses `--source-path` or already-cached data. Downloading must be
explicit because raw public datasets can be large and may have publication terms that
need user review.

## Workload JSONL Schema

Each output line must validate as `WorkloadItem`.

Required fields:

```json
{
  "schema_version": "1",
  "workload_item_version": "1",
  "task_id": "qasper_paper123_q1",
  "workload": "qasper_shared_prefix",
  "category": "shared_prefix_long_document_qa",
  "prompt": "You are given the following paper...",
  "expected_answer": "the expected answer",
  "target_input_tokens": 32768,
  "target_output_tokens": 256,
  "prefix_group_id": "qasper_paper123",
  "shared_prefix_tokens": 28672,
  "eval_type": "contains_expected",
  "metadata": {
    "dataset": "qasper",
    "dataset_source_url": "https://huggingface.co/datasets/allenai/qasper",
    "mode": "shared_prefix",
    "source_document_id": "paper123",
    "source_question_id": "q1",
    "prefix_hash": "sha256 of shared prefix text",
    "prompt_hash": "sha256 of full prompt text",
    "expected_answer_hash": "sha256 of expected answer when answers should not be exposed",
    "tokenizer_id": "Qwen/Qwen3-32B or null",
    "tokenizer_revision": "revision hash or null",
    "token_count_method": "hf_tokenizer|tiktoken|char_approx_4",
    "measured_input_tokens": 32768,
    "measured_shared_prefix_tokens": 28672,
    "measured_suffix_tokens": 4096,
    "truncated": false,
    "truncation_policy": "none",
    "redistributable_prompt": false,
    "redistributable_output": false,
    "rights_note": "Dataset/corpus rights note."
  }
}
```

Recommended metadata:

```json
{
  "dataset": "qasper",
  "dataset_source_url": "https://huggingface.co/datasets/allenai/qasper",
  "dataset_revision": "pinned revision or version",
  "source_license": "cc-by-4.0",
  "split": "validation",
  "source_id": "paper123",
  "source_record_id": "original dataset row id",
  "source_document_id": "paper123",
  "question_id": "q1",
  "source_question_id": "q1",
  "answer_source": "extractive_span",
  "answer_type": "extractive",
  "context_bucket": 32768,
  "prefix_hash": "sha256 of shared prefix text",
  "prompt_hash": "sha256 of full prompt text",
  "expected_answer_hash": "sha256 of expected answer when answer text is not redistributable",
  "tokenizer_id": "Qwen/Qwen3-32B or null",
  "tokenizer_revision": "revision hash or null",
  "token_count_method": "hf_tokenizer|tiktoken|char_approx_4",
  "measured_input_tokens": 32768,
  "measured_shared_prefix_tokens": 28672,
  "measured_suffix_tokens": 4096,
  "prompt_template": "qasper_shared_prefix_v1",
  "prompt_template_hash": "sha256 of prompt template text",
  "adapter": "qasper",
  "adapter_version": "0.1.0",
  "truncated": false,
  "truncation_policy": "none|head|tail|middle",
  "excluded_reason": null,
  "needle_id": null,
  "needle_position_ratio": null,
  "needle_position_bucket": null,
  "evaluator": "contains_expected",
  "evaluator_version": "0.1.0",
  "difficulty": null,
  "redistributable_prompt": false,
  "redistributable_output": false,
  "rights_note": "Dataset/corpus rights note."
}
```

Required rules:

- `task_id` must be stable for the same dataset revision and adapter options.
- `prefix_group_id` must be set for shared-prefix/cache workloads.
- `shared_prefix_tokens` must be nonzero for intentional prefix reuse.
- `metadata.prefix_hash` must be set for cache workloads so users can verify
  that requests actually share the same prefix.
- `metadata.prompt_hash` must be set for every row.
- `metadata.dataset` must be present for public dataset adapters.
- `metadata.dataset_source_url` must be present.
- `metadata.source_license` or `metadata.rights_note` must be present.
- `metadata.token_count_method` must explain whether counts came from an exact tokenizer
  or an approximation.
- If exact tokenization is used, `metadata.tokenizer_id` and `metadata.tokenizer_revision`
  should be recorded.
- If prompt text is truncated, `metadata.truncated` must be true and the truncation rule must be recorded.
- If expected answers are derived or normalized, record the derivation in metadata.
- If full prompt text or answers cannot be redistributed, keep the hashes and metadata but
  set `redistributable_prompt` or `redistributable_output` to false.

## Manifest Schema

Every adapter run must write a manifest next to the workload JSONL.

Recommended manifest:

```json
{
  "schema_version": "1",
  "manifest_version": "1",
  "kvoptbench_version": "0.1.0",
  "git_commit": "repository commit hash",
  "adapter_name": "qasper",
  "adapter_version": "0.1.0",
  "dataset_name": "QASPER",
  "dataset": "qasper",
  "dataset_source_url": "https://huggingface.co/datasets/allenai/qasper",
  "source_url": "https://huggingface.co/datasets/allenai/qasper",
  "dataset_revision": "pinned revision or version",
  "source_revision": "pinned revision or version",
  "license": "cc-by-4.0",
  "rights_note": "plain-English rights note",
  "license_review_status": "checked",
  "redistribution_policy": "adapter_only|manifest_only|tiny_fixture_allowed|full_workload_allowed",
  "adapter": "qasper",
  "mode": "shared_prefix",
  "split": "validation",
  "generation_command": "kvoptbench dataset prepare ...",
  "generation_started_at": "ISO-8601 timestamp",
  "generation_finished_at": "ISO-8601 timestamp",
  "generation_duration_sec": 12.3,
  "created_at": "ISO-8601 timestamp",
  "python_version": "3.11.x",
  "tokenizer_id": "Qwen/Qwen3-32B or null",
  "tokenizer_revision": "revision hash or null",
  "token_count_method": "hf_tokenizer|tiktoken|char_approx_4",
  "sampling_method": "deterministic|random_seeded",
  "seed": 7,
  "max_items_requested": 100,
  "max_items_emitted": 100,
  "row_count": 100,
  "excluded_count": 12,
  "exclusion_reasons": {
    "missing_answer": 8,
    "too_short": 4
  },
  "target_input_tokens": 32768,
  "target_output_tokens": 256,
  "context_buckets": [32768],
  "min_input_tokens": 8192,
  "max_input_tokens": 32768,
  "avg_input_tokens": 21120,
  "input_token_histogram": {},
  "prefix_group_count": 20,
  "items_per_prefix_group": {},
  "workload_sha256": "hex digest",
  "workload_hash": "hex digest",
  "source_sha256": "hex digest or null",
  "prompt_template": "qasper_shared_prefix_v1",
  "prompt_template_hash": "sha256 of prompt template text",
  "normalization_rules": [],
  "truncation_policy": "none|head|tail|middle",
  "truncation_count": 0,
  "known_limitations": [],
  "notes": [
    "Raw dataset is not committed to the repository."
  ]
}
```

Required manifest fields:

- `schema_version`
- `manifest_version`
- `kvoptbench_version`
- `git_commit`
- `dataset`
- `dataset_name`
- `dataset_source_url`
- `source_url`
- `license` or `rights_note`
- `license_review_status`
- `redistribution_policy`
- `adapter`
- `adapter_name`
- `adapter_version`
- `mode`
- `generation_command`
- `row_count`
- `workload_sha256`
- `token_count_method`
- `prompt_template`
- `prompt_template_hash`

Optional but strongly recommended:

- `source_revision`
- `dataset_revision`
- `split`
- `source_sha256`
- `excluded_count`
- `exclusion_reasons`
- `context_buckets`
- `created_at`
- `generation_started_at`
- `generation_finished_at`
- `generation_duration_sec`
- `python_version`
- `tokenizer_id`
- `tokenizer_revision`
- `sampling_method`
- `seed`
- `prefix_group_count`
- `items_per_prefix_group`
- `truncation_policy`
- `truncation_count`
- `known_limitations`

## Adapter Registry

Adapter code uses a registry so CLI dispatch does not hardcode adapter logic inside
experiment runners.

Current layout:

```text
kvoptbench/datasets/
  __init__.py
  registry.py
  manifest.py
  cache.py
  download.py
  huggingface.py
  qasper.py
  gutenberg.py
  longbench.py
  beir.py
  bfcl.py
```

Possible future adapters include InfiniteBench, Natural Questions, CodeSearchNet, and
SWE-bench.

The registry should expose:

```python
def list_dataset_adapters() -> list[DatasetAdapterInfo]:
    ...

def get_dataset_adapter(name: str) -> DatasetAdapter:
    ...
```

Each adapter should declare:

- adapter name
- supported modes
- source URL
- license or rights note
- whether download is supported
- whether test fixtures are included
- required optional dependencies, if any

## Adapter Interface

Recommended interface:

```python
class DatasetAdapter(Protocol):
    name: str
    source_url: str
    supported_modes: tuple[str, ...]

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        ...
```

Recommended result:

```python
class DatasetPrepareResult(BaseModel):
    output_path: Path
    manifest_path: Path
    row_count: int
    workload_sha256: str
    excluded_count: int = 0
```

The adapter can use `WorkloadItem` directly, but it should not import runner or
analysis code.

## Prompt Template Rules

Prompt templates should be versioned because prompt wording affects benchmark results.

Example template name:

```text
qasper_shared_prefix_v1
```

Template rules:

- Keep task instructions stable once used in a public run.
- Include answer constraints needed by the evaluator.
- Do not include hidden benchmark answers in the prompt.
- Record template name in every row and manifest.
- If the template changes, use a new version suffix.

## Dataset-Specific Modes

### QASPER

Supported modes:

- `shared_prefix`
- `random_prefix`
- `partial_prefix_sweep`

Required options:

- `--split`
- `--target-input-tokens`
- `--target-output-tokens`
- `--max-items`
- one of `--source-path` or `--download`

Output expectations:

- `prefix_group_id` is the paper ID for shared-prefix mode.
- Random-prefix mode uses unrelated paper prefixes and records both source IDs.
- Partial-prefix sweep mode records the configured overlap ratio and `prefix_hash` for
  each overlap bucket.
- Questions without usable answers are skipped or marked with an exclusion reason.

### Project Gutenberg

Supported modes:

- `needle`
- `no_needle_control`
- `multi_needle`
- `conflicting_needle`
- `long_document_qa` later

Required options:

- `--book-ids` or a source directory with `books.json`
- one of `--source-path` or `--download`
- `--context-buckets`
- `--seed`

Output expectations:

- each needle has a deterministic answer string
- `metadata.needle_position` is one of `beginning`, `middle`, `end`, `multi`, or `conflict`
- no-needle controls have a null expected needle answer and an explicit evaluator mode
- context bucket and book ID are recorded
- rights note is recorded for each book

### LongBench

Supported modes:

- `long_context_qa`
- `long_context_retrieval`
- `code_context`

Required options:

- `--subset`
- `--split`
- `--max-items`
- one of `--source-path` or `--download`

Output expectations:

- subtask is preserved in metadata
- answers are copied into `expected_answer` or `metadata.expected_answers`
- evaluation method is chosen per subtask

### BEIR SciFact

Supported modes:

- `rag`

Required options:

- one of `--source-path` or `--download`
- `--split` when selecting a non-default qrels split

Output expectations:

- prompt includes only selected passages
- expected documents and qrels are recorded
- retrieval method and source are recorded

Natural Questions is a useful future RAG adapter target, but it is not currently
implemented.

### BFCL

Supported modes:

- `tool_calling`

Required options:

- one of `--source-path` or `--download`
- `--subset` when selecting BFCL source files other than the default `BFCL_v3_simple`

Output expectations:

- prompt includes available tools and one user request
- expected tool name is recorded in `expected_answer`
- `expected_schema` describes the placeholder JSON tool-call shape
- tool count and source file are preserved in metadata

## Testing Requirements

Default tests must not download large datasets or require network access.

Required tests:

- registry lists adapter metadata
- tiny fixture adapter outputs valid `WorkloadItem` rows
- manifest includes required fields
- workload hash changes when output changes
- QASPER shared-prefix rows set `prefix_group_id`
- QASPER random-prefix rows avoid intentional prefix reuse
- Gutenberg needle rows include expected answer and needle metadata
- docs list every implemented adapter
- README links to dataset docs

Optional integration tests:

- real dataset download
- adapter generation against a small official split
- hash and row-count stability for pinned revisions

Integration tests should be gated by an environment variable such as:

```bash
KVOPTBENCH_DATASET_DOWNLOAD=1
```

## Failure Handling

Adapters should fail loudly when:

- source data is missing
- source license or rights note cannot be recorded
- required expected-answer fields are absent
- output path already exists and overwrite is not allowed
- generated rows fail `WorkloadItem` validation

Adapters should record exclusions when:

- a row is missing an answer
- context is too short for the requested bucket
- context exceeds the adapter's configured max length and truncation is disabled
- source text fails cleanup

## Publication Contract

A public benchmark result should cite the manifest, not just the workload file.

At publication time, include:

- workload JSONL hash
- manifest hash
- adapter name and version
- prompt template version
- dataset source URL
- dataset revision or dump date
- license or rights note
- generated row count
- exclusions
- exact generation command

Without this metadata, results should be treated as exploratory.

## Metric Provenance Rules

Reports must distinguish backend-reported metrics from KVOptBench-derived proxies.

Use explicit names when possible:

```text
engine_reported_cache_hit_rate
engine_reported_cache_hit_tokens
engine_reported_cache_miss_tokens
cache_hit_proxy
cache_miss_penalty_ms
```

Rules:

- Do not present a proxy as if it were an internal engine metric.
- If the backend does not expose a metric, keep the metric null and list it in
  `missing_metrics`.
- If KVOptBench derives a proxy from timing, name it as a proxy and document the formula.
- If two engines expose semantically different metrics under similar names, keep them
  engine-specific until an adapter normalizes them.

## Comparability Rules

Two result packages are not directly comparable when any of these differ:

- model ID or model revision
- engine name or engine version
- tokenizer ID, tokenizer revision, or token-count method
- prompt template or prompt template hash
- dataset revision, split, or adapter version
- context bucket or truncation policy
- strategy flags or cache configuration
- hardware type, GPU count, runtime image, driver, or CUDA/runtime version
- sampling parameters, max output tokens, stream mode, retry policy, or timeout policy
- request order, warmup procedure, concurrency, or request rate

If any of those differ, the report should either avoid direct comparison or label the
comparison as exploratory with the differing fields listed.
