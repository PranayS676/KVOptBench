# Public Dataset Guide

KVOptBench can validate its local harness with synthetic workloads, but credible real
endpoint testing needs public, reproducible workload data. This guide defines the
implemented dataset adapters, recommended future datasets, how each maps to KVOptBench
workload families, and what must be recorded before using the data in a self-hosted
frontier-model run.

KVOptBench should not commit large raw datasets. The repo should commit adapter code,
manifest schemas, tiny fixtures, source URLs, and reproducible commands. Raw downloaded
data and generated workload JSONL files should stay under ignored local paths such as
`workloads/generated/`, `data/raw/`, or a provider volume.

## Selection Criteria

A dataset is suitable for public KVOptBench runs when it has:

- a stable public source URL
- clear license or usage terms
- task inputs that can be transformed into `WorkloadItem` JSONL
- expected answers, evidence, labels, or scoring rules
- enough length variation to test cache and long-context behavior
- a control strategy, such as random-prefix or shuffled-document controls
- a reproducible split, revision, hash, and generation command

Avoid datasets when:

- terms prohibit benchmark redistribution or publication
- answers are unavailable and no evaluator can score the task
- source text includes private, sensitive, or user-identifiable data
- source downloads are unstable and cannot be pinned
- adapter behavior would silently truncate or alter the task without recording it

## Recommended Dataset Matrix

| KVOptBench need | Primary dataset | Source | First use | Notes |
|---|---|---|---|---|
| Shared-prefix cache | QASPER | https://huggingface.co/datasets/allenai/qasper | repeated paper prefix with many questions | Strong first choice because many questions share the same paper context. |
| Random-prefix control | QASPER-derived control | https://huggingface.co/datasets/allenai/qasper | same token lengths, unrelated paper prefixes | Build by pairing questions with different papers or sampling unrelated papers. |
| Partial-prefix cache sweep | QASPER-derived overlap control | https://huggingface.co/datasets/allenai/qasper | 0%, 25%, 50%, 75%, and 90% prefix overlap | Use after shared/random controls to estimate cache sensitivity. |
| Long-context QA | LongBench / LongBench-E | https://huggingface.co/datasets/zai-org/LongBench | long-context QA, summarization, code, synthetic retrieval | Composite benchmark; record subtask and upstream source notes. |
| 100K+ stress | InfiniteBench | https://github.com/OpenBMB/InfiniteBench | frontier-scale long-context stress | Use after shorter context tests pass; higher GPU and timeout risk. |
| RAG and retrieval | BEIR | https://github.com/beir-cellar/beir | retrieval-grounded prompts with known qrels | Good for RAG faithfulness and retrieval quality experiments. |
| Real QA over Wikipedia | Natural Questions | https://github.com/google-research-datasets/natural-questions | Wikipedia-grounded QA | Good for real user-style questions; large downloads need local cache. |
| Public-domain long documents | Project Gutenberg | https://www.gutenberg.org/ | long contexts and needle injection | Check each work's rights and remove Project Gutenberg license/header text if required by the use case. |
| Public wiki corpus | Wikimedia dumps | https://dumps.wikimedia.org/ | RAG corpus and passage retrieval | Respect user-agent and rate-limit policy; record dump date. |
| Code retrieval | CodeSearchNet | https://github.com/github/CodeSearchNet | code/docstring retrieval and code context | Archived but useful; large corpus, adapter should support tiny sample mode. |
| Agentic/code repair | SWE-bench | https://github.com/SWE-bench/SWE-bench | issue-to-patch or repo-context workflows | Higher setup cost; use later after basic endpoint testing works. |
| Function/tool calling | BFCL | https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard | function-call correctness | Good for structured output and tool-call quality checks. |
| Tool-use workflows | API-Bank / ToolBench | https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank and https://github.com/OpenBMB/ToolBench | multi-step tool-use prompts | Use after the basic tool-call evaluator is upgraded. |

Implemented adapters:

```text
qasper: shared_prefix, random_prefix, partial_prefix_sweep
gutenberg: needle, no_needle_control, multi_needle, conflicting_needle
longbench: long_context_qa, long_context_retrieval, code_context
beir_scifact: rag
bfcl: tool_calling
```

All implemented adapters support local fixture/source mode. Download support is explicit
through `--download`; Hugging Face-backed downloads require `pip install -e ".[data]"`.

## Recommended First Dataset Pack

The first publishable self-hosted frontier-model run should be narrow. Use QASPER
shared-prefix and random-prefix controls at 8K and 32K first. That proves the core
cache-aware claim before adding broader datasets.

Recommended first benchmark shape:

```text
dataset: QASPER validation split
modes: shared_prefix and random_prefix_control
engine: one of vLLM or SGLang
model: one self-hosted frontier-adjacent model
context buckets: 8K and 32K
concurrency: 1, 4, 8
passes: cold and warm
items: 50-100 shared-prefix and 50-100 random-prefix
```

The broader first dataset pack should be implemented in this order:

1. **QASPER shared-prefix cache pack**
   - Purpose: prefix/radix cache benefit.
   - Shape: one paper body reused as the prefix; multiple questions against the same paper.
   - Output: `workloads/generated/qasper_shared_prefix_32k.jsonl`.
   - Evaluator: `contains_expected`, `exact_match`, or future evidence-aware evaluator.
   - Key metadata: `dataset=qasper`, `paper_id`, `question_id`, `answer_type`,
     `evidence_count`, `source_license=cc-by-4.0`.

   Command:

   ```bash
   kvoptbench dataset prepare \
     --source qasper \
     --mode shared_prefix \
     --download \
     --cache-dir data/raw \
     --split validation \
     --target-input-tokens 32768 \
     --target-output-tokens 256 \
     --max-items 100 \
     --out workloads/generated/qasper_shared_prefix_32k.jsonl \
     --manifest workloads/generated/qasper_shared_prefix_manifest.json
   ```

2. **QASPER random-prefix control pack**
   - Purpose: prove cache gains are not caused by unrelated scheduling effects.
   - Shape: same approximate input/output buckets as shared-prefix pack, but unrelated
     papers and no intentional prefix reuse.
   - Output: `workloads/generated/qasper_random_prefix_32k.jsonl`.
   - Evaluator: same as shared-prefix pack.
   - Key metadata: `control_type=random_prefix`, `source_paper_id`,
     `question_paper_id`, `target_input_bucket`.

   Command:

   ```bash
   kvoptbench dataset prepare \
     --source qasper \
     --mode random_prefix \
     --download \
     --cache-dir data/raw \
     --split validation \
     --target-input-tokens 32768 \
     --target-output-tokens 256 \
     --max-items 100 \
     --out workloads/generated/qasper_random_prefix_32k.jsonl \
     --manifest workloads/generated/qasper_random_prefix_manifest.json
   ```

3. **QASPER partial-prefix sweep**
   - Purpose: measure cache sensitivity across controlled overlap levels.
   - Shape: related paper prefixes with 0%, 25%, 50%, 75%, and 90% shared prefix.
   - Output: `workloads/generated/qasper_partial_prefix_sweep.jsonl`.
   - Evaluator: same as shared-prefix pack.
   - Key metadata: `prefix_overlap_ratio`, `prefix_hash`, `prompt_hash`,
     `measured_shared_prefix_tokens`, `measured_suffix_tokens`.

4. **Project Gutenberg needle pack**
   - Purpose: long-context pressure and answer retrieval at known positions.
   - Shape: public-domain book text with inserted synthetic needles at beginning,
     middle, end, and multi-needle positions.
   - Output: `workloads/generated/gutenberg_needle_8k_128k.jsonl`.
   - Evaluator: `needle`.
   - Key metadata: `dataset=gutenberg`, `book_id`, `book_title`, `needle_position`,
     `context_bucket`, `source_rights_note`.

5. **LongBench subset**
   - Purpose: real long-context QA and summarization beyond synthetic needles.
   - Shape: selected English tasks such as `qasper`, `multifieldqa_en`,
     `hotpotqa`, `passage_retrieval_en`, and `repobench-p`.
   - Output: `workloads/generated/longbench_core.jsonl`.
   - Evaluator: task-specific placeholder first, stronger metrics later.
   - Key metadata: `dataset=longbench`, `subset`, `language`, `length`,
     `original_id`, `task_type`.

6. **BEIR or Natural Questions RAG pack**
   - Purpose: retrieval-grounded prompts and answer faithfulness.
   - Shape: query plus selected passages, expected answer/qrels, and citation requirement.
   - Output: `workloads/generated/rag_public_small.jsonl`.
   - Evaluator: `rag_placeholder` first, future citation/evidence evaluator later.
   - Key metadata: `dataset`, `query_id`, `doc_ids`, `qrels`, `retrieval_source`,
     `source_revision`.

## Dataset-Specific Adapter Notes

### QASPER

Use QASPER first for cache experiments because its natural unit is a paper with many
questions. The implemented adapter:

- flattens paper sections into a deterministic document prefix
- creates one `prefix_group_id` per paper
- generates one task per question
- chooses an expected answer from extractive spans, free-form answers, or yes/no fields
- stores answer provenance in metadata
- records skipped or unanswerable questions as exclusions
- generates random-prefix controls by using unrelated paper prefixes
- generates partial-prefix sweeps at 0%, 25%, 50%, 75%, and 90% overlap after the
  shared-prefix and random-prefix modes pass
- records `prefix_hash`, `prompt_hash`, tokenizer metadata, measured token counts,
  truncation policy, and exclusion reason for every row

Recommended outputs:

```text
workloads/generated/qasper_shared_prefix_32k.jsonl
workloads/generated/qasper_random_prefix_32k.jsonl
workloads/generated/qasper_partial_prefix_sweep.jsonl
workloads/generated/qasper_manifest.json
```

### Project Gutenberg

Use Project Gutenberg for long-context and needle tests because it provides long public
texts with stable book identifiers. The implemented adapter:

- accepts explicit book IDs or a local source directory with `books.json`
- records the rights note for each selected book
- downloads selected books only when `--download` is passed
- inserts synthetic needles with deterministic IDs
- generates context buckets such as `8192`, `32768`, `65536`, and `131072`
- creates controls with no needle, multi-needle, or conflicting needles

Recommended outputs:

```text
workloads/generated/gutenberg_needle_8k_128k.jsonl
workloads/generated/gutenberg_manifest.json
```

### LongBench

Use LongBench after QASPER/Gutenberg pass because it is broader and more heterogeneous.
The implemented adapter:

- allows selecting subsets explicitly
- preserves original fields such as `input`, `context`, `answers`, `length`,
  `dataset`, `language`, and `_id` in metadata
- maps `context` into the prompt body and `answers` into expected answers
- records the subtask because evaluation differs by task
- records any truncation or context-bucket selection

Recommended first subsets:

```text
qasper
multifieldqa_en
hotpotqa
passage_retrieval_en
repobench-p
```

### BEIR SciFact And Natural Questions

Use these for RAG experiments, not pure cache experiments. The implemented BEIR SciFact
adapter:

- reads BEIR-style corpus, queries, and qrels from a local source path or downloaded zip
- records document IDs and query IDs
- preserves qrels or expected answer references
- writes prompts that require grounded answers and citations
- supports a tiny smoke subset for CI fixtures

Natural Questions is recommended for future RAG coverage, but it is not implemented yet.

### Code And Tool Datasets

BFCL is implemented for a first tool-calling smoke path. SWE-bench, CodeSearchNet,
API-Bank, and ToolBench remain useful future extensions with heavier setup, richer
evaluation needs, and more licensing or redistribution details.

## Storage And Git Policy

Commit:

- adapter code
- docs
- manifests for tiny public samples when license allows
- tiny fixtures used by tests
- source URLs and generation commands

Do not commit:

- full dataset downloads
- large generated workload JSONL files
- model outputs from private prompts
- endpoint credentials
- provider tokens
- private benchmark corpora

Recommended local layout:

```text
data/raw/<dataset>/
data/processed/<dataset>/
workloads/generated/<dataset>/
reports/outputs/
results/raw/
```

The `.gitignore` should continue to exclude generated data and result files by default.

## Known Risks And Mitigations

QASPER does not guarantee that every paper can produce a clean 32K prompt. Adapters must
record `target_input_tokens`, `measured_input_tokens`, `truncated`, `truncation_policy`,
and `excluded_reason` instead of silently padding, truncating, or dropping rows.

Project Gutenberg should be treated as a corpus source, not a permission shortcut. Commit
book IDs, adapter code, tiny fixtures, and manifests. Do not commit full book text or large
generated prompts, and record a `rights_note` per book.

BEIR is a collection with per-subdataset licensing. RAG adapters must record the exact
subdataset, source URL, license, and rights note.

Backend cache telemetry may be unavailable or engine-specific. Reports should keep
engine metrics such as `engine_reported_cache_hit_rate` separate from KVOptBench-derived
fields such as `cache_hit_proxy`, and unavailable metrics must remain `null` and appear
in `missing_metrics`.

Quality checks should exist from the first public run. Use `contains_expected`,
`exact_match`, `needle`, and `json_validity` where applicable, then mark richer RAG,
tool-call, or LLM-judge scoring as placeholder until implemented.

## Publication Checklist

Every public result using a real dataset should include:

- run manifest
- dataset name
- dataset source URL
- dataset revision, split, or dump date
- adapter name and version
- repository commit
- generation command
- generated workload hash
- manifest hash
- license or rights note
- license review status
- redistribution policy
- row count
- context buckets
- excluded rows and exclusion reasons
- prompt template version
- prompt template hash
- tokenizer ID and revision
- token count method
- evaluator type
- `engine_reported_cache_hit_rate` when exposed
- `cache_hit_proxy` when derived from timing
- `missing_metrics`
- known limitations

If any of those fields are missing, mark the run as exploratory rather than official.
