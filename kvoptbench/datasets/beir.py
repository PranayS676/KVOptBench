"""BEIR SciFact adapter for RAG-style workload preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kvoptbench.datasets.download import download_file, extract_zip
from kvoptbench.datasets.hashing import sha256_text
from kvoptbench.datasets.manifest import (
    DatasetAdapterInfo,
    DatasetPrepareOptions,
    DatasetPrepareResult,
    build_manifest,
    source_hash,
    write_manifest,
    write_workload_jsonl,
)
from kvoptbench.datasets.cache import resolve_dataset_cache_dir
from kvoptbench.datasets.token_counting import count_tokens, truncate_to_token_budget
from kvoptbench.schemas import WorkloadItem, utc_now_iso

BEIR_SCIFACT_SOURCE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
BEIR_SCIFACT_PAGE_URL = "https://huggingface.co/datasets/BeIR/scifact"
BEIR_DOWNLOAD_DEFAULT_MAX_ITEMS = 100
BEIR_TEMPLATE = (
    "Answer the scientific claim using only the retrieved evidence. Cite the source id.\n\n"
    "Claim:\n{query}\n\nEvidence:\n{evidence}\n\nAnswer:"
)


class BeirScifactAdapter:
    """Prepare BEIR SciFact queries for RAG experiments."""

    info = DatasetAdapterInfo(
        name="beir_scifact",
        dataset_name="BEIR SciFact",
        source_url=BEIR_SCIFACT_PAGE_URL,
        supported_modes=("rag",),
        license=None,
        rights_note="Use upstream BEIR/SciFact dataset terms and attribution.",
        download_supported=True,
        fixture_supported=True,
    )

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        if options.mode not in self.info.supported_modes:
            raise ValueError(f"BEIR SciFact adapter does not support mode '{options.mode}'")
        run_options = _options_with_download_defaults(options)
        source_path, download_meta = _resolve_source_path(run_options)

        started = time.perf_counter()
        started_at = utc_now_iso()
        dataset = _load_beir_dataset(source_path, run_options.split or "test")
        items, exclusions = _generate_items(dataset, run_options)
        if run_options.max_items is not None:
            items = items[: run_options.max_items]
        if not items:
            raise ValueError("BEIR SciFact adapter produced no workload rows")

        workload_sha256 = write_workload_jsonl(items, run_options.out)
        finished_at = utc_now_iso()
        manifest = build_manifest(
            info=self.info,
            options=run_options,
            items=items,
            workload_sha256=workload_sha256,
            prompt_template="beir_scifact_rag_v1",
            prompt_template_text=BEIR_TEMPLATE,
            license_review_status=_license_review_status(run_options),
            redistribution_policy=_redistribution_policy(run_options),
            excluded_reasons=exclusions,
            rights_note=self.info.rights_note,
            source_sha256=source_hash(source_path),
            generation_started_at=started_at,
            generation_finished_at=finished_at,
            generation_duration_sec=round(time.perf_counter() - started, 3),
            cache_path=download_meta.get("cache_path"),
            download_method=download_meta.get("download_method"),
            downloaded_at=download_meta.get("downloaded_at"),
            source_url=download_meta.get("source_url"),
            known_limitations=[
                "RAG quality uses a placeholder check over the expected source id.",
                "Downloaded rows are cached locally; raw public dataset files should not be committed.",
            ],
        )
        write_manifest(manifest, run_options.manifest)
        return DatasetPrepareResult(
            output_path=run_options.out,
            manifest_path=run_options.manifest,
            row_count=len(items),
            workload_sha256=workload_sha256,
            excluded_count=len(exclusions),
        )


def _generate_items(
    dataset: dict[str, Any], options: DatasetPrepareOptions
) -> tuple[list[WorkloadItem], list[str]]:
    items: list[WorkloadItem] = []
    exclusions: list[str] = []
    corpus: dict[str, dict[str, Any]] = dataset["corpus"]
    queries: dict[str, str] = dataset["queries"]
    qrels: dict[str, list[str]] = dataset["qrels"]
    for query_id in sorted(queries):
        source_ids = qrels.get(query_id, [])
        if not source_ids:
            exclusions.append("missing_qrel")
            continue
        evidence_parts = []
        for source_id in source_ids[:3]:
            document = corpus.get(source_id)
            if not document:
                continue
            title = str(document.get("title") or source_id)
            text = str(document.get("text") or "")
            evidence_parts.append(f"Source {source_id}: {title}\n{text}")
        if not evidence_parts:
            exclusions.append("missing_corpus_document")
            continue
        expected_answer = source_ids[0]
        evidence = truncate_to_token_budget(
            "\n\n".join(evidence_parts), options.target_input_tokens, options.token_count_method
        )
        prompt = BEIR_TEMPLATE.format(query=queries[query_id], evidence=evidence)
        measured_input_tokens = count_tokens(prompt, options.token_count_method)
        items.append(
            WorkloadItem(
                task_id=f"beir_scifact_rag_{_slug(query_id)}",
                workload="beir_scifact_rag",
                category="rag",
                prompt=prompt,
                expected_answer=expected_answer,
                target_input_tokens=options.target_input_tokens,
                target_output_tokens=options.target_output_tokens,
                prefix_group_id=None,
                shared_prefix_tokens=0,
                eval_type="rag_placeholder",
                metadata={
                    "dataset": "beir_scifact",
                    "source": "beir_scifact",
                    "dataset_source_url": BEIR_SCIFACT_PAGE_URL,
                    "dataset_revision": options.dataset_revision,
                    "mode": options.mode,
                    "split": options.split,
                    "source_record_id": query_id,
                    "source_document_id": expected_answer,
                    "source_question_id": query_id,
                    "source_id": expected_answer,
                    "source_ids": source_ids,
                    "prefix_hash": sha256_text(evidence),
                    "prompt_hash": sha256_text(prompt),
                    "expected_answer_hash": sha256_text(expected_answer),
                    "tokenizer_id": options.tokenizer_id,
                    "tokenizer_revision": options.tokenizer_revision,
                    "token_count_method": options.token_count_method,
                    "target_input_tokens": options.target_input_tokens,
                    "target_output_tokens": options.target_output_tokens,
                    "measured_input_tokens": measured_input_tokens,
                    "measured_shared_prefix_tokens": 0,
                    "measured_suffix_tokens": measured_input_tokens,
                    "context_bucket": options.target_input_tokens,
                    "prompt_template": "beir_scifact_rag_v1",
                    "prompt_template_hash": sha256_text(BEIR_TEMPLATE),
                    "adapter": "beir_scifact",
                    "adapter_version": "0.1.0",
                    "truncated": count_tokens("\n\n".join(evidence_parts)) > options.target_input_tokens,
                    "truncation_policy": "tail",
                    "excluded_reason": None,
                    "answer_type": "source_id",
                    "evaluator": "rag_placeholder",
                    "evaluator_version": "0.1.0",
                    "redistributable_prompt": False,
                    "redistributable_output": False,
                    "rights_note": "Use upstream BEIR/SciFact dataset terms and attribution.",
                    "license_review_status": _license_review_status(options),
                    "redistribution_policy": _redistribution_policy(options),
                },
            )
        )
    return items, exclusions


def _load_beir_dataset(source_path: Path, split: str) -> dict[str, Any]:
    root = _resolve_beir_root(source_path)
    corpus = _load_corpus(root)
    queries = _load_queries(root)
    qrels = _load_qrels(root, split)
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def _resolve_beir_root(path: Path) -> Path:
    candidates = [path, path / "scifact"]
    for candidate in candidates:
        if (candidate / "corpus.json").exists() or (candidate / "corpus.jsonl").exists():
            return candidate
    raise ValueError(f"Could not find BEIR SciFact files under {path}")


def _load_corpus(root: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json_or_jsonl(root / "corpus.json", root / "corpus.jsonl")
    return {str(row.get("_id") or row.get("id")): row for row in rows}


def _load_queries(root: Path) -> dict[str, str]:
    rows = _read_json_or_jsonl(root / "queries.json", root / "queries.jsonl")
    return {
        str(row.get("_id") or row.get("id")): str(row.get("text") or row.get("query") or "")
        for row in rows
    }


def _load_qrels(root: Path, split: str) -> dict[str, list[str]]:
    json_path = root / "qrels.json"
    if json_path.exists():
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        qrels: dict[str, list[str]] = {}
        for row in rows:
            qrels.setdefault(str(row["query_id"]), []).append(str(row["doc_id"]))
        return qrels

    tsv_path = root / "qrels" / f"{split}.tsv"
    if not tsv_path.exists():
        matches = sorted((root / "qrels").glob("*.tsv")) if (root / "qrels").exists() else []
        if not matches:
            raise ValueError(f"Could not find BEIR qrels for split '{split}'")
        tsv_path = matches[0]
    qrels: dict[str, list[str]] = {}
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lower().startswith("query-id"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        query_id, doc_id = parts[0], parts[1]
        qrels.setdefault(query_id, []).append(doc_id)
    return qrels


def _read_json_or_jsonl(json_path: Path, jsonl_path: Path) -> list[dict[str, Any]]:
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records") or payload.get("data") or list(payload.values())
        return [dict(row) for row in payload]
    if jsonl_path.exists():
        return [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError(f"Expected {json_path.name} or {jsonl_path.name}")


def _options_with_download_defaults(options: DatasetPrepareOptions) -> DatasetPrepareOptions:
    if options.download and options.max_items is None:
        return options.model_copy(update={"max_items": BEIR_DOWNLOAD_DEFAULT_MAX_ITEMS})
    return options


def _resolve_source_path(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    if options.download:
        return _download_beir_source(options)
    if options.source_path is None:
        raise ValueError("BEIR SciFact adapter requires --source-path or --download")
    return options.source_path, {}


def _download_beir_source(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    cache_dir = resolve_dataset_cache_dir("beir_scifact", options.cache_dir)
    zip_path = cache_dir / "scifact.zip"
    extract_dir = cache_dir / "scifact"
    if extract_dir.exists() and not options.force:
        return _resolve_beir_root(extract_dir), {
            "cache_path": str(extract_dir),
            "download_method": "beir_public_zip",
            "downloaded_at": None,
            "source_url": BEIR_SCIFACT_SOURCE_URL,
        }
    result = download_file(BEIR_SCIFACT_SOURCE_URL, zip_path, force=options.force)
    extract_zip(result.path, extract_dir)
    return _resolve_beir_root(extract_dir), {
        "cache_path": str(extract_dir),
        "download_method": "beir_public_zip",
        "downloaded_at": result.downloaded_at or utc_now_iso(),
        "source_url": BEIR_SCIFACT_SOURCE_URL,
    }


def _license_review_status(options: DatasetPrepareOptions) -> str:
    return "upstream_terms_need_user_review" if options.download else "fixture_only"


def _redistribution_policy(options: DatasetPrepareOptions) -> str:
    return "generated_workloads_not_redistributable" if options.download else "tiny_fixture_allowed"


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
