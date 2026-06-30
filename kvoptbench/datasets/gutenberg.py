"""Project Gutenberg-style adapter for deterministic long-context needle workloads."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kvoptbench.datasets.cache import resolve_dataset_cache_dir, write_json_payload
from kvoptbench.datasets.download import download_first_available
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
from kvoptbench.datasets.token_counting import count_tokens, truncate_to_token_budget
from kvoptbench.schemas import WorkloadItem, utc_now_iso

GUTENBERG_SOURCE_URL = "https://www.gutenberg.org/"
DEFAULT_GUTENBERG_BOOK_IDS = ("1342", "84", "2701")
GUTENBERG_TEMPLATE = (
    "Read the context and answer the needle question. If no needle answer exists, "
    "answer NO_NEEDLE_PRESENT.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
POSITIONS = ("beginning", "middle", "end")


class GutenbergAdapter:
    """Prepare deterministic needle workloads from local long-document text fixtures."""

    info = DatasetAdapterInfo(
        name="gutenberg",
        dataset_name="Project Gutenberg",
        source_url=GUTENBERG_SOURCE_URL,
        supported_modes=("needle", "no_needle_control", "multi_needle", "conflicting_needle"),
        license=None,
        rights_note="Record rights notes per book; do not commit full Project Gutenberg texts.",
        download_supported=True,
        fixture_supported=True,
    )

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        if options.mode not in self.info.supported_modes:
            raise ValueError(f"Gutenberg adapter does not support mode '{options.mode}'")
        run_options = _options_with_download_defaults(options)
        source_path, download_meta = _resolve_source_path(run_options)

        started = time.perf_counter()
        started_at = utc_now_iso()
        books = _load_books(source_path, run_options.book_ids)
        items = _generate_items(books, run_options)
        if run_options.max_items is not None:
            items = items[: run_options.max_items]
        if not items:
            raise ValueError("Gutenberg adapter produced no workload rows")

        workload_sha256 = write_workload_jsonl(items, run_options.out)
        finished_at = utc_now_iso()
        manifest = build_manifest(
            info=self.info,
            options=run_options,
            items=items,
            workload_sha256=workload_sha256,
            prompt_template=f"gutenberg_{run_options.mode}_v1",
            prompt_template_text=GUTENBERG_TEMPLATE,
            license_review_status=_license_review_status(run_options),
            redistribution_policy=_redistribution_policy(run_options),
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
                (
                    "Downloaded Project Gutenberg texts are cached locally; do not commit "
                    "full text files."
                )
                if run_options.download
                else "Local fixture/source mode does not perform network download.",
                "Rights notes are manifest metadata; users must review rights for real books.",
            ],
        )
        write_manifest(manifest, run_options.manifest)
        return DatasetPrepareResult(
            output_path=run_options.out,
            manifest_path=run_options.manifest,
            row_count=len(items),
            workload_sha256=workload_sha256,
        )


def _generate_items(books: list[dict[str, Any]], options: DatasetPrepareOptions) -> list[WorkloadItem]:
    buckets = options.context_buckets or (options.target_input_tokens,)
    items: list[WorkloadItem] = []
    for book in books:
        for bucket in buckets:
            for position in _positions_for_mode(options.mode):
                items.append(_make_item(book=book, bucket=bucket, position=position, options=options))
    return items


def _make_item(
    *, book: dict[str, Any], bucket: int, position: str, options: DatasetPrepareOptions
) -> WorkloadItem:
    book_id = str(book["book_id"])
    expanded_source = _expand_text_to_bucket(str(book["text"]), bucket, options.token_count_method)
    base_context = truncate_to_token_budget(expanded_source, bucket, options.token_count_method)
    context, expected_answer, needle_id = _context_for_mode(
        base_context=base_context,
        book_id=book_id,
        bucket=bucket,
        position=position,
        options=options,
    )
    question = "What is the secret needle answer?"
    prompt = GUTENBERG_TEMPLATE.format(context=context, question=question)
    measured_input_tokens = count_tokens(prompt, options.token_count_method)
    metadata = {
        "dataset": "gutenberg",
        "source": "gutenberg",
        "dataset_source_url": GUTENBERG_SOURCE_URL,
        "dataset_revision": options.dataset_revision,
        "mode": options.mode,
        "source_record_id": book_id,
        "source_document_id": book_id,
        "source_question_id": needle_id,
        "book_id": book_id,
        "book_title": book["title"],
        "prefix_hash": sha256_text(context),
        "prompt_hash": sha256_text(prompt),
        "expected_answer_hash": sha256_text(expected_answer),
        "tokenizer_id": options.tokenizer_id,
        "tokenizer_revision": options.tokenizer_revision,
        "token_count_method": options.token_count_method,
        "target_input_tokens": bucket,
        "target_output_tokens": options.target_output_tokens,
        "measured_input_tokens": measured_input_tokens,
        "measured_shared_prefix_tokens": 0,
        "measured_suffix_tokens": measured_input_tokens,
        "context_bucket": bucket,
        "needle_id": needle_id,
        "needle_position_ratio": _position_ratio(position),
        "needle_position_bucket": position,
        "answer_type": "needle",
        "prompt_template": f"gutenberg_{options.mode}_v1",
        "prompt_template_hash": sha256_text(GUTENBERG_TEMPLATE),
        "adapter": "gutenberg",
        "adapter_version": "0.1.0",
        "truncated": count_tokens(expanded_source, options.token_count_method) > bucket,
        "truncation_policy": "tail",
        "excluded_reason": None,
        "evaluator": "needle" if options.mode != "no_needle_control" else "contains_expected",
        "evaluator_version": "0.1.0",
        "difficulty": None,
        "redistributable_prompt": False,
        "redistributable_output": False,
        "rights_note": book["rights_note"],
        "license_review_status": _license_review_status(options),
        "redistribution_policy": _redistribution_policy(options),
    }
    eval_type = "contains_expected" if options.mode == "no_needle_control" else "needle"
    return WorkloadItem(
        task_id=f"gutenberg_{options.mode}_{_slug(book_id)}_{bucket}_{position}",
        workload=f"gutenberg_{options.mode}",
        category="long_context",
        prompt=prompt,
        expected_answer=expected_answer,
        target_input_tokens=bucket,
        target_output_tokens=options.target_output_tokens,
        prefix_group_id=f"gutenberg_{book_id}" if options.mode != "no_needle_control" else None,
        shared_prefix_tokens=0,
        eval_type=eval_type,
        metadata=metadata,
    )


def _context_for_mode(
    *,
    base_context: str,
    book_id: str,
    bucket: int,
    position: str,
    options: DatasetPrepareOptions,
) -> tuple[str, str, str]:
    if options.mode == "no_needle_control":
        return base_context, "NO_NEEDLE_PRESENT", f"no_needle_{book_id}_{bucket}_{position}"

    answer = f"KVOB-{options.seed}-{book_id}-{bucket}-{position}"
    needle_id = f"needle_{book_id}_{bucket}_{position}_{options.seed}"
    if options.mode == "multi_needle":
        second = f"{answer}-SECOND"
        needle = f"\nNEEDLE {needle_id}: first answer {answer}; second answer {second}.\n"
        return _insert_text(base_context, needle, position), f"{answer}; {second}", needle_id
    if options.mode == "conflicting_needle":
        conflict = f"KVOB-CONFLICT-{options.seed}-{book_id}-{bucket}-{position}"
        needle = (
            f"\nNEEDLE {needle_id}: older answer {conflict}. "
            f"Use the newer answer {answer}.\n"
        )
        return _insert_text(base_context, needle, position), answer, needle_id

    needle = f"\nNEEDLE {needle_id}: the secret needle answer is {answer}.\n"
    return _insert_text(base_context, needle, position), answer, needle_id


def _insert_text(context: str, inserted: str, position: str) -> str:
    if position == "beginning":
        return inserted + context
    if position == "end":
        return context + inserted
    midpoint = len(context) // 2
    return context[:midpoint] + inserted + context[midpoint:]


def _positions_for_mode(mode: str) -> tuple[str, ...]:
    if mode in {"multi_needle", "conflicting_needle"}:
        return ("middle",)
    return POSITIONS


def _load_books(source_path: Path, requested_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    if not source_path.exists():
        raise ValueError(f"Gutenberg source path does not exist: {source_path}")
    if source_path.is_file():
        book_id = source_path.stem
        return [
            {
                "book_id": book_id,
                "title": source_path.stem,
                "rights_note": "Local source file; user must verify rights before publication.",
                "text": source_path.read_text(encoding="utf-8"),
            }
        ]

    manifest_path = source_path / "books.json"
    if not manifest_path.exists():
        raise ValueError("Gutenberg source directory must contain books.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_books = payload.get("books", [])
    requested = set(requested_ids)
    books: list[dict[str, Any]] = []
    for raw in raw_books:
        book_id = str(raw["book_id"])
        if requested and book_id not in requested:
            continue
        text_path = source_path / str(raw["file"])
        books.append(
            {
                "book_id": book_id,
                "title": str(raw.get("title") or book_id),
                "rights_note": str(raw.get("rights_note") or ""),
                "text": text_path.read_text(encoding="utf-8"),
            }
        )
    if not books:
        raise ValueError("No matching Gutenberg books found")
    return books


def _options_with_download_defaults(options: DatasetPrepareOptions) -> DatasetPrepareOptions:
    if options.download and not options.book_ids:
        return options.model_copy(update={"book_ids": DEFAULT_GUTENBERG_BOOK_IDS})
    return options


def _resolve_source_path(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    if options.download:
        return _download_gutenberg_source(options)
    if options.source_path is None:
        raise ValueError("Gutenberg adapter requires --source-path or --download")
    return options.source_path, {}


def _download_gutenberg_source(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    book_ids = options.book_ids or DEFAULT_GUTENBERG_BOOK_IDS
    source_dir = resolve_dataset_cache_dir("gutenberg", options.cache_dir) / "books"
    if _cached_books_available(source_dir, book_ids) and not options.force:
        return source_dir, {
            "cache_path": str(source_dir),
            "download_method": "project_gutenberg_http",
            "downloaded_at": None,
            "source_url": GUTENBERG_SOURCE_URL,
        }

    source_dir.mkdir(parents=True, exist_ok=True)
    books: list[dict[str, str]] = []
    downloaded_at: str | None = None
    for book_id in book_ids:
        text_path = source_dir / f"book_{book_id}.txt"
        result = download_first_available(
            _gutenberg_candidate_urls(book_id),
            text_path,
            force=options.force,
        )
        downloaded_at = downloaded_at or result.downloaded_at
        books.append(
            {
                "book_id": book_id,
                "title": f"Project Gutenberg book {book_id}",
                "file": text_path.name,
                "rights_note": (
                    "Project Gutenberg source; verify public-domain status and "
                    "distribution terms for your jurisdiction before publication."
                ),
                "source_url": result.url,
            }
        )
    write_json_payload(source_dir / "books.json", {"books": books}, force=True)
    return source_dir, {
        "cache_path": str(source_dir),
        "download_method": "project_gutenberg_http",
        "downloaded_at": downloaded_at or utc_now_iso(),
        "source_url": GUTENBERG_SOURCE_URL,
    }


def _cached_books_available(source_dir: Path, book_ids: tuple[str, ...]) -> bool:
    manifest_path = source_dir / "books.json"
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    books = payload.get("books", [])
    available = {str(book.get("book_id")): book for book in books if isinstance(book, dict)}
    for book_id in book_ids:
        book = available.get(str(book_id))
        if not book or not (source_dir / str(book.get("file", ""))).exists():
            return False
    return True


def _gutenberg_candidate_urls(book_id: str) -> tuple[str, ...]:
    return (
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    )


def _license_review_status(options: DatasetPrepareOptions) -> str:
    return "source_rights_need_user_review" if options.download else "fixture_only"


def _redistribution_policy(options: DatasetPrepareOptions) -> str:
    return "generated_workloads_not_redistributable" if options.download else "tiny_fixture_allowed"


def _expand_text_to_bucket(text: str, bucket: int, token_count_method: str) -> str:
    if count_tokens(text, token_count_method) >= bucket:
        return text
    if not text:
        return ""
    approx_target_chars = bucket * 4
    repeat_count = max(1, (approx_target_chars // len(text)) + 1)
    return (text + "\n") * repeat_count


def _position_ratio(position: str) -> float | None:
    if position == "beginning":
        return 0.0
    if position == "middle":
        return 0.5
    if position == "end":
        return 1.0
    return None


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
