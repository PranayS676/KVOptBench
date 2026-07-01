"""Manifest-oriented strategy planning and execution helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kvoptbench.config import load_config
from kvoptbench.engines.profiles import get_engine_profile
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.runner.schedule import ScheduledRun, build_schedule
from kvoptbench.schemas import ExperimentConfig


DEFAULT_WORKLOAD_ROLE = "primary"
CACHE_WORKLOAD_ROLES = ("shared_prefix", "random_prefix")
CACHE_PASSES = ("cold", "warm")


@dataclass(frozen=True, slots=True)
class StrategyPlanResult:
    """Files produced by strategy-plan."""

    plan_dir: Path
    manifest_path: Path
    config_paths: list[Path]


@dataclass(frozen=True, slots=True)
class StrategyRunResult:
    """Files produced by strategy-run."""

    run_manifest_path: Path
    output_paths: list[Path]


def write_strategy_plan(
    *,
    plan_dir: str | Path,
    matrix_id: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    workload_pack: str | Path,
    strategy_families: Iterable[str],
    strategies: Iterable[str],
    concurrencies: Iterable[int],
    output_dir: str | Path,
    repeat_count: int = 1,
    randomization_seed: int = 0,
    run_label: str = "exploratory",
    max_output_tokens: int = 256,
    endpoint_type: str | None = None,
    stream: bool = True,
) -> StrategyPlanResult:
    """Write experiment configs and a matrix manifest for strategy experiments."""
    plan_path = Path(plan_dir)
    plan_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir)
    profile = get_engine_profile(engine)
    workloads = read_workload_pack(workload_pack)
    normalized_families = _normalized_values(strategy_families, field="strategy_families")
    normalized_strategies = _normalized_values(strategies, field="strategies")
    concurrency_values = [int(value) for value in concurrencies]
    if not concurrency_values or any(value < 1 for value in concurrency_values):
        raise ValueError("At least one positive concurrency is required")

    _validate_strategies(profile.engine, normalized_strategies)
    cases = _expand_strategy_cases(
        matrix_id=matrix_id,
        provider=provider,
        engine=profile.engine,
        endpoint_type=endpoint_type or profile.engine,
        model_id=model_id,
        base_url=base_url,
        workloads=workloads,
        strategy_families=normalized_families,
        strategies=normalized_strategies,
        concurrencies=concurrency_values,
        output_dir=output_path,
        max_output_tokens=max_output_tokens,
        run_label=run_label,
        stream=stream,
    )

    config_paths: list[Path] = []
    manifest_entries: list[dict[str, Any]] = []
    for config in cases:
        config_path = plan_path / f"{config.experiment_id}.yaml"
        config_path.write_text(
            yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False),
            encoding="utf-8",
        )
        config_hash = _sha256_file(config_path)
        config_paths.append(config_path)
        manifest_entries.append(
            {
                "experiment_id": config.experiment_id,
                "config_path": _relative_or_name(config_path, plan_path),
                "config_sha256": config_hash,
                "output_file": config.output_file.as_posix(),
                "strategy_family": config.metadata.get("strategy_family"),
                "strategy": config.strategy,
                "workload_role": config.metadata.get("workload_role"),
                "cache_pass": config.metadata.get("cache_pass"),
                "concurrency": config.concurrency,
            }
        )

    manifest = {
        "schema_version": "1",
        "matrix_id": matrix_id,
        "provider": provider,
        "engine": profile.engine,
        "model_id": model_id,
        "base_url": base_url,
        "workload_pack": Path(workload_pack).name,
        "strategy_families": normalized_families,
        "strategies": normalized_strategies,
        "concurrencies": concurrency_values,
        "repeat_count": repeat_count,
        "randomization_seed": randomization_seed,
        "run_label": run_label,
        "config_count": len(config_paths),
        "configs": manifest_entries,
    }
    manifest_path = plan_path / "plan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return StrategyPlanResult(
        plan_dir=plan_path,
        manifest_path=manifest_path,
        config_paths=sorted(config_paths),
    )


def run_strategy_plan(
    *,
    matrix_manifest: str | Path,
    output_run_manifest: str | Path | None = None,
    repeat_count: int | None = None,
    randomization_seed: int | None = None,
    randomize: bool = False,
    dry_run: bool = False,
    runner: Callable[[Path], Awaitable[Path]] = run_experiment,
) -> StrategyRunResult:
    """Execute or dry-run a planned strategy matrix through a deterministic schedule."""
    manifest_path = Path(matrix_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_paths = _manifest_config_paths(manifest, manifest_path.parent)
    repeats = repeat_count or int(manifest.get("repeat_count") or 1)
    seed = randomization_seed if randomization_seed is not None else int(
        manifest.get("randomization_seed") or 0
    )
    schedule = build_schedule(config_paths, repeat_count=repeats, seed=seed, randomize=randomize)
    run_manifest_path = Path(output_run_manifest) if output_run_manifest else (
        manifest_path.parent / "run_manifest.json"
    )
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    if not dry_run:
        outputs = _run_schedule_with_metadata(schedule, runner=runner)

    run_manifest = {
        "schema_version": "1",
        "matrix_id": manifest.get("matrix_id"),
        "matrix_manifest": manifest_path.name,
        "dry_run": dry_run,
        "repeat_count": repeats,
        "randomization_seed": seed,
        "randomized_order": randomize,
        "schedule_id": schedule[0].schedule_id if schedule else None,
        "planned_runs": [
            {
                **scheduled.metadata,
                "config_path": _relative_or_name(scheduled.config_path, manifest_path.parent),
            }
            for scheduled in schedule
        ],
        "outputs": [path.as_posix() for path in outputs],
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    return StrategyRunResult(run_manifest_path=run_manifest_path, output_paths=outputs)


def read_workload_pack(path: str | Path) -> dict[str, Path]:
    """Read a workload-pack YAML file into role -> workload path mappings."""
    pack_path = Path(path)
    payload = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Workload pack must be a YAML mapping")
    raw_workloads = payload.get("workloads", payload)
    if not isinstance(raw_workloads, dict):
        raise ValueError("Workload pack workloads must be a mapping")
    workloads: dict[str, Path] = {}
    for role, value in raw_workloads.items():
        if role in {"schema_version", "metadata", "name", "description"}:
            continue
        if isinstance(value, dict):
            raw_path = value.get("path") or value.get("workload_file")
        else:
            raw_path = value
        if raw_path is None:
            continue
        workloads[str(role).strip()] = Path(str(raw_path))
    if not workloads:
        raise ValueError("Workload pack must define at least one workload path")
    return workloads


def _expand_strategy_cases(
    *,
    matrix_id: str,
    provider: str,
    engine: str,
    endpoint_type: str,
    model_id: str,
    base_url: str,
    workloads: dict[str, Path],
    strategy_families: list[str],
    strategies: list[str],
    concurrencies: list[int],
    output_dir: Path,
    max_output_tokens: int,
    run_label: str,
    stream: bool,
) -> list[ExperimentConfig]:
    cases: list[ExperimentConfig] = []
    for family in strategy_families:
        roles = _roles_for_family(family, workloads)
        cache_passes = CACHE_PASSES if family == "cache" else ("measured",)
        for strategy in strategies:
            for role in roles:
                workload_file = workloads[role]
                for concurrency in concurrencies:
                    for cache_pass in cache_passes:
                        parts = [
                            matrix_id,
                            family,
                            strategy,
                            role,
                            f"c{concurrency}",
                        ]
                        if family == "cache":
                            parts.append(cache_pass)
                        experiment_id = _safe_identifier("_".join(parts))
                        metadata = {
                            "strategy_matrix_id": matrix_id,
                            "strategy_family": family,
                            "workload_role": role,
                            "run_label": run_label,
                        }
                        if family == "cache":
                            metadata["cache_experiment"] = True
                            metadata["cache_pass"] = cache_pass
                            metadata["control_workload"] = role == "random_prefix"
                            metadata["workload_profile"] = role
                        cases.append(
                            ExperimentConfig(
                                experiment_id=experiment_id,
                                official_run=False,
                                provider=provider,
                                engine=engine,
                                endpoint_type=endpoint_type,  # type: ignore[arg-type]
                                model_id=model_id,
                                strategy=strategy,
                                base_url=base_url,
                                workload_file=workload_file,
                                output_file=output_dir / f"{experiment_id}.jsonl",
                                concurrency=concurrency,
                                max_output_tokens=max_output_tokens,
                                stream=stream,
                                metadata=metadata,
                            )
                        )
    return cases


def _run_schedule_with_metadata(
    schedule: list[ScheduledRun],
    *,
    runner: Callable[[Path], Awaitable[Path]],
) -> list[Path]:
    async def _run_all() -> list[Path]:
        outputs: list[Path] = []
        for scheduled in schedule:
            config = load_config(scheduled.config_path)
            scheduled_config_path = _write_scheduled_config(scheduled, config)
            outputs.append(await runner(scheduled_config_path))
        return outputs

    return asyncio.run(_run_all())


def _write_scheduled_config(scheduled: ScheduledRun, config: ExperimentConfig) -> Path:
    scheduled_dir = scheduled.config_path.parent / ".strategy_run"
    scheduled_dir.mkdir(parents=True, exist_ok=True)
    scheduled_config = config.model_copy(
        update={
            "metadata": {
                **config.metadata,
                "schedule": scheduled.metadata,
                "schedule_id": scheduled.schedule_id,
                "run_group_id": scheduled.schedule_id,
                "order_index": scheduled.order_index,
                "trial_index": scheduled.trial_index,
                "repeat_index": scheduled.repeat_index,
                "repeat_count": scheduled.repeat_count,
                "randomization_seed": scheduled.seed,
                "randomized_order": scheduled.randomize,
                "warmup": False,
            }
        }
    )
    path = scheduled_dir / (
        f"{scheduled.order_index:04d}-r{scheduled.repeat_index}-"
        f"{scheduled.config_path.name}"
    )
    path.write_text(
        yaml.safe_dump(_config_to_yaml_dict(scheduled_config), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _roles_for_family(family: str, workloads: dict[str, Path]) -> list[str]:
    if family == "cache":
        missing = [role for role in CACHE_WORKLOAD_ROLES if role not in workloads]
        if missing:
            raise ValueError(
                "Cache strategy plans require workload roles: " + ", ".join(missing)
            )
        return list(CACHE_WORKLOAD_ROLES)
    if DEFAULT_WORKLOAD_ROLE in workloads:
        return [DEFAULT_WORKLOAD_ROLE]
    return sorted(workloads)


def _validate_strategies(engine: str, strategies: list[str]) -> None:
    profile = get_engine_profile(engine)
    unknown = [strategy for strategy in strategies if strategy not in profile.strategies]
    if unknown:
        valid = ", ".join(sorted(profile.strategies))
        raise ValueError(
            f"Unknown strategies for engine '{profile.engine}': "
            f"{', '.join(unknown)}. Valid strategies: {valid}"
        )


def _normalized_values(values: Iterable[str], *, field: str) -> list[str]:
    normalized = [value.strip().lower() for value in values if value and value.strip()]
    if not normalized:
        raise ValueError(f"At least one {field} value is required")
    return normalized


def _safe_identifier(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _manifest_config_paths(manifest: dict[str, Any], base_dir: Path) -> list[Path]:
    configs = manifest.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Matrix manifest does not contain config entries")
    paths: list[Path] = []
    for entry in configs:
        if not isinstance(entry, dict) or "config_path" not in entry:
            raise ValueError("Each matrix manifest config entry must include config_path")
        config_path = Path(str(entry["config_path"]))
        paths.append(config_path if config_path.is_absolute() else base_dir / config_path)
    return paths


def _config_to_yaml_dict(config: ExperimentConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    return {key: value for key, value in data.items() if value is not None}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_name(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.name
