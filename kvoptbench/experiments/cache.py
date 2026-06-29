"""Cache ablation experiment plan helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Awaitable, Callable, Any

import yaml

from kvoptbench.config import load_config
from kvoptbench.engines.profiles import get_engine_profile
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import CacheExperimentCase, ExperimentConfig

DEFAULT_CACHE_STRATEGIES = ("cache_off", "cache_on")
CACHE_PASSES = ("cold", "warm")


def build_cache_ablation_plan(
    *,
    experiment_prefix: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    shared_workload_file: str | Path,
    random_workload_file: str | Path,
    output_dir: str | Path,
    strategies: Iterable[str] = DEFAULT_CACHE_STRATEGIES,
    concurrency: int = 1,
    max_output_tokens: int = 256,
    stream: bool = True,
) -> list[CacheExperimentCase]:
    """Build cold/warm shared-prefix and random-control configs for cache ablation."""
    profile = get_engine_profile(engine)
    output_dir = Path(output_dir)
    workload_paths = {
        "shared_prefix": Path(shared_workload_file),
        "random_prefix": Path(random_workload_file),
    }

    cases: list[CacheExperimentCase] = []
    for strategy in strategies:
        normalized_strategy = strategy.strip().lower()
        if normalized_strategy not in profile.strategies:
            valid = ", ".join(sorted(profile.strategies))
            raise ValueError(
                f"Unknown strategy '{strategy}' for engine '{profile.engine}'. Valid strategies: {valid}"
            )
        for workload_profile, workload_file in workload_paths.items():
            is_control = workload_profile == "random_prefix"
            for cache_pass in CACHE_PASSES:
                experiment_id = (
                    f"{experiment_prefix}_{profile.engine}_{normalized_strategy}_"
                    f"{workload_profile}_{cache_pass}"
                )
                config = ExperimentConfig(
                    experiment_id=experiment_id,
                    official_run=False,
                    provider=provider,
                    engine=profile.engine,
                    endpoint_type=profile.engine,
                    model_id=model_id,
                    strategy=normalized_strategy,
                    base_url=base_url,
                    workload_file=workload_file,
                    output_file=output_dir / f"{experiment_id}.jsonl",
                    concurrency=concurrency,
                    max_output_tokens=max_output_tokens,
                    stream=stream,
                    metadata={
                        "cache_experiment": True,
                        "cache_pass": cache_pass,
                        "workload_profile": workload_profile,
                        "control_workload": is_control,
                    },
                )
                cases.append(
                    CacheExperimentCase(
                        strategy=normalized_strategy,
                        workload_profile=workload_profile,
                        cache_pass=cache_pass,
                        is_control=is_control,
                        config=config,
                    )
                )
    return cases


def write_cache_plan_configs(
    *,
    plan_dir: str | Path,
    experiment_prefix: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    shared_workload_file: str | Path,
    random_workload_file: str | Path,
    output_dir: str | Path,
    strategies: Iterable[str] = DEFAULT_CACHE_STRATEGIES,
    concurrency: int = 1,
    max_output_tokens: int = 256,
    stream: bool = True,
) -> list[Path]:
    """Write cache ablation experiment configs as YAML files."""
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cache_ablation_plan(
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        shared_workload_file=shared_workload_file,
        random_workload_file=random_workload_file,
        output_dir=output_dir,
        strategies=strategies,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
        stream=stream,
    )
    cache_plan_id = f"{experiment_prefix}_{get_engine_profile(engine).engine}"
    written: list[Path] = []
    for case in cases:
        config = case.config.model_copy(
            update={
                "metadata": {
                    **case.config.metadata,
                    "cache_plan_id": cache_plan_id,
                }
            }
        )
        path = plan_dir / f"{config.experiment_id}.yaml"
        path.write_text(yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8")
        written.append(path)
    return sorted(written)


def run_cache_plan(
    plan_dir: str | Path,
    *,
    runner: Callable[[Path], Awaitable[Path]] = run_experiment,
) -> list[Path]:
    """Run all YAML configs in a cache plan directory in deterministic order."""
    plan_path = Path(plan_dir)
    config_files = sorted(plan_path.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"No cache plan YAML configs found under {plan_path}")

    async def _run_all() -> list[Path]:
        outputs: list[Path] = []
        for config_file in config_files:
            # Validate before execution so config errors identify the YAML file.
            load_config(config_file)
            outputs.append(await runner(config_file))
        return outputs

    return asyncio.run(_run_all())


def _config_to_yaml_dict(config: ExperimentConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    return {key: value for key, value in data.items() if value is not None}
