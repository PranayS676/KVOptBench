"""Prefill/decode decomposition experiment plan helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from pydantic import BaseModel

from kvoptbench.config import load_config
from kvoptbench.engines.profiles import get_engine_profile
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import ExperimentConfig

DEFAULT_PREFILL_DECODE_STRATEGIES = ("baseline",)


class PrefillDecodeExperimentCase(BaseModel):
    """One generated prefill/decode experiment config."""

    strategy: str
    workload_profile: str = "prefill_decode_grid"
    config: ExperimentConfig


def build_prefill_decode_plan(
    *,
    experiment_prefix: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    workload_file: str | Path,
    output_dir: str | Path,
    strategies: Iterable[str] = DEFAULT_PREFILL_DECODE_STRATEGIES,
    concurrency: int = 1,
    max_output_tokens: int = 512,
    stream: bool = True,
) -> list[PrefillDecodeExperimentCase]:
    """Build configs for a prefill/decode workload grid."""
    profile = get_engine_profile(engine)
    output_dir = Path(output_dir)
    cases: list[PrefillDecodeExperimentCase] = []
    for strategy in strategies:
        normalized_strategy = strategy.strip().lower()
        if normalized_strategy not in profile.strategies:
            valid = ", ".join(sorted(profile.strategies))
            raise ValueError(
                f"Unknown strategy '{strategy}' for engine '{profile.engine}'. Valid strategies: {valid}"
            )
        experiment_id = f"{experiment_prefix}_{profile.engine}_{normalized_strategy}_grid"
        config = ExperimentConfig(
            experiment_id=experiment_id,
            official_run=False,
            provider=provider,
            engine=profile.engine,
            endpoint_type=profile.engine,
            model_id=model_id,
            strategy=normalized_strategy,
            base_url=base_url,
            workload_file=Path(workload_file),
            output_file=output_dir / f"{experiment_id}.jsonl",
            concurrency=concurrency,
            max_output_tokens=max_output_tokens,
            stream=stream,
            metadata={
                "prefill_decode_experiment": True,
                "workload_profile": "prefill_decode_grid",
            },
        )
        cases.append(PrefillDecodeExperimentCase(strategy=normalized_strategy, config=config))
    return cases


def write_prefill_decode_plan_configs(
    *,
    plan_dir: str | Path,
    experiment_prefix: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    workload_file: str | Path,
    output_dir: str | Path,
    strategies: Iterable[str] = DEFAULT_PREFILL_DECODE_STRATEGIES,
    concurrency: int = 1,
    max_output_tokens: int = 512,
    stream: bool = True,
) -> list[Path]:
    """Write prefill/decode experiment configs as YAML files."""
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    cases = build_prefill_decode_plan(
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        strategies=strategies,
        concurrency=concurrency,
        max_output_tokens=max_output_tokens,
        stream=stream,
    )
    plan_id = f"{experiment_prefix}_{get_engine_profile(engine).engine}"
    written: list[Path] = []
    for case in cases:
        config = case.config.model_copy(
            update={
                "metadata": {
                    **case.config.metadata,
                    "prefill_decode_plan_id": plan_id,
                }
            }
        )
        path = plan_dir / f"{config.experiment_id}.yaml"
        path.write_text(yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8")
        written.append(path)
    return sorted(written)


def run_prefill_decode_plan(
    plan_dir: str | Path,
    *,
    runner: Callable[[Path], Awaitable[Path]] = run_experiment,
) -> list[Path]:
    """Run all YAML configs in a prefill/decode plan directory in deterministic order."""
    plan_path = Path(plan_dir)
    config_files = sorted(plan_path.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"No prefill/decode plan YAML configs found under {plan_path}")

    async def _run_all() -> list[Path]:
        outputs: list[Path] = []
        for config_file in config_files:
            load_config(config_file)
            outputs.append(await runner(config_file))
        return outputs

    return asyncio.run(_run_all())


def _config_to_yaml_dict(config: ExperimentConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    return {key: value for key, value in data.items() if value is not None}
