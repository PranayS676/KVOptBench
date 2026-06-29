"""KV cache quantization experiment plan helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import yaml
from pydantic import BaseModel

from kvoptbench.config import load_config
from kvoptbench.engines.profiles import get_engine_profile
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import ExperimentConfig

DEFAULT_KV_QUANTIZATION_STRATEGIES = ("baseline", "kv_fp8")
DEFAULT_KV_QUANTIZATION_WORKLOAD_PROFILE = "long_context_pressure"


class KVQuantizationExperimentCase(BaseModel):
    """One generated KV quantization experiment config."""

    strategy: str
    role: Literal["control", "quantized"]
    workload_profile: str = DEFAULT_KV_QUANTIZATION_WORKLOAD_PROFILE
    config: ExperimentConfig


def build_kv_quantization_plan(
    *,
    experiment_prefix: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    workload_file: str | Path,
    output_dir: str | Path,
    workload_profile: str = DEFAULT_KV_QUANTIZATION_WORKLOAD_PROFILE,
    strategies: Iterable[str] = DEFAULT_KV_QUANTIZATION_STRATEGIES,
    concurrency: int = 1,
    max_output_tokens: int = 256,
    stream: bool = True,
) -> list[KVQuantizationExperimentCase]:
    """Build baseline and quantized KV cache experiment configs."""
    profile = get_engine_profile(engine)
    output_dir = Path(output_dir)
    normalized_strategies = tuple(strategy.strip().lower() for strategy in strategies)
    cases: list[KVQuantizationExperimentCase] = []
    for strategy in normalized_strategies:
        if strategy not in profile.strategies:
            valid = ", ".join(sorted(profile.strategies))
            raise ValueError(
                f"Unknown strategy '{strategy}' for engine '{profile.engine}'. Valid strategies: {valid}"
            )
        role: Literal["control", "quantized"] = "control" if strategy == "baseline" else "quantized"
        experiment_id = f"{experiment_prefix}_{profile.engine}_{strategy}_{workload_profile}"
        config = ExperimentConfig(
            experiment_id=experiment_id,
            official_run=False,
            provider=provider,
            engine=profile.engine,
            endpoint_type=profile.engine,
            model_id=model_id,
            strategy=strategy,
            base_url=base_url,
            workload_file=Path(workload_file),
            output_file=output_dir / f"{experiment_id}.jsonl",
            concurrency=concurrency,
            max_output_tokens=max_output_tokens,
            stream=stream,
            metadata={
                "kv_quantization_experiment": True,
                "kv_quantization_role": role,
                "control_strategy": "baseline",
                "quantized_strategy": "kv_fp8",
                "workload_profile": workload_profile,
            },
        )
        cases.append(
            KVQuantizationExperimentCase(
                strategy=strategy,
                role=role,
                workload_profile=workload_profile,
                config=config,
            )
        )
    return cases


def write_kv_quantization_plan_configs(
    *,
    plan_dir: str | Path,
    experiment_prefix: str,
    provider: str,
    engine: str,
    model_id: str,
    base_url: str,
    workload_file: str | Path,
    output_dir: str | Path,
    workload_profile: str = DEFAULT_KV_QUANTIZATION_WORKLOAD_PROFILE,
    strategies: Iterable[str] = DEFAULT_KV_QUANTIZATION_STRATEGIES,
    concurrency: int = 1,
    max_output_tokens: int = 256,
    stream: bool = True,
) -> list[Path]:
    """Write KV quantization experiment configs as YAML files."""
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    cases = build_kv_quantization_plan(
        experiment_prefix=experiment_prefix,
        provider=provider,
        engine=engine,
        model_id=model_id,
        base_url=base_url,
        workload_file=workload_file,
        output_dir=output_dir,
        workload_profile=workload_profile,
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
                    "kv_quantization_plan_id": plan_id,
                }
            }
        )
        path = plan_dir / f"{config.experiment_id}.yaml"
        path.write_text(yaml.safe_dump(_config_to_yaml_dict(config), sort_keys=False), encoding="utf-8")
        written.append(path)
    return sorted(written)


def run_kv_quantization_plan(
    plan_dir: str | Path,
    *,
    runner: Callable[[Path], Awaitable[Path]] = run_experiment,
) -> list[Path]:
    """Run all YAML configs in a KV quantization plan directory in deterministic order."""
    plan_path = Path(plan_dir)
    config_files = sorted(plan_path.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"No KV quantization plan YAML configs found under {plan_path}")

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
