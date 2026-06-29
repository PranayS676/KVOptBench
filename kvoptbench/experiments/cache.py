"""Cache ablation experiment plan helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from kvoptbench.engines.profiles import get_engine_profile
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
