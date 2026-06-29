from pathlib import Path

import yaml

from kvoptbench.config import load_config
from kvoptbench.experiments.cache import (
    build_cache_ablation_plan,
    run_cache_plan,
    write_cache_plan_configs,
)


def test_cache_ablation_plan_includes_cold_warm_shared_and_random_controls() -> None:
    cases = build_cache_ablation_plan(
        experiment_prefix="cache_exp",
        provider="local",
        engine="vllm",
        model_id="example/model",
        base_url="http://127.0.0.1:8000/v1",
        shared_workload_file=Path("workloads/generated/shared.jsonl"),
        random_workload_file=Path("workloads/generated/random.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert len(cases) == 8
    assert {case.strategy for case in cases} == {"cache_off", "cache_on"}
    assert {case.cache_pass for case in cases} == {"cold", "warm"}
    assert {case.workload_profile for case in cases} == {"shared_prefix", "random_prefix"}
    assert all(case.config.engine == "vllm" for case in cases)
    assert all(case.config.endpoint_type == "vllm" for case in cases)
    assert all(case.config.metadata["cache_experiment"] is True for case in cases)
    assert all(case.config.output_file.parent == Path("results/raw") for case in cases)
    assert len({case.config.experiment_id for case in cases}) == len(cases)


def test_cache_ablation_plan_marks_random_prefix_as_required_control() -> None:
    cases = build_cache_ablation_plan(
        experiment_prefix="cache_exp",
        provider="local",
        engine="sglang",
        model_id="example/model",
        base_url="http://127.0.0.1:30000/v1",
        shared_workload_file=Path("shared.jsonl"),
        random_workload_file=Path("random.jsonl"),
        output_dir=Path("results/raw"),
        strategies=("cache_on",),
    )

    random_cases = [case for case in cases if case.workload_profile == "random_prefix"]
    shared_cases = [case for case in cases if case.workload_profile == "shared_prefix"]

    assert len(random_cases) == 2
    assert len(shared_cases) == 2
    assert all(case.is_control for case in random_cases)
    assert not any(case.is_control for case in shared_cases)
    assert all(case.config.endpoint_type == "sglang" for case in cases)


def test_write_cache_plan_configs_creates_runnable_yaml_files(tmp_path: Path) -> None:
    written = write_cache_plan_configs(
        plan_dir=tmp_path / "plan",
        experiment_prefix="cache_exp",
        provider="mock",
        engine="vllm",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:18000/v1",
        shared_workload_file=Path("workloads/generated/shared.jsonl"),
        random_workload_file=Path("workloads/generated/random.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert len(written) == 8
    assert all(path.suffix == ".yaml" for path in written)
    loaded = [load_config(path) for path in written]
    assert {config.strategy for config in loaded} == {"cache_off", "cache_on"}
    assert {config.metadata["cache_pass"] for config in loaded} == {"cold", "warm"}
    assert {config.metadata["workload_profile"] for config in loaded} == {
        "shared_prefix",
        "random_prefix",
    }
    assert all(config.metadata["cache_plan_id"] == "cache_exp_vllm" for config in loaded)
    raw = yaml.safe_load(written[0].read_text(encoding="utf-8"))
    assert raw["endpoint_type"] == "vllm"


def test_run_cache_plan_executes_configs_in_sorted_order(tmp_path: Path) -> None:
    calls: list[Path] = []
    first = tmp_path / "plan" / "b.yaml"
    second = tmp_path / "plan" / "a.yaml"
    first.parent.mkdir()
    base_config = "\n".join(
        [
            "provider: mock",
            "engine: mock",
            "model_id: mock-frontier-model",
            "strategy: cache_on",
            "base_url: http://127.0.0.1:8000/v1",
            "workload_file: workloads/generated/shared.jsonl",
            "output_file: results/raw/out.jsonl",
        ]
    )
    first.write_text(f"experiment_id: b\n{base_config}\n", encoding="utf-8")
    second.write_text(f"experiment_id: a\n{base_config}\n", encoding="utf-8")

    async def fake_runner(path: Path) -> Path:
        calls.append(path)
        return Path("results") / f"{path.stem}.jsonl"

    outputs = run_cache_plan(first.parent, runner=fake_runner)

    assert calls == [second, first]
    assert outputs == [Path("results/a.jsonl"), Path("results/b.jsonl")]

