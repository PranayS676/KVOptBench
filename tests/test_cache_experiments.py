from pathlib import Path

from kvoptbench.experiments.cache import build_cache_ablation_plan


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

