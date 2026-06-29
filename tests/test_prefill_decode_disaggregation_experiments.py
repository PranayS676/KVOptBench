from pathlib import Path

from kvoptbench.config import load_config
from kvoptbench.experiments.prefill_decode_disaggregation import (
    build_prefill_decode_disaggregation_plan,
    run_prefill_decode_disaggregation_plan,
    write_prefill_decode_disaggregation_plan_configs,
)


def test_disaggregation_plan_builds_baseline_and_disaggregated_runs() -> None:
    cases = build_prefill_decode_disaggregation_plan(
        experiment_prefix="disagg",
        provider="mock",
        engine="vllm",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:8000/v1",
        workload_file=Path("workloads/generated/prefill_decode_grid.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert [case.strategy for case in cases] == ["baseline", "prefill_decode_disaggregation"]
    assert {case.role for case in cases} == {"control", "disaggregated"}
    assert all(case.workload_profile == "prefill_decode_grid" for case in cases)
    assert all(case.config.endpoint_type == "vllm" for case in cases)
    assert all(
        case.config.metadata["prefill_decode_disaggregation_experiment"] is True
        for case in cases
    )
    assert cases[0].config.output_file == Path(
        "results/raw/disagg_vllm_baseline_prefill_decode_grid.jsonl"
    )
    assert cases[1].config.output_file == Path(
        "results/raw/disagg_vllm_prefill_decode_disaggregation_prefill_decode_grid.jsonl"
    )


def test_write_disaggregation_plan_configs_creates_runnable_yaml(tmp_path: Path) -> None:
    written = write_prefill_decode_disaggregation_plan_configs(
        plan_dir=tmp_path / "plan",
        experiment_prefix="disagg",
        provider="mock",
        engine="sglang",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:30000/v1",
        workload_file=Path("workloads/generated/prefill_decode_grid.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert len(written) == 2
    loaded = [load_config(path) for path in written]
    assert [config.strategy for config in loaded] == ["baseline", "prefill_decode_disaggregation"]
    assert all(config.engine == "sglang" for config in loaded)
    assert all(config.endpoint_type == "sglang" for config in loaded)
    assert all(
        config.metadata["prefill_decode_disaggregation_plan_id"] == "disagg_sglang"
        for config in loaded
    )


def test_run_disaggregation_plan_executes_configs_in_sorted_order(tmp_path: Path) -> None:
    calls: list[Path] = []
    first = tmp_path / "plan" / "b.yaml"
    second = tmp_path / "plan" / "a.yaml"
    first.parent.mkdir()
    base_config = "\n".join(
        [
            "provider: mock",
            "engine: vllm",
            "endpoint_type: vllm",
            "model_id: mock-frontier-model",
            "strategy: baseline",
            "base_url: http://127.0.0.1:8000/v1",
            "workload_file: workloads/generated/prefill_decode_grid.jsonl",
            "output_file: results/raw/out.jsonl",
        ]
    )
    first.write_text(f"experiment_id: b\n{base_config}\n", encoding="utf-8")
    second.write_text(f"experiment_id: a\n{base_config}\n", encoding="utf-8")

    async def fake_runner(path: Path) -> Path:
        calls.append(path)
        return Path("results") / f"{path.stem}.jsonl"

    outputs = run_prefill_decode_disaggregation_plan(first.parent, runner=fake_runner)

    assert calls == [second, first]
    assert outputs == [Path("results/a.jsonl"), Path("results/b.jsonl")]
