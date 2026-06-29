from pathlib import Path

from kvoptbench.config import load_config
from kvoptbench.experiments.prefill_decode import (
    build_prefill_decode_plan,
    run_prefill_decode_plan,
    write_prefill_decode_plan_configs,
)


def test_prefill_decode_plan_builds_config_driven_grid_run() -> None:
    cases = build_prefill_decode_plan(
        experiment_prefix="prefill_decode",
        provider="mock",
        engine="vllm",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:8000/v1",
        workload_file=Path("workloads/generated/prefill_decode_grid.jsonl"),
        output_dir=Path("results/raw"),
        strategies=("baseline",),
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.strategy == "baseline"
    assert case.config.endpoint_type == "vllm"
    assert case.config.metadata["prefill_decode_experiment"] is True
    assert case.config.metadata["workload_profile"] == "prefill_decode_grid"
    assert case.config.output_file == Path("results/raw/prefill_decode_vllm_baseline_grid.jsonl")


def test_write_prefill_decode_plan_configs_creates_runnable_yaml(tmp_path: Path) -> None:
    written = write_prefill_decode_plan_configs(
        plan_dir=tmp_path / "plan",
        experiment_prefix="prefill_decode",
        provider="mock",
        engine="sglang",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:30000/v1",
        workload_file=Path("workloads/generated/prefill_decode_grid.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert len(written) == 1
    config = load_config(written[0])
    assert config.engine == "sglang"
    assert config.endpoint_type == "sglang"
    assert config.strategy == "baseline"
    assert config.metadata["prefill_decode_plan_id"] == "prefill_decode_sglang"


def test_run_prefill_decode_plan_executes_configs_in_sorted_order(tmp_path: Path) -> None:
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

    outputs = run_prefill_decode_plan(first.parent, runner=fake_runner)

    assert calls == [second, first]
    assert outputs == [Path("results/a.jsonl"), Path("results/b.jsonl")]
