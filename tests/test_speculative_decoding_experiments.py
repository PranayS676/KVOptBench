from pathlib import Path

from kvoptbench.config import load_config
from kvoptbench.experiments.speculative_decoding import (
    build_speculative_decoding_plan,
    run_speculative_decoding_plan,
    write_speculative_decoding_plan_configs,
)


def test_speculative_decoding_plan_builds_baseline_and_speculative_runs() -> None:
    cases = build_speculative_decoding_plan(
        experiment_prefix="spec_decode",
        provider="mock",
        engine="vllm",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:8000/v1",
        workload_file=Path("workloads/generated/decode_heavy.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert [case.strategy for case in cases] == ["baseline", "speculative_decoding"]
    assert {case.role for case in cases} == {"control", "speculative"}
    assert all(case.workload_profile == "decode_heavy" for case in cases)
    assert all(case.config.endpoint_type == "vllm" for case in cases)
    assert all(case.config.metadata["speculative_decoding_experiment"] is True for case in cases)
    assert cases[0].config.output_file == Path(
        "results/raw/spec_decode_vllm_baseline_decode_heavy.jsonl"
    )
    assert cases[1].config.output_file == Path(
        "results/raw/spec_decode_vllm_speculative_decoding_decode_heavy.jsonl"
    )


def test_write_speculative_decoding_plan_configs_creates_runnable_yaml(tmp_path: Path) -> None:
    written = write_speculative_decoding_plan_configs(
        plan_dir=tmp_path / "plan",
        experiment_prefix="spec_decode",
        provider="mock",
        engine="sglang",
        model_id="mock-frontier-model",
        base_url="http://127.0.0.1:30000/v1",
        workload_file=Path("workloads/generated/decode_heavy.jsonl"),
        output_dir=Path("results/raw"),
    )

    assert len(written) == 2
    loaded = [load_config(path) for path in written]
    assert [config.strategy for config in loaded] == ["baseline", "speculative_decoding"]
    assert all(config.engine == "sglang" for config in loaded)
    assert all(config.endpoint_type == "sglang" for config in loaded)
    assert all(config.metadata["speculative_decoding_plan_id"] == "spec_decode_sglang" for config in loaded)


def test_run_speculative_decoding_plan_executes_configs_in_sorted_order(tmp_path: Path) -> None:
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
            "workload_file: workloads/generated/decode_heavy.jsonl",
            "output_file: results/raw/out.jsonl",
        ]
    )
    first.write_text(f"experiment_id: b\n{base_config}\n", encoding="utf-8")
    second.write_text(f"experiment_id: a\n{base_config}\n", encoding="utf-8")

    async def fake_runner(path: Path) -> Path:
        calls.append(path)
        return Path("results") / f"{path.stem}.jsonl"

    outputs = run_speculative_decoding_plan(first.parent, runner=fake_runner)

    assert calls == [second, first]
    assert outputs == [Path("results/a.jsonl"), Path("results/b.jsonl")]
