from pathlib import Path
import json

from kvoptbench.runner.schedule import build_schedule, run_schedule
from kvoptbench.strategy.plan_run import run_strategy_plan, write_strategy_plan


def test_build_schedule_preserves_input_order_without_randomization() -> None:
    paths = [Path("configs/b.yaml"), Path("configs/a.yaml")]

    schedule = build_schedule(paths, repeat_count=1, seed=99, randomize=False)

    assert [(run.config_path, run.trial_index, run.repeat_index, run.order_index) for run in schedule] == [
        (paths[0], 0, 0, 0),
        (paths[1], 1, 0, 1),
    ]
    assert len({run.schedule_id for run in schedule}) == 1
    assert all(run.metadata["run_group_id"] == run.schedule_id for run in schedule)


def test_build_schedule_expands_repeats_by_repeat_then_config_order() -> None:
    paths = [Path("configs/a.yaml"), Path("configs/b.yaml")]

    schedule = build_schedule(paths, repeat_count=3, seed=0, randomize=False)

    assert [(run.config_path, run.trial_index, run.repeat_index) for run in schedule] == [
        (paths[0], 0, 0),
        (paths[1], 1, 0),
        (paths[0], 0, 1),
        (paths[1], 1, 1),
        (paths[0], 0, 2),
        (paths[1], 1, 2),
    ]
    assert [run.order_index for run in schedule] == list(range(6))


def test_build_schedule_randomized_order_is_reproducible_by_seed() -> None:
    paths = [Path(f"configs/{name}.yaml") for name in ("a", "b", "c", "d")]

    first = build_schedule(paths, repeat_count=3, seed=17, randomize=True)
    second = build_schedule(paths, repeat_count=3, seed=17, randomize=True)
    different_seed = build_schedule(paths, repeat_count=3, seed=18, randomize=True)

    first_order = [(run.config_path, run.trial_index, run.repeat_index) for run in first]
    second_order = [(run.config_path, run.trial_index, run.repeat_index) for run in second]
    different_seed_order = [
        (run.config_path, run.trial_index, run.repeat_index) for run in different_seed
    ]

    assert first_order == second_order
    assert first_order != different_seed_order
    assert {run.schedule_id for run in first} == {run.schedule_id for run in second}
    assert {run.schedule_id for run in first} != {run.schedule_id for run in different_seed}


def test_build_schedule_supports_randomized_blocks_by_repeat() -> None:
    paths = [Path("configs/a.yaml"), Path("configs/b.yaml"), Path("configs/c.yaml")]

    schedule = build_schedule(
        paths,
        repeat_count=3,
        seed=11,
        randomize=True,
        block_randomization=True,
    )

    assert [run.repeat_index for run in schedule[:3]] == [0, 0, 0]
    assert [run.repeat_index for run in schedule[3:6]] == [1, 1, 1]
    assert [run.repeat_index for run in schedule[6:]] == [2, 2, 2]
    assert all(run.metadata["block_randomization"] is True for run in schedule)
    assert [run.config_path for run in schedule[:3]] != paths


def test_run_schedule_delegates_in_scheduled_order() -> None:
    calls: list[Path] = []
    paths = [Path("configs/a.yaml"), Path("configs/b.yaml")]
    schedule = build_schedule(paths, repeat_count=2, seed=0, randomize=False)

    async def fake_runner(path: Path) -> Path:
        calls.append(path)
        return Path("results") / f"{len(calls)}-{path.stem}.jsonl"

    outputs = run_schedule(schedule, runner=fake_runner)

    assert calls == [paths[0], paths[1], paths[0], paths[1]]
    assert outputs == [
        Path("results/1-a.jsonl"),
        Path("results/2-b.jsonl"),
        Path("results/3-a.jsonl"),
        Path("results/4-b.jsonl"),
    ]


def test_strategy_plan_writes_matrix_manifest_and_configs(tmp_path: Path) -> None:
    workload_pack = tmp_path / "workload_pack.yaml"
    workload_pack.write_text(
        "\n".join(
            [
                "workloads:",
                "  shared_prefix: workloads/generated/shared.jsonl",
                "  random_prefix: workloads/generated/random.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    result = write_strategy_plan(
        plan_dir=tmp_path / "plan",
        matrix_id="cache_matrix",
        provider="local",
        engine="vllm",
        model_id="example/model",
        base_url="http://127.0.0.1:8000/v1",
        workload_pack=workload_pack,
        strategy_families=["cache"],
        strategies=["cache_on"],
        concurrencies=[1, 4],
        output_dir=tmp_path / "raw",
        repeat_count=3,
        randomization_seed=17,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert len(result.config_paths) == 8
    assert manifest["matrix_id"] == "cache_matrix"
    assert manifest["repeat_count"] == 3
    assert manifest["randomization_seed"] == 17
    assert {entry["workload_role"] for entry in manifest["configs"]} == {
        "shared_prefix",
        "random_prefix",
    }
    assert {entry["cache_pass"] for entry in manifest["configs"]} == {"cold", "warm"}
    assert all("config_sha256" in entry for entry in manifest["configs"])


def test_strategy_run_dry_run_writes_reproducible_run_manifest(tmp_path: Path) -> None:
    workload_pack = tmp_path / "workload_pack.yaml"
    workload_pack.write_text(
        "\n".join(
            [
                "workloads:",
                "  primary: workloads/generated/decode.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    planned = write_strategy_plan(
        plan_dir=tmp_path / "plan",
        matrix_id="decode_matrix",
        provider="local",
        engine="vllm",
        model_id="example/model",
        base_url="http://127.0.0.1:8000/v1",
        workload_pack=workload_pack,
        strategy_families=["decode-heavy"],
        strategies=["baseline"],
        concurrencies=[1],
        output_dir=tmp_path / "raw",
        repeat_count=2,
        randomization_seed=5,
    )

    first = run_strategy_plan(
        matrix_manifest=planned.manifest_path,
        output_run_manifest=tmp_path / "run_manifest.json",
        randomize=True,
        dry_run=True,
    )
    second = run_strategy_plan(
        matrix_manifest=planned.manifest_path,
        output_run_manifest=tmp_path / "run_manifest_2.json",
        randomize=True,
        dry_run=True,
    )

    first_manifest = json.loads(first.run_manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.run_manifest_path.read_text(encoding="utf-8"))

    assert first_manifest["dry_run"] is True
    assert first_manifest["outputs"] == []
    assert first_manifest["planned_runs"] == second_manifest["planned_runs"]
    assert first_manifest["planned_runs"][0]["repeat_count"] == 2
    assert first_manifest["planned_runs"][0]["randomize"] is True
