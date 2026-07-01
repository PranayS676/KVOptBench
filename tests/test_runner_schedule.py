from pathlib import Path

from kvoptbench.runner.schedule import build_schedule, run_schedule


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
