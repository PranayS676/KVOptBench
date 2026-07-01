"""Deterministic local run scheduling helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from kvoptbench.runner.experiment import run_experiment


@dataclass(frozen=True, slots=True)
class ScheduledRun:
    """One config execution within a reproducible local run schedule."""

    config_path: Path
    trial_index: int
    repeat_index: int
    order_index: int
    schedule_id: str
    seed: int
    randomize: bool
    block_randomization: bool
    repeat_count: int

    @property
    def metadata(self) -> dict[str, object]:
        """Return reproducibility metadata suitable for attaching to run records."""
        return {
            "schedule_id": self.schedule_id,
            "run_group_id": self.schedule_id,
            "config_path": self.config_path.as_posix(),
            "trial_index": self.trial_index,
            "repeat_index": self.repeat_index,
            "order_index": self.order_index,
            "seed": self.seed,
            "randomize": self.randomize,
            "block_randomization": self.block_randomization,
            "repeat_count": self.repeat_count,
        }


def build_schedule(
    config_paths: Iterable[str | Path],
    repeat_count: int = 1,
    seed: int = 0,
    randomize: bool = False,
    block_randomization: bool = False,
) -> list[ScheduledRun]:
    """Build a deterministic schedule from experiment config paths."""
    paths = [Path(path) for path in config_paths]
    if not paths:
        raise ValueError("At least one config path is required")
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")

    schedule_id = _build_schedule_id(
        config_paths=paths,
        repeat_count=repeat_count,
        seed=seed,
        randomize=randomize,
        block_randomization=block_randomization,
    )
    expanded = []
    rng = random.Random(seed)
    for repeat_index in range(repeat_count):
        block = [(config_path, trial_index, repeat_index) for trial_index, config_path in enumerate(paths)]
        if randomize and block_randomization:
            rng.shuffle(block)
        expanded.extend(block)
    if randomize:
        if not block_randomization:
            rng.shuffle(expanded)

    return [
        ScheduledRun(
            config_path=config_path,
            trial_index=trial_index,
            repeat_index=repeat_index,
            order_index=order_index,
            schedule_id=schedule_id,
            seed=seed,
            randomize=randomize,
            block_randomization=block_randomization,
            repeat_count=repeat_count,
        )
        for order_index, (config_path, trial_index, repeat_index) in enumerate(expanded)
    ]


def run_schedule(
    schedule: Sequence[ScheduledRun],
    *,
    runner: Callable[[Path], Awaitable[Path]] = run_experiment,
) -> list[Path]:
    """Run scheduled configs sequentially and return output paths in schedule order."""

    async def _run_all() -> list[Path]:
        outputs: list[Path] = []
        for scheduled_run in schedule:
            outputs.append(await runner(scheduled_run.config_path))
        return outputs

    return asyncio.run(_run_all())


def _build_schedule_id(
    *,
    config_paths: Sequence[Path],
    repeat_count: int,
    seed: int,
    randomize: bool,
    block_randomization: bool,
) -> str:
    payload = {
        "config_paths": [path.as_posix() for path in config_paths],
        "repeat_count": repeat_count,
        "seed": seed,
        "randomize": randomize,
        "block_randomization": block_randomization,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"schedule-{digest}"
