"""YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from kvoptbench.schemas import ExperimentConfig
from kvoptbench.telemetry.profiles import apply_telemetry_profile_defaults


class ConfigError(ValueError):
    """Raised when an experiment config cannot be loaded or validated."""


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Experiment config must be a YAML mapping")

    try:
        raw = apply_telemetry_profile_defaults(raw, config_path=config_path)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def validate_config(path: str | Path) -> ExperimentConfig:
    """Validate a config path and return the parsed config."""
    return load_config(path)

