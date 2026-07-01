"""Named telemetry profile loading and config expansion."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_PROFILE_RESOURCE = "default_profiles.yaml"


class TelemetryProfile(BaseModel):
    """Reusable telemetry defaults for a common benchmark environment."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    telemetry: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


def load_telemetry_profiles(profile_path: str | Path | None = None) -> dict[str, TelemetryProfile]:
    """Load built-in telemetry profiles and optional user profile overrides."""
    profiles = _load_default_profiles()
    if profile_path is not None:
        profiles.update(_load_profiles_from_path(Path(profile_path)))
    return dict(sorted(profiles.items()))


def get_telemetry_profile(
    name: str,
    *,
    profile_path: str | Path | None = None,
) -> TelemetryProfile:
    """Return one named telemetry profile or raise a helpful error."""
    profiles = load_telemetry_profiles(profile_path)
    if name not in profiles:
        available = ", ".join(profiles) or "<none>"
        raise ValueError(f"Unknown telemetry profile '{name}'. Available profiles: {available}.")
    return profiles[name]


def apply_telemetry_profile_defaults(
    raw_config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Expand ``telemetry.profile`` into a normal telemetry config mapping.

    Profile defaults are only applied before Pydantic validation. Explicit fields
    in the experiment YAML override profile values, including complete list
    replacement for source lists such as ``prometheus`` and ``lmcache``.
    """
    telemetry = raw_config.get("telemetry")
    if not isinstance(telemetry, dict):
        return raw_config
    profile_name = telemetry.get("profile")
    if profile_name is None:
        return raw_config
    profile_name = str(profile_name).strip()
    if not profile_name:
        return raw_config

    profile_path = _resolve_profile_path(telemetry.get("profile_path"), config_path)
    profile = get_telemetry_profile(profile_name, profile_path=profile_path)
    merged = _deep_merge(profile.telemetry, telemetry)

    expanded = dict(raw_config)
    expanded["telemetry"] = merged
    return expanded


def profile_to_dict(profile: TelemetryProfile) -> dict[str, Any]:
    """Serialize a profile for CLI output without Pydantic internals."""
    return profile.model_dump(mode="json")


def _load_default_profiles() -> dict[str, TelemetryProfile]:
    resource = resources.files("kvoptbench.telemetry").joinpath(DEFAULT_PROFILE_RESOURCE)
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return _profiles_from_payload(payload)


def _load_profiles_from_path(path: Path) -> dict[str, TelemetryProfile]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _profiles_from_payload(payload)


def _profiles_from_payload(payload: Any) -> dict[str, TelemetryProfile]:
    if not isinstance(payload, dict):
        raise ValueError("Telemetry profile file must contain a YAML mapping.")
    raw_profiles = payload.get("profiles", payload)
    if not isinstance(raw_profiles, dict):
        raise ValueError("Telemetry profile file must contain a 'profiles' mapping.")

    profiles: dict[str, TelemetryProfile] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Telemetry profile '{name}' must be a mapping.")
        profile_payload = {"name": str(name), **raw_profile}
        profile = TelemetryProfile.model_validate(profile_payload)
        profiles[profile.name] = profile
    return profiles


def _resolve_profile_path(raw_path: Any, config_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute() or config_path is None:
        return path
    return Path(config_path).parent / path


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = _merge_named_source_lists(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_named_source_lists(defaults: list[Any], overrides: list[Any]) -> list[Any]:
    if not overrides:
        return []
    if not _all_named_mappings(defaults) or not _all_named_mappings(overrides):
        return deepcopy(overrides)

    merged_by_name = {str(item["name"]): deepcopy(item) for item in defaults}
    order = [str(item["name"]) for item in defaults]
    for override in overrides:
        name = str(override["name"])
        if name in merged_by_name:
            merged_by_name[name] = _deep_merge(merged_by_name[name], override)
        else:
            order.append(name)
            merged_by_name[name] = deepcopy(override)
    return [merged_by_name[name] for name in order if name in merged_by_name]


def _all_named_mappings(items: list[Any]) -> bool:
    return all(isinstance(item, dict) and item.get("name") for item in items)
