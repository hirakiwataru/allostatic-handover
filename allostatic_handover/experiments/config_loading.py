"""Config loading helpers for experiment CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_mapping(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML config files.") from exc
        value = yaml.safe_load(text) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return dict(value)


def parse_key_value_overrides(items: list[str] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key is empty: {item}")
        values[key] = parse_scalar(raw_value.strip())
    return values


def parse_scalar(value: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(value)
    except Exception:
        pass
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def nested_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return dict(value)


def merge_config(defaults: Mapping[str, Any] | None, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults or {})
    merged.update(dict(overrides or {}))
    return merged
