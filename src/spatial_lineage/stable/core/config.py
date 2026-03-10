from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def resolve_inheritance(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_yaml(config_path)
    bases = config.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base_path in bases:
        parent_config = resolve_inheritance(config_path.parent / base_path)
        merged = deep_merge(merged, parent_config)
    return deep_merge(merged, config)


class ConfigNode:
    def __init__(self, data: dict[str, Any], source_path: Path | None = None) -> None:
        self._data = deepcopy(data)
        self.source_path = source_path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._wrap(self._data.get(name))

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._wrap(self._data.get(key, default))

    def items(self) -> Iterable[tuple[str, Any]]:
        for key, value in self._data.items():
            yield key, self._wrap(value)

    def set_nested(self, dotted_key: str, value: Any) -> None:
        cursor = self._data
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self._data, handle, sort_keys=False)
        return target

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict):
            return ConfigNode(value, source_path=self.source_path)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> ConfigNode:
    resolved_path = Path(path).resolve()
    config = ConfigNode(resolve_inheritance(resolved_path), source_path=resolved_path)
    for override in overrides or []:
        key, raw_value = override.split("=", maxsplit=1)
        config.set_nested(key, yaml.safe_load(raw_value))
    return config
