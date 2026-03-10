from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


def apply_transforms(records: list[dict[str, Any]], data_config: dict[str, Any]) -> list[dict[str, Any]]:
    processed = deepcopy(records)
    if data_config.get("normalize_expression", False):
        _normalize_expression(processed)
    if data_config.get("log1p", False):
        _log1p_expression(processed)
    return processed


def _normalize_expression(records: list[dict[str, Any]]) -> None:
    for record in records:
        values = list(record["expression"])
        total = sum(values) or 1.0
        record["expression"] = [value / total for value in values]


def _log1p_expression(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["expression"] = [math.log1p(value) for value in record["expression"]]
