from __future__ import annotations

from typing import Any, Iterable


def batch_records(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for index in range(0, len(records), batch_size):
        yield records[index : index + batch_size]
