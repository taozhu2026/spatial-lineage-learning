from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Batch:
    cell_ids: list[str]
    expression: list[list[float]]
    spatial_coords: list[tuple[float, float]]
    clone_ids: list[str]
    targets: list[int] | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.cell_ids)
