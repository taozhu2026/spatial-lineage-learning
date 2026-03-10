from __future__ import annotations

from spatial_lineage.stable.utils.tensor import mean


def mean_confidence(scores: list[list[float]]) -> float:
    return mean([max(row) for row in scores]) if scores else 0.0
