from __future__ import annotations

import math
from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.evaluator import BaseEvaluator
from spatial_lineage.stable.contracts.outputs import PredictionBundle
from spatial_lineage.stable.core.registry import EVALUATORS


@EVALUATORS.register("spatial_coherence")
class SpatialCoherenceEvaluator(BaseEvaluator):
    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def evaluate(self, batch: Batch, bundle: PredictionBundle) -> dict[str, float]:
        agreements = 0
        comparisons = 0
        for left in range(len(batch.cell_ids)):
            for right in range(left + 1, len(batch.cell_ids)):
                distance = math.dist(batch.spatial_coords[left], batch.spatial_coords[right])
                if distance <= 1.5:
                    comparisons += 1
                    agreements += int(bundle.predicted_labels[left] == bundle.predicted_labels[right])
        score = agreements / comparisons if comparisons else 0.0
        return {"spatial_coherence": score}
