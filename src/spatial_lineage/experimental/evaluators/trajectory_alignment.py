from __future__ import annotations

from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.evaluator import BaseEvaluator
from spatial_lineage.stable.contracts.outputs import PredictionBundle


class TrajectoryAlignmentEvaluator(BaseEvaluator):
    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def evaluate(self, batch: Batch, bundle: PredictionBundle) -> dict[str, float]:
        del batch, bundle
        return {"trajectory_alignment": 0.0}
