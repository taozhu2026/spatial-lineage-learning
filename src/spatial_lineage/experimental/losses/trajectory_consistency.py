from __future__ import annotations

from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.loss import BaseLoss
from spatial_lineage.stable.contracts.outputs import LossOutput, ModelOutput


class TrajectoryConsistencyLoss(BaseLoss):
    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def compute(self, batch: Batch, output: ModelOutput) -> LossOutput:
        del batch, output
        return LossOutput(loss=0.0, metrics={"trajectory_consistency": 0.0})
