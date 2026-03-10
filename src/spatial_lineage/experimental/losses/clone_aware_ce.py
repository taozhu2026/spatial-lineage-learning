from __future__ import annotations

import math
from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.loss import BaseLoss
from spatial_lineage.stable.contracts.outputs import LossOutput, ModelOutput
from spatial_lineage.stable.core.registry import LOSSES
from spatial_lineage.stable.utils.tensor import argmax, softmax


@LOSSES.register("clone_aware_ce")
class CloneAwareCrossEntropy(BaseLoss):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.label_smoothing = float(config.loss.label_smoothing)

    def compute(self, batch: Batch, output: ModelOutput) -> LossOutput:
        if batch.targets is None:
            return LossOutput(loss=0.0, metrics={"accuracy": 0.0})

        losses: list[float] = []
        predictions: list[int] = []
        for logits, target in zip(output.logits, batch.targets):
            probabilities = softmax(logits)
            adjusted = (1.0 - self.label_smoothing) * probabilities[target] + self.label_smoothing / len(probabilities)
            losses.append(-math.log(max(adjusted, 1e-8)))
            predictions.append(argmax(probabilities))

        accuracy = sum(int(pred == target) for pred, target in zip(predictions, batch.targets)) / len(batch.targets)
        return LossOutput(loss=sum(losses) / len(losses), metrics={"accuracy": accuracy})
