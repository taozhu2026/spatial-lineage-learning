from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.outputs import LossOutput, ModelOutput


class BaseLoss(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def compute(self, batch: Batch, output: ModelOutput) -> LossOutput:
        raise NotImplementedError
