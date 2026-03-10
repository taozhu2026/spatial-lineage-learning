from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.outputs import PredictionBundle


class BaseEvaluator(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def evaluate(self, batch: Batch, bundle: PredictionBundle) -> dict[str, float]:
        raise NotImplementedError
