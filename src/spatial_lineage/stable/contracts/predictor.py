from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.model import BaseModel
from spatial_lineage.stable.contracts.outputs import PredictionBundle


class BasePredictor(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def predict(self, model: BaseModel, batches: Iterable[Batch]) -> PredictionBundle:
        raise NotImplementedError
