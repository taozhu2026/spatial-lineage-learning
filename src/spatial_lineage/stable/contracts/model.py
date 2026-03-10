from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spatial_lineage.stable.contracts.batch import Batch
from spatial_lineage.stable.contracts.outputs import ModelOutput


class BaseModel(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state
