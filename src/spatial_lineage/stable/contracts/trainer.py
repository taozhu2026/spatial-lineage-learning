from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def train(self) -> dict[str, float]:
        raise NotImplementedError
