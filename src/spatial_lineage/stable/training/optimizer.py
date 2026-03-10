from __future__ import annotations

from typing import Any

from spatial_lineage.stable.core.config import ConfigNode


def build_optimizer(config: ConfigNode) -> dict[str, Any]:
    return {
        "name": config.optimizer.name,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }
