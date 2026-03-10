from __future__ import annotations

from typing import Any

from spatial_lineage.stable.core.config import ConfigNode


def build_scheduler(config: ConfigNode) -> dict[str, Any]:
    return {
        "name": config.scheduler.name,
        "warmup_epochs": config.scheduler.warmup_epochs,
    }
