from __future__ import annotations

from spatial_lineage.stable.core.config import ConfigNode


def resolve_device(config: ConfigNode) -> str:
    return config.runtime.device or "cpu"
