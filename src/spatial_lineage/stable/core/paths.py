from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spatial_lineage.stable.core.config import ConfigNode


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    checkpoints: Path
    logs: Path
    eval_dir: Path
    inference_dir: Path

    @classmethod
    def from_config(cls, config: ConfigNode) -> "ExperimentPaths":
        output_root = Path(config.runtime.output_root)
        root = output_root / config.experiment.name
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            logs=root / "logs",
            eval_dir=root / "eval",
            inference_dir=root / "inference",
        )

    def ensure(self) -> None:
        for path in (self.root, self.checkpoints, self.logs, self.eval_dir, self.inference_dir):
            path.mkdir(parents=True, exist_ok=True)
