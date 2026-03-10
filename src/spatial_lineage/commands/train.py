from __future__ import annotations

from spatial_lineage.api import build_train_job
from spatial_lineage.commands.common import load_config_from_args


def main(argv: list[str] | None = None) -> int:
    config = load_config_from_args("Run a spatial-lineage training job.", argv)
    result = build_train_job(config).run()
    print(f"best_metric={result.best_metric:.4f}")
    print(f"checkpoint={result.checkpoint_path}")
    return 0
