from __future__ import annotations

from spatial_lineage.api import build_eval_job
from spatial_lineage.commands.common import load_config_from_args


def main(argv: list[str] | None = None) -> int:
    config = load_config_from_args("Run spatial-lineage evaluation.", argv)
    report = build_eval_job(config).run()
    print(f"metrics={report.metrics}")
    return 0
