from __future__ import annotations

from spatial_lineage.api import build_inference_job
from spatial_lineage.commands.common import load_config_from_args


def main(argv: list[str] | None = None) -> int:
    config = load_config_from_args("Run spatial-lineage inference.", argv)
    output_paths = build_inference_job(config).run()
    print(f"outputs={output_paths}")
    return 0
