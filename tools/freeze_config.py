from __future__ import annotations

import argparse
from pathlib import Path

from spatial_lineage.stable.core.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a YAML config and write the frozen result.")
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    config.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
