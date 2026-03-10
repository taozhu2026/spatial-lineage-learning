from __future__ import annotations

import argparse
from pathlib import Path

from spatial_lineage.stable.core.config import ConfigNode, load_config


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=Path, help="Path to the YAML experiment config.")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. -o training.batch_size=8",
    )
    return parser


def load_config_from_args(description: str, argv: list[str] | None = None) -> ConfigNode:
    parser = build_parser(description)
    args = parser.parse_args(argv)
    return load_config(args.config, overrides=args.override)
