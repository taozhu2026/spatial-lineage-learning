from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a JSON checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    with args.checkpoint.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
