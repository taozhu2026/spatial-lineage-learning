from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize prediction records from a JSONL file.")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    count = 0
    labels: dict[str, int] = {}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = str(record["predicted_label"])
            labels[label] = labels.get(label, 0) + 1
            count += 1
    print(json.dumps({"count": count, "labels": labels}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
