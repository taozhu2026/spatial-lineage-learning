from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.log_dir / "metrics.jsonl"
        self._events_path = self.log_dir / "events.log"

    def info(self, message: str) -> None:
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def log_metrics(self, split: str, step: int, metrics: dict[str, float]) -> None:
        payload: dict[str, Any] = {"split": split, "step": step, "metrics": metrics}
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
