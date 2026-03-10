from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    offset = max(logits)
    exps = [math.exp(value - offset) for value in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index
