from __future__ import annotations


def pooled_expression(expression: list[float]) -> float:
    return sum(expression) / len(expression) if expression else 0.0
