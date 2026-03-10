from __future__ import annotations


def clone_bias(clone_id: str) -> float:
    return (sum(ord(char) for char in clone_id) % 13) / 13.0
