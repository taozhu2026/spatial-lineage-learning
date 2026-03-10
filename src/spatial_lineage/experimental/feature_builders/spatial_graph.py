from __future__ import annotations


def spatial_signal(coords: tuple[float, float]) -> tuple[float, float]:
    x_coord, y_coord = coords
    return (x_coord / 10.0, y_coord / 10.0)
