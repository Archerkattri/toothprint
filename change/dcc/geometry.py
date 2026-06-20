"""Small 2D geometry helpers used by the benchmark scaffolding."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

Point = list[float]


def mean_point(points: Sequence[Sequence[float]]) -> Point:
    if not points:
        raise ValueError("Cannot average an empty point sequence")
    x = sum(float(point[0]) for point in points) / len(points)
    y = sum(float(point[1]) for point in points) / len(points)
    return [x, y]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return sqrt(dx * dx + dy * dy)


def translate_point(point: Sequence[float], dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> Point:
    return [round(float(point[0]) * scale + dx, 10), round(float(point[1]) * scale + dy, 10)]


def translate_points(points: Iterable[Sequence[float]], dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> list[Point]:
    return [translate_point(point, dx=dx, dy=dy, scale=scale) for point in points]
