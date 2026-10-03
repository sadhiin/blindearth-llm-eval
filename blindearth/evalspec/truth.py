"""Per-point ground truth from a mask.

Rules:

- ``cell_center``: the mask pixel containing the queried point (the coordinate the model was
  asked about). With corner placement that is the corner itself, not the middle of the cell.
  A point exactly on a pixel edge falls in the pixel to its south / east; lon 180 wraps to the
  -180 column; lat +-90 clamps to the first/last row.
- ``majority``: land when the mean of the mask pixels whose *centers* lie inside the cell bounds
  (``grid.cell_bounds``, edges inclusive) is > 0.5; an exact tie is water. Cells crossing the
  +-180 seam wrap around column-wise; rows are clipped at the poles. A cell smaller than one mask
  pixel falls back to the pixel containing the cell's middle.
"""

from __future__ import annotations

import math

import numpy as np

from blindearth.evalspec.grid import cell_bounds
from blindearth.evalspec.masks import Mask
from blindearth.types import GridSpec, Point, TruthRule

_EPS = 1e-9


def _pixel_of(lat: np.ndarray, lon: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    dy, dx = 180.0 / h, 360.0 / w
    rows = np.floor((90.0 - lat) / dy + _EPS).astype(np.int64)
    rows = np.clip(rows, 0, h - 1)
    cols = np.floor((lon + 180.0) / dx + _EPS).astype(np.int64) % w
    return rows, cols


def sample_points(mask: Mask, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Mask value at each (lat, lon), as int8 0/1."""
    h, w = mask.data.shape
    rows, cols = _pixel_of(np.asarray(lat, float), np.asarray(lon, float), h, w)
    return mask.data[rows, cols].astype(np.int8)


def majority_in_bounds(mask: Mask, west: float, south: float, east: float, north: float) -> int:
    """1 when > 50% of the mask pixels centred inside the box are land, else 0."""
    data = mask.data
    h, w = data.shape
    dy, dx = 180.0 / h, 360.0 / w
    south, north = max(-90.0, south), min(90.0, north)

    r0 = max(0, math.ceil((90.0 - north) / dy - 0.5 - _EPS))
    r1 = min(h - 1, math.floor((90.0 - south) / dy - 0.5 + _EPS))
    c0 = math.ceil((west + 180.0) / dx - 0.5 - _EPS)
    c1 = math.floor((east + 180.0) / dx - 0.5 + _EPS)

    if r1 < r0 or c1 < c0:
        mid_lat = (south + north) / 2.0
        mid_lon = (west + east) / 2.0
        rr, cc = _pixel_of(np.array([mid_lat]), np.array([mid_lon]), h, w)
        return int(data[rr[0], cc[0]])

    rows = slice(r0, r1 + 1)
    if c1 - c0 + 1 >= w:
        block = data[rows]
    elif 0 <= c0 and c1 < w:
        block = data[rows, c0 : c1 + 1]
    else:  # crosses the +-180 seam
        cols = np.arange(c0, c1 + 1) % w
        block = data[rows][:, cols]
    n = block.size
    land = int(np.count_nonzero(block))
    return 1 if land * 2 > n else 0


def cell_truth(mask: Mask, points: list[Point], grid: GridSpec, rule: TruthRule) -> np.ndarray:
    """Truth per point (1 land, 0 water), int8, aligned with ``points``."""
    if not points:
        return np.zeros(0, dtype=np.int8)
    if rule == "cell_center":
        lat = np.array([p.lat for p in points], dtype=float)
        lon = np.array([p.lon for p in points], dtype=float)
        return sample_points(mask, lat, lon)
    if rule == "majority":
        out = np.empty(len(points), dtype=np.int8)
        for i, p in enumerate(points):
            west, south, east, north = cell_bounds(p, grid.step_deg, grid.placement)
            out[i] = majority_in_bounds(mask, west, south, east, north)
        return out
    raise ValueError(f"unknown truth rule {rule!r} (use cell_center or majority)")
