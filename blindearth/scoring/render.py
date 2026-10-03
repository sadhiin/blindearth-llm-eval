"""Points <-> equirectangular grids and images.

Orientation follows INTERFACES.md: row 0 = northernmost row, col 0 = westernmost column
(180°W), True/1 = land.

Grid geometry for step ``s``:
- ``cell_center``: row r has lat ``90 - s/2 - r*s``, col c has lon ``-180 + s/2 + c*s``.
- ``cell_corner``: row r has lat ``90 - r*s``, col c has lon ``-180 + c*s``.
Both have ``ceil(180/s)`` rows and ``ceil(360/s)`` columns.

``points_to_grid`` returns float arrays with NaN where there is no value:
- ``"p_land"``: P(Land); NaN for invalid answers and points not in the frame.
- ``"binary"``: 1.0 land / 0.0 water by ``p_land > threshold``; NaN as above.
- ``"error"``: one of the ``ERR_*`` codes below; invalid answers get ``ERR_INVALID``
  (counted as wrong), points not in the frame or with unknown truth are NaN.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from blindearth.types import Placement

ERR_CORRECT = 0
ERR_FALSE_LAND = 1  # model said Land, truth is water
ERR_FALSE_WATER = 2  # model said Water, truth is land
ERR_INVALID = 3  # no valid answer; wrong in the headline accuracy

# Report error map: green where correct, red shades where wrong (split by error type).
ERROR_MAP_COLORS: dict[int, tuple[int, int, int]] = {
    ERR_CORRECT: (44, 160, 44),
    ERR_FALSE_LAND: (214, 39, 40),
    ERR_FALSE_WATER: (140, 16, 60),
    ERR_INVALID: (150, 150, 150),
}
MISSING_COLOR: tuple[int, int, int] = (32, 32, 32)

# Image difference overlay (spec "Image comparison"): red = false Land, blue = false Water.
OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "land": (235, 235, 235),
    "water": (0, 0, 0),
    "false_land": (220, 30, 30),
    "false_water": (30, 90, 230),
}


# --------------------------------------------------------------------------- geometry


def _offset(placement: Placement) -> float:
    return 0.5 if placement == "cell_center" else 0.0


def grid_shape(step_deg: float, placement: Placement = "cell_center") -> tuple[int, int]:
    """(rows, cols) of the grid for this step (placement does not change the count)."""
    del placement
    step = float(step_deg)
    return int(math.ceil(180.0 / step - 1e-9)), int(math.ceil(360.0 / step - 1e-9))


def grid_index(
    lat: np.ndarray, lon: np.ndarray, step_deg: float, placement: Placement
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column index of each (lat, lon) point. Rows are clipped, columns wrap."""
    nr, nc = grid_shape(step_deg, placement)
    off = _offset(placement)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r = np.rint((90.0 - lat) / step_deg - off).astype(np.int64)
    c = np.rint((lon + 180.0) / step_deg - off).astype(np.int64)
    return np.clip(r, 0, nr - 1), np.mod(c, nc)


def infer_step(lat: np.ndarray, lon: np.ndarray, default: float = 2.0) -> float:
    """Grid step from the smallest spacing between distinct lats or lons."""
    diffs = []
    for a in (np.asarray(lat, float), np.asarray(lon, float)):
        u = np.unique(np.round(a[np.isfinite(a)], 6))
        if u.size > 1:
            d = np.diff(u)
            d = d[d > 1e-6]
            if d.size:
                diffs.append(d.min())
    if not diffs:
        return float(default)
    return float(np.round(min(diffs), 6))


def infer_placement(lat: np.ndarray, step_deg: float) -> Placement:
    """``cell_center`` if lats sit half a step off the 90° line, else ``cell_corner``."""
    lat = np.asarray(lat, float)
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return "cell_center"
    frac = np.mod((90.0 - lat) / step_deg, 1.0)
    dist_half = np.abs(frac - 0.5)
    return "cell_center" if float(np.median(dist_half)) < 0.25 else "cell_corner"


def values_to_grid(
    lat: np.ndarray, lon: np.ndarray, values: np.ndarray, step_deg: float, placement: Placement
) -> np.ndarray:
    """Scatter values onto a float grid (NaN where no point)."""
    nr, nc = grid_shape(step_deg, placement)
    out = np.full((nr, nc), np.nan, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return out
    r, c = grid_index(lat, lon, step_deg, placement)
    out[r, c] = values
    return out


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def error_codes(p_land: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Per-point ``ERR_*`` code (float, NaN where truth unknown)."""
    p = np.asarray(p_land, float)
    t = np.asarray(truth, float)
    valid = np.isfinite(p)
    pred = valid & (p > threshold)
    t_land = t > 0.5
    codes = np.full(p.shape, np.nan)
    codes[valid & (pred == t_land)] = ERR_CORRECT
    codes[valid & pred & ~t_land] = ERR_FALSE_LAND
    codes[valid & ~pred & t_land] = ERR_FALSE_WATER
    codes[~valid] = ERR_INVALID
    codes[~np.isfinite(t)] = np.nan
    return codes


def points_to_grid(
    df: pd.DataFrame,
    step_deg: float,
    placement: Placement,
    value: Literal["p_land", "binary", "error"],
    threshold: float = 0.5,
) -> np.ndarray:
    """Points table -> (rows, cols) float grid; see module docstring for values."""
    if df is None or len(df) == 0:
        return np.full(grid_shape(step_deg, placement), np.nan)
    lat = _col(df, "lat")
    lon = _col(df, "lon")
    p = _col(df, "p_land")
    if value == "p_land":
        v = p
    elif value == "binary":
        v = np.where(np.isfinite(p), (p > threshold).astype(float), np.nan)
    elif value == "error":
        v = error_codes(p, _col(df, "truth"), threshold)
    else:
        raise ValueError(f"unknown value {value!r}")
    return values_to_grid(lat, lon, v, step_deg, placement)


# --------------------------------------------------------------------------- upsampling


def _src_coords(n_out: int, n_src: int, extent_deg: float, step: float, off: float) -> np.ndarray:
    """Source grid coordinate (in cells) of each output pixel center."""
    return (np.arange(n_out) + 0.5) * (extent_deg / n_out) / step - off


def render_to_shape(
    grid: np.ndarray,
    shape: tuple[int, int],
    step_deg: float | None = None,
    placement: Placement = "cell_center",
    *,
    binary: bool,
) -> np.ndarray:
    """Render a grid onto an equirectangular image of ``shape``.

    Each grid value is placed at its point's coordinates (so ``cell_corner`` points are shifted
    half a cell relative to ``cell_center``). Binary: nearest neighbour, dtype preserved (NaN kept),
    so blockiness counts against the model. Probability: bilinear, longitude periodic, latitude
    clamped at the poles; NaN is treated as 0 (no land); returns float32.
    """
    g = np.asarray(grid)
    nr, nc = g.shape
    H, W = int(shape[0]), int(shape[1])
    step = float(step_deg) if step_deg is not None else 360.0 / nc
    off = _offset(placement)
    sy = _src_coords(H, nr, 180.0, step, off)
    sx = _src_coords(W, nc, 360.0, step, off)
    if binary:
        ri = np.clip(np.floor(sy + 0.5).astype(np.int64), 0, nr - 1)
        ci = np.mod(np.floor(sx + 0.5).astype(np.int64), nc)
        return g[np.ix_(ri, ci)]
    gf = np.nan_to_num(g.astype(np.float32), nan=0.0)
    y0 = np.floor(sy).astype(np.int64)
    wy = (sy - y0).astype(np.float32)
    y1 = np.clip(y0 + 1, 0, nr - 1)
    y0 = np.clip(y0, 0, nr - 1)
    rows = gf[y0] * (1.0 - wy)[:, None] + gf[y1] * wy[:, None]  # (H, nc)
    x0 = np.floor(sx).astype(np.int64)
    wx = (sx - x0).astype(np.float32)
    x1 = np.mod(x0 + 1, nc)
    x0 = np.mod(x0, nc)
    out = rows[:, x0] * (1.0 - wx)[None, :] + rows[:, x1] * wx[None, :]
    return out.astype(np.float32, copy=False)


def upsample_to(grid: np.ndarray, shape: tuple[int, int], *, binary: bool) -> np.ndarray:
    """Upsample a grid to ``shape`` assuming cell-center placement and a step of 360/cols.

    Nearest neighbour for binary (dtype preserved), bilinear for probability (float32, NaN -> 0).
    Use ``render_to_shape`` for corner placement or steps that do not divide 180°.
    """
    return render_to_shape(grid, shape, None, "cell_center", binary=binary)


# --------------------------------------------------------------------------- colour images


def error_overlay(pred_binary: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB uint8 overlay: red = false Land, blue = false Water, white/black where correct."""
    m = np.asarray(getattr(mask, "data", mask), dtype=bool)
    p = np.asarray(pred_binary)
    if p.dtype != bool:
        p = np.nan_to_num(p.astype(float), nan=0.0) > 0.5
    if p.shape != m.shape:
        raise ValueError(f"shape mismatch: prediction {p.shape} vs mask {m.shape}")
    out = np.empty(m.shape + (3,), dtype=np.uint8)
    out[...] = OVERLAY_COLORS["water"]
    out[p & m] = OVERLAY_COLORS["land"]
    out[p & ~m] = OVERLAY_COLORS["false_land"]
    out[~p & m] = OVERLAY_COLORS["false_water"]
    return out


def error_map_rgb(err_grid: np.ndarray) -> np.ndarray:
    """RGB uint8 image of an ``"error"`` grid using ``ERROR_MAP_COLORS`` (missing = dark grey)."""
    e = np.asarray(err_grid, dtype=float)
    out = np.empty(e.shape + (3,), dtype=np.uint8)
    out[...] = MISSING_COLOR
    for code, color in ERROR_MAP_COLORS.items():
        out[e == code] = color
    return out
