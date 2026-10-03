"""Point grids over the globe.

Conventions (see INTERFACES.md):

- ``Point.idx`` is the row-major index in the *full* grid, north to south, west to east. It does
  not change when a subset is taken, so a 10% smoke run and a full run share indices.
- Cell-center placement at step ``s``: lats ``90 - s/2, 90 - 3s/2, ..., -90 + s/2``,
  lons ``-180 + s/2, ..., 180 - s/2`` (2 deg: lats 89..-89, lons -179..179).
- Cell-corner placement: each point is the north-west corner of its cell. Lats ``90 .. -90 + s``,
  lons ``-180 .. 180 - s`` (2 deg: lats 90..-88, lons -180..178).
- ``weight = cos(lat)``, clamped at 0. With corner placement the 90 deg N row therefore has zero
  weight; that is what the spec's "weighted by cos(latitude)" gives and it is kept on purpose so
  weights depend only on the coordinate the model was asked about.
- The step must divide 180 exactly (1, 2, 3, 4, 5, 6, 9, 10, 0.5, ...). Steps that do not tile
  the globe (e.g. 7) are rejected, because the last row/column would cover a partial cell.
"""

from __future__ import annotations

import math

import numpy as np

from blindearth.types import GridSpec, Placement, Point

_EPS = 1e-9
_COORD_DECIMALS = 9


def grid_dims(step_deg: float) -> tuple[int, int]:
    """(n_rows, n_cols) for a step; raises ValueError when the step does not tile the globe."""
    step = float(step_deg)
    if not math.isfinite(step) or step <= 0 or step > 180:
        raise ValueError(f"grid step_deg must be in (0, 180], got {step_deg!r}")
    n_rows_f = 180.0 / step
    n_rows = int(round(n_rows_f))
    if n_rows < 1 or abs(n_rows_f - n_rows) > 1e-6:
        raise ValueError(
            f"grid step_deg={step_deg!r} does not divide 180 evenly; pick a step such as "
            "1, 2, 3, 4, 5, 6, 9 or 10"
        )
    return n_rows, 2 * n_rows


def _clean(x: float) -> float:
    v = round(float(x), _COORD_DECIMALS)
    return 0.0 if v == 0 else v  # no -0.0


def _axes(step: float, placement: Placement) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = grid_dims(step)
    if placement == "cell_center":
        lats = 90.0 - step / 2.0 - step * np.arange(n_rows)
        lons = -180.0 + step / 2.0 + step * np.arange(n_cols)
    elif placement == "cell_corner":
        lats = 90.0 - step * np.arange(n_rows)
        lons = -180.0 + step * np.arange(n_cols)
    else:
        raise ValueError(f"unknown placement {placement!r} (use cell_center or cell_corner)")
    return lats, lons


def make_grid(grid: GridSpec) -> list[Point]:
    """All points of the grid in row-major order, or a seeded subset of them (still idx-sorted)."""
    step = float(grid.step_deg)
    lats, lons = _axes(step, grid.placement)
    n_rows, n_cols = len(lats), len(lons)
    n_total = n_rows * n_cols

    if grid.subset_frac is None:
        chosen = np.arange(n_total)
    else:
        frac = float(grid.subset_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"grid subset_frac must be in (0, 1], got {grid.subset_frac!r}")
        n_keep = max(1, int(round(frac * n_total)))
        rng = np.random.default_rng(int(grid.seed))
        chosen = np.sort(rng.choice(n_total, size=n_keep, replace=False))

    points: list[Point] = []
    for idx in chosen.tolist():
        r, c = divmod(idx, n_cols)
        lat = _clean(lats[r])
        lon = _clean(lons[c])
        w = math.cos(math.radians(lat))
        w = 0.0 if w < 1e-12 else w
        points.append(Point(idx=int(idx), lat=lat, lon=lon, weight=w))
    return points


def _trailing_zeros(x: np.ndarray, cap: int) -> np.ndarray:
    """Number of trailing zero bits of each non-negative int; ``cap`` for zero."""
    x = x.astype(np.int64)
    out = np.full(x.shape, cap, dtype=np.int64)
    nz = x != 0
    low = x[nz] & (-x[nz])  # lowest set bit
    out[nz] = np.round(np.log2(low)).astype(np.int64)
    return np.minimum(out, cap)


def stratified_order(points: list[Point], seed: int) -> list[Point]:
    """Coarse-to-fine visiting order, so any prefix of the run covers the globe evenly.

    Each point gets a (row, col) rank from the distinct lats/lons present. Both ranks are shifted
    by a seeded random offset (cyclically), then a point's *level* is the coarsest power-of-two
    sub-grid that contains it: level 0 is one point, level 1 adds the points at half spacing, and
    so on (a 2-D bit-reversal / progressive sub-grid scheme). Points are ordered by level, with a
    seeded shuffle inside each level. The result is a permutation of ``points``.
    """
    n = len(points)
    if n <= 1:
        return list(points)
    lats = np.array([p.lat for p in points], dtype=float)
    lons = np.array([p.lon for p in points], dtype=float)
    ulat_desc = np.unique(lats)[::-1]
    ulon = np.unique(lons)
    rows = np.searchsorted(-ulat_desc, -lats)
    cols = np.searchsorted(ulon, lons)
    nr, nc = len(ulat_desc), len(ulon)
    k = max(1, int(math.ceil(math.log2(max(nr, nc)))))

    rng = np.random.default_rng(int(seed))
    rows = (rows + int(rng.integers(nr))) % nr
    cols = (cols + int(rng.integers(nc))) % nc

    tz = np.minimum(_trailing_zeros(rows, k), _trailing_zeros(cols, k))
    level = k - tz
    tiebreak = rng.random(n)
    order = np.lexsort((tiebreak, level))
    return [points[i] for i in order.tolist()]


def cell_bounds(
    point: Point, step_deg: float, placement: Placement
) -> tuple[float, float, float, float]:
    """Bounds of the cell a point stands for, as ``(west, south, east, north)`` in degrees.

    Same order as rasterio/shapely bounds (minx, miny, maxx, maxy). Cell-center: the point is the
    middle of the cell. Cell-corner: the point is the north-west corner. Latitudes are clipped to
    [-90, 90]; longitudes are not wrapped (east may be 180, never beyond for valid grids).
    """
    s = float(step_deg)
    if placement == "cell_center":
        w, e = point.lon - s / 2.0, point.lon + s / 2.0
        south, north = point.lat - s / 2.0, point.lat + s / 2.0
    elif placement == "cell_corner":
        w, e = point.lon, point.lon + s
        south, north = point.lat - s, point.lat
    else:
        raise ValueError(f"unknown placement {placement!r}")
    south = max(-90.0, south)
    north = min(90.0, north)
    return (_clean(w), _clean(south), _clean(e), _clean(north))
