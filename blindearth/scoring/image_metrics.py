"""Image comparison of a run against the ground-truth mask (spec "Image comparison").

Everything is computed at a working resolution (default 1800 x 3600, ~11 km at the equator):
masks larger than that are downsampled first (block majority when the shape divides evenly,
otherwise nearest sampling); smaller masks are used as they are.

Prediction images:
- binary: P(Land) > threshold, nearest-neighbour upsampling (blockiness counts against the model);
  invalid answers and points not yet run render as water.
- probability: bilinear upsampling of P(Land) (missing -> 0).

Metrics (all at working resolution):
- ``pixel_iou``, ``dice``: Land overlap weighted by cos(lat) per pixel row.
- ``ssim``: skimage SSIM of the probability image against the mask (data_range 1).
- ``boundary_f@T``: boundary pixels = pixels that differ from a 4-neighbour (lon wraps). Precision
  = share of predicted boundary within T degrees of the true boundary, recall the reverse,
  F = harmonic mean. ``boundary_p@T`` / ``boundary_r@T`` are reported too.
- ``contour_mean_*`` / ``contour_p95_*``: distance of each predicted boundary pixel to the true
  boundary (EDT of the true boundary, wrap-padded across ±180°). Degrees are plate-carrée degrees;
  km = degrees x 111.195 (meridian degree), so east-west offsets at high latitude are overstated.
- connected components: 8-connectivity, with labels merged across the ±180° seam.
  ``spurious_specks`` = predicted components that touch no true land.
- ``per_landmass_iou``: for each true component of at least ``min_landmass_km2`` (largest first,
  at most ``max_landmasses``), the cos-weighted IoU with the predicted component that overlaps it
  most. Names come from ``regions.land_continent`` of the component's pixels.
- ``prob_rmse``: cos-weighted RMSE between the probability image and the mask (probabilistic only).

Resolution ceiling: the mask is reduced to the eval grid with the eval's truth rule (via
``evalspec.truth.cell_truth``, or a ``truth_grid`` passed in), rendered back the same way and
scored with the same code. ``share_of_ceiling`` = model / ceiling for higher-is-better metrics
(IoU, Dice, SSIM, boundary F) and ceiling / model for distances and RMSE, so 1.0 means "as good
as a perfect model at this grid".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from blindearth.scoring.coastline import (
    EARTH_RADIUS_KM,
    KM_PER_DEG,
    boundary,
    boundary_f_score,
    distance_to,
)
from blindearth.scoring.render import points_to_grid, render_to_shape, values_to_grid
from blindearth.types import Placement, TruthRule

HIGHER_BETTER = ("pixel_iou", "dice", "ssim")
LOWER_BETTER = ("contour_mean_deg", "contour_p95_deg", "contour_mean_km", "contour_p95_km", "prob_rmse")
_EIGHT = np.ones((3, 3), dtype=bool)


# --------------------------------------------------------------------------- helpers


def downsample_mask(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reduce a bool mask to at most ``shape`` (block majority if divisible, else nearest)."""
    m = np.asarray(data, dtype=bool)
    H, W = m.shape
    h, w = min(int(shape[0]), H), min(int(shape[1]), W)
    if (h, w) == (H, W):
        return m
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        out = np.empty((h, w), dtype=bool)
        chunk = max(1, 1024 // fy)  # output rows per chunk keeps temporaries small
        for r0 in range(0, h, chunk):
            r1 = min(h, r0 + chunk)
            block = m[r0 * fy : r1 * fy].reshape(r1 - r0, fy, w, fx)
            out[r0:r1] = block.sum(axis=(1, 3), dtype=np.int64) * 2 >= fy * fx
        return out
    ri = ((np.arange(h) + 0.5) * H / h).astype(np.int64)
    ci = ((np.arange(w) + 0.5) * W / w).astype(np.int64)
    return m[np.ix_(np.clip(ri, 0, H - 1), np.clip(ci, 0, W - 1))]


def row_weights(h: int) -> np.ndarray:
    """cos(lat) at each pixel-row center."""
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    return np.cos(np.deg2rad(lat))


def row_area_km2(h: int, w: int) -> np.ndarray:
    return EARTH_RADIUS_KM**2 * (np.pi / h) * (2 * np.pi / w) * row_weights(h)


def _wsum(b: np.ndarray, rw: np.ndarray) -> float:
    return float((b.sum(axis=1, dtype=np.int64) * rw).sum())


def label_wrapped(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels with the ±180° seam treated as continuous."""
    b = np.asarray(binary, dtype=bool)
    labels, n = ndimage.label(b, structure=_EIGHT)
    if n == 0 or b.shape[1] < 2:
        return labels, n
    left, right = labels[:, 0], labels[:, -1]
    pairs = []
    h = b.shape[0]
    for dr in (-1, 0, 1):
        r = np.arange(max(0, -dr), min(h, h - dr))
        l_ = left[r]
        r_ = right[r + dr]
        ok = (l_ > 0) & (r_ > 0)
        if ok.any():
            pairs.append(np.stack([l_[ok], r_[ok]], axis=1))
    if not pairs:
        return labels, n
    p = np.unique(np.concatenate(pairs), axis=0)
    g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n + 1, n + 1))
    _, comp = connected_components(g, directed=False)
    _, inv = np.unique(comp[1:], return_inverse=True)
    remap = np.concatenate([[0], inv + 1]).astype(labels.dtype)
    return remap[labels], int(inv.max()) + 1


def _component_areas(labels: np.ndarray, n: int, area_row: np.ndarray) -> np.ndarray:
    """Area in km² of components 1..n (index 0 = background)."""
    w = labels.shape[1]
    return np.bincount(labels.ravel(), weights=np.repeat(area_row, w), minlength=n + 1)


def _name_landmass(mask_k: np.ndarray, max_samples: int = 4000) -> tuple[str, float, float]:
    from blindearth.scoring.regions import land_continent

    h, w = mask_k.shape
    rows, cols = np.nonzero(mask_k)
    if rows.size > max_samples:
        sel = np.linspace(0, rows.size - 1, max_samples).astype(np.int64)
        rows, cols = rows[sel], cols[sel]
    lat = 90.0 - (rows + 0.5) * 180.0 / h
    lon = -180.0 + (cols + 0.5) * 360.0 / w
    labels = land_continent(lat, lon)
    names, counts = np.unique(labels, return_counts=True)
    share = counts / counts.sum()
    keep = [str(n) for n, s in sorted(zip(names, share), key=lambda x: -x[1]) if s >= 0.15]
    name = " + ".join(keep) if keep else str(names[np.argmax(counts)])
    # circular mean longitude so a seam-straddling landmass gets a sensible centroid
    ang = np.deg2rad(lon)
    c_lon = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())))
    return name, float(lat.mean()), c_lon


class _Truth:
    """Precomputed truth-side quantities shared by the model and the ceiling."""

    def __init__(self, true_img: np.ndarray, min_landmass_km2: float, max_landmasses: int):
        h, w = true_img.shape
        self.img = true_img
        self.imgf = true_img.astype(np.float32)
        self.rw = row_weights(h)
        self.area_row = row_area_km2(h, w)
        self.dy, self.dx = 180.0 / h, 360.0 / w
        self.pad = w // 4
        self.bnd = boundary(true_img)
        self.dist = distance_to(self.bnd, self.dy, self.dx, wrap_pad=self.pad, dtype=np.float32)
        self.labels, self.n = label_wrapped(true_img)
        self.areas = _component_areas(self.labels, self.n, self.area_row)
        big = np.argsort(-self.areas[1:])[:max_landmasses] + 1 if self.n else np.array([], int)
        self.landmasses: list[tuple[int, str, float, float, float]] = []
        seen: dict[str, int] = {}
        for k in big:
            if self.areas[k] < min_landmass_km2:
                break
            name, clat, clon = _name_landmass(self.labels == k)
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                name = f"{name} #{seen[name]}"
            self.landmasses.append((int(k), name, float(self.areas[k]), clat, clon))


def _score(pred: np.ndarray, prob: np.ndarray, T: _Truth, tolerances: tuple[float, ...],
           probabilistic: bool, min_speck_km2: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t = T.img
    inter = _wsum(pred & t, T.rw)
    p_sum = _wsum(pred, T.rw)
    t_sum = _wsum(t, T.rw)
    union = p_sum + t_sum - inter
    out["pixel_iou"] = float(inter / union) if union > 0 else 1.0
    out["dice"] = float(2 * inter / (p_sum + t_sum)) if (p_sum + t_sum) > 0 else 1.0

    from skimage.metrics import structural_similarity

    out["ssim"] = float(structural_similarity(prob, T.imgf, data_range=1.0))

    pb = boundary(pred)
    d_pred = distance_to(pb, T.dy, T.dx, wrap_pad=T.pad, dtype=np.float32)
    for tol in tolerances:
        f, p, r = boundary_f_score(pb, T.bnd, T.dist, d_pred, tol)
        out[f"boundary_f@{tol:g}"] = f
        out[f"boundary_p@{tol:g}"] = p
        out[f"boundary_r@{tol:g}"] = r
    del d_pred
    if pb.any() and T.dist is not None:
        d = T.dist[pb]
        mean_deg, p95_deg = float(d.mean()), float(np.percentile(d, 95))
        out.update(
            contour_mean_deg=mean_deg, contour_p95_deg=p95_deg,
            contour_mean_km=mean_deg * KM_PER_DEG, contour_p95_km=p95_deg * KM_PER_DEG,
        )
    else:
        out.update(contour_mean_deg=None, contour_p95_deg=None, contour_mean_km=None, contour_p95_km=None)

    labels, n = label_wrapped(pred)
    areas = _component_areas(labels, n, T.area_row)
    out["n_components_pred"] = int(n)
    out["n_components_true"] = int(T.n)
    out["n_components_pred_large"] = int((areas[1:] >= min_speck_km2).sum())
    out["n_components_true_large"] = int((T.areas[1:] >= min_speck_km2).sum())
    overlap = np.bincount(labels[t], minlength=n + 1)
    out["spurious_specks"] = int((overlap[1:] == 0).sum())
    out["largest_components_pred_km2"] = [float(a) for a in np.sort(areas[1:])[::-1][:10]]
    out["largest_components_true_km2"] = [float(a) for a in np.sort(T.areas[1:])[::-1][:10]]

    per: dict[str, float] = {}
    for k, name, _area, _clat, _clon in T.landmasses:
        tk = T.labels == k
        hits = labels[tk]
        hits = hits[hits > 0]
        if hits.size == 0:
            per[name] = 0.0
            continue
        m = int(np.bincount(hits).argmax())
        pk = labels == m
        i = _wsum(tk & pk, T.rw)
        u = _wsum(tk | pk, T.rw)
        per[name] = float(i / u) if u > 0 else 0.0
    out["per_landmass_iou"] = per

    if probabilistic:
        se = ((prob - T.imgf) ** 2).mean(axis=1, dtype=np.float64)
        out["prob_rmse"] = float(np.sqrt((se * T.rw).sum() / T.rw.sum()))
    else:
        out["prob_rmse"] = None
    return out


def _truth_grid_from_mask(mask: Any, step_deg: float, placement: Placement, rule: TruthRule) -> np.ndarray:
    from blindearth.evalspec.grid import make_grid
    from blindearth.evalspec.truth import cell_truth
    from blindearth.types import GridSpec

    grid = GridSpec(step_deg=step_deg, placement=placement)
    pts = make_grid(grid)
    t = np.asarray(cell_truth(mask, pts, grid, rule), dtype=float)
    t[t < 0] = np.nan
    lat = np.array([p.lat for p in pts], dtype=float)
    lon = np.array([p.lon for p in pts], dtype=float)
    return values_to_grid(lat, lon, t, step_deg, placement)


def _share(model: dict, ceil: dict, tolerances: tuple[float, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    keys_hi = list(HIGHER_BETTER) + [f"boundary_f@{t:g}" for t in tolerances]
    for k in keys_hi:
        m, c = model.get(k), ceil.get(k)
        out[k] = None if m is None or c is None or c <= 0 else float(m / c)
    for k in LOWER_BETTER:
        m, c = model.get(k), ceil.get(k)
        if m is None or c is None:
            out[k] = None
        elif m <= 0:
            out[k] = 1.0
        else:
            out[k] = float(c / m)
    return out


# --------------------------------------------------------------------------- public


def image_metrics(
    df: pd.DataFrame,
    mask: Any,
    step_deg: float,
    placement: Placement,
    threshold: float = 0.5,
    tolerances_deg: tuple[float, ...] = (2.0, 4.0),
    work_shape: tuple[int, int] = (1800, 3600),
    *,
    truth_rule: TruthRule = "cell_center",
    probabilistic: bool = True,
    truth_grid: np.ndarray | None = None,
    min_landmass_km2: float = 5.0e5,
    max_landmasses: int = 12,
) -> dict:
    """Image metrics of one run against ``mask`` (a ``Mask``; only ``mask.data`` is used).

    Extra keyword-only arguments beyond INTERFACES.md: ``truth_rule`` (rule used to build the
    ceiling grid), ``probabilistic`` (``prob_rmse`` is None when False), ``truth_grid`` (a
    precomputed (rows, cols) truth grid for the ceiling; computed from the mask when None),
    ``min_landmass_km2`` / ``max_landmasses`` (which true landmasses get a per-landmass IoU).
    """
    data = np.asarray(getattr(mask, "data", mask), dtype=bool)
    true_img = downsample_mask(data, work_shape)
    shape = true_img.shape
    T = _Truth(true_img, min_landmass_km2, max_landmasses)
    tolerances = tuple(float(t) for t in tolerances_deg)
    # A speck threshold of one grid cell at the equator.
    min_speck_km2 = (step_deg * KM_PER_DEG) ** 2

    bin_grid = points_to_grid(df, step_deg, placement, "binary", threshold)
    p_grid = points_to_grid(df, step_deg, placement, "p_land", threshold)
    coverage = float(np.isfinite(bin_grid).mean())
    pred = render_to_shape(np.nan_to_num(bin_grid, nan=0.0) > 0.5, shape, step_deg, placement, binary=True)
    prob = render_to_shape(p_grid, shape, step_deg, placement, binary=False)
    model = _score(pred, prob, T, tolerances, probabilistic, min_speck_km2)
    del pred, prob

    if truth_grid is None:
        truth_grid = _truth_grid_from_mask(mask, step_deg, placement, truth_rule)
    tg = np.asarray(truth_grid, dtype=float)
    c_pred = render_to_shape(np.nan_to_num(tg, nan=0.0) > 0.5, shape, step_deg, placement, binary=True)
    c_prob = render_to_shape(np.clip(tg, 0.0, 1.0), shape, step_deg, placement, binary=False)
    ceiling = _score(c_pred, c_prob, T, tolerances, probabilistic, min_speck_km2)
    del c_pred, c_prob

    out = dict(model)
    out["ceiling"] = ceiling
    out["share_of_ceiling"] = _share(model, ceiling, tolerances)
    out["landmasses"] = [
        {"name": name, "area_km2": area, "centroid_lat": clat, "centroid_lon": clon}
        for _k, name, area, clat, clon in T.landmasses
    ]
    out["work_shape"] = [int(shape[0]), int(shape[1])]
    out["coverage"] = coverage
    out["tolerances_deg"] = list(tolerances)
    return out
