"""Boundary and distance helpers shared by cell metrics and image metrics.

All arrays are equirectangular, row 0 = north, col 0 = 180°W. Longitude is periodic: the
±180° seam is handled by comparing with ``np.roll`` for boundaries and by wrap-padding the
columns before the scipy Euclidean distance transform, so a coastline just east of the seam
is ~1 cell away from a boundary just west of it, not ~360°.

Distances are plate-carrée distances in degrees (sampling = degrees per row / per column).
They are not great-circle distances: east-west separations at high latitude are overstated
by 1/cos(lat). ``KM_PER_DEG`` converts degrees of latitude to km.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG = EARTH_RADIUS_KM * np.pi / 180.0  # ~111.195 km per degree along a meridian


def boundary(binary: np.ndarray, known: np.ndarray | None = None) -> np.ndarray:
    """Cells that differ from at least one 4-neighbour (both sides of the coast are marked).

    Longitude wraps; latitude does not. When ``known`` is given, only pairs where both cells are
    known count, so missing cells (partial runs, invalid answers) never create a boundary.
    """
    b = np.asarray(binary, dtype=bool)
    k = np.ones_like(b) if known is None else np.asarray(known, dtype=bool)
    out = np.zeros_like(b)
    if b.ndim != 2 or b.size == 0:
        return out
    if b.shape[1] > 1:
        # pair (j, j+1 mod W)
        diff = (b != np.roll(b, -1, axis=1)) & k & np.roll(k, -1, axis=1)
        out |= diff
        out |= np.roll(diff, 1, axis=1)
    if b.shape[0] > 1:
        d = (b[:-1] != b[1:]) & k[:-1] & k[1:]
        out[:-1] |= d
        out[1:] |= d
    return out


def distance_to(
    targets: np.ndarray,
    step_y: float,
    step_x: float,
    *,
    wrap_pad: int | None = None,
    dtype: type = np.float64,
) -> np.ndarray | None:
    """Distance from every cell to the nearest True cell of ``targets``.

    Units are those of ``step_y``/``step_x`` (degrees per row/column). Columns are wrap-padded by
    ``wrap_pad`` cells on each side (default: half the width, which makes the seam exact) so the
    distance is periodic in longitude. Returns None when there are no targets.
    """
    t = np.asarray(targets, dtype=bool)
    if t.size == 0 or not t.any():
        return None
    _, w = t.shape
    pad = w // 2 if wrap_pad is None else max(0, min(int(wrap_pad), w))
    padded = np.pad(t, ((0, 0), (pad, pad)), mode="wrap") if pad else t
    d = ndimage.distance_transform_edt(~padded, sampling=(float(step_y), float(step_x)))
    if pad:
        d = d[:, pad : pad + w]
    return np.ascontiguousarray(d, dtype=dtype)


def coastline_error(
    pred: np.ndarray,
    truth: np.ndarray,
    step_deg: float,
    *,
    pred_known: np.ndarray | None = None,
    truth_known: np.ndarray | None = None,
) -> float | None:
    """Mean distance (degrees) from each predicted boundary cell to the nearest true coastline cell.

    Both grids are bool (H, W) at the eval grid. Returns None if the prediction has no boundary
    (for example an all-water map) or the truth has no coastline.
    """
    pb = boundary(pred, pred_known)
    if not pb.any():
        return None
    tb = boundary(truth, truth_known)
    d = distance_to(tb, step_deg, step_deg)
    if d is None:
        return None
    return float(d[pb].mean())


def boundary_f_score(
    pred_b: np.ndarray,
    true_b: np.ndarray,
    d_to_true: np.ndarray | None,
    d_to_pred: np.ndarray | None,
    tol: float,
) -> tuple[float, float, float]:
    """(F, precision, recall) of boundary pixels within ``tol`` of the other boundary."""
    n_pred = int(pred_b.sum())
    n_true = int(true_b.sum())
    if n_pred == 0 and n_true == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0 or n_true == 0 or d_to_true is None or d_to_pred is None:
        return 0.0, 0.0, 0.0
    precision = float((d_to_true[pred_b] <= tol).mean())
    recall = float((d_to_pred[true_b] <= tol).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f, precision, recall
