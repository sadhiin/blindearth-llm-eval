"""Cell-level metrics on a points table (spec "Scoring and metrics").

Conventions:
- Every metric except ``acc_unweighted`` and ``invalid_rate`` is area-weighted by the ``weight``
  column (cos(lat); recomputed from ``lat`` when missing). Land precision/recall/F1/IoU are
  area-weighted too, matching the headline accuracy and the image IoU.
- A point is *valid* when ``p_land`` is a number. Invalid/error points count as wrong in
  ``acc_area`` / ``acc_unweighted`` and as "not predicted land" in the Land-class metrics (so an
  invalid answer on a land cell is a false negative). ``acc_valid_only`` is the area-weighted
  accuracy over valid points only.
- Points with unknown truth (NaN) are dropped before scoring and counted in ``n_unknown_truth``.
- Brier, log loss (p clipped to [1e-6, 1-1e-6]) and ECE (10 equal-width bins) use valid points
  only and are None unless ``probabilistic``.
- Coastline error: grids are built at the eval step; a cell is a boundary cell if it differs from
  a 4-neighbour (lon wraps at ±180°). Invalid/missing predictions and unknown truths are treated as
  unknown, so they never create boundaries. Value = mean plate-carrée distance in degrees from each
  predicted boundary cell to the nearest true coastline cell (scipy EDT, wrap-padded columns).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from blindearth.scoring.coastline import coastline_error
from blindearth.scoring.render import infer_placement, infer_step, values_to_grid
from blindearth.types import Placement

LOG_LOSS_EPS = 1e-6
ECE_BINS = 10


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def point_arrays(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Per-point arrays used by every scorer.

    Keys: lat, lon, w, truth (float, NaN unknown), t_land (bool), p (float, NaN invalid),
    valid, pred_land, known, correct (bool; invalid -> False).
    """
    lat = _col(df, "lat")
    lon = _col(df, "lon")
    w = _col(df, "weight")
    bad_w = ~np.isfinite(w)
    if bad_w.any():
        w = np.where(bad_w, np.cos(np.deg2rad(lat)), w)
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
    truth = _col(df, "truth")
    p = _col(df, "p_land")
    known = np.isfinite(truth)
    t_land = truth > 0.5
    valid = np.isfinite(p)
    pred_land = valid & (p > threshold)
    correct = valid & known & (pred_land == t_land)
    return {
        "lat": lat,
        "lon": lon,
        "w": w,
        "truth": truth,
        "t_land": t_land,
        "p": p,
        "valid": valid,
        "pred_land": pred_land,
        "known": known,
        "correct": correct,
    }


def _ratio(num: float, den: float) -> float | None:
    return None if den <= 0 else float(num / den)


def acc_area(df: pd.DataFrame, threshold: float = 0.5) -> float | None:
    """Area-weighted accuracy with invalid counted as wrong (points with unknown truth dropped)."""
    a = point_arrays(df, threshold)
    k = a["known"]
    return _ratio(float((a["w"] * a["correct"])[k].sum()), float(a["w"][k].sum()))


def calibration_bins(p: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int = ECE_BINS) -> list[dict]:
    """Reliability-diagram bins: [{lo, hi, mean_p, frac_land, weight, n}] (empty bins omitted)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for i in range(n_bins):
        m = b == i
        if not m.any():
            continue
        wi = w[m]
        sw = float(wi.sum())
        if sw <= 0:
            continue
        out.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "mean_p": float((wi * p[m]).sum() / sw),
                "frac_land": float((wi * y[m]).sum() / sw),
                "weight": sw,
                "n": int(m.sum()),
            }
        )
    return out


def _empty(threshold: float, probabilistic: bool) -> dict[str, Any]:
    keys = [
        "acc_area", "acc_unweighted", "acc_valid_only", "baseline_area", "skill", "precision",
        "recall", "f1", "iou", "brier", "log_loss", "ece", "invalid_rate", "coastline_error_deg",
        "latency_p50", "latency_p95", "land_share_true", "land_share_pred",
    ]
    d: dict[str, Any] = {k: None for k in keys}
    d.update(
        n_points=0, n_valid=0, n_unknown_truth=0, input_tokens=0, output_tokens=0,
        thinking_tokens=0, threshold=threshold, probabilistic=probabilistic, calibration_bins=None,
    )
    return d


def cell_metrics(
    df: pd.DataFrame,
    threshold: float = 0.5,
    probabilistic: bool = True,
    *,
    step_deg: float | None = None,
    placement: Placement | None = None,
) -> dict:
    """All cell-level metrics for one run (see module docstring).

    ``step_deg``/``placement`` are only used for the coastline error; they are inferred from the
    point coordinates when not given.
    """
    out = _empty(threshold, probabilistic)
    n_all = 0 if df is None else len(df)
    out["n_points"] = n_all
    if n_all == 0:
        return out

    # Run-level cost/speed figures use every point, including ones with unknown truth.
    for col in ("input_tokens", "output_tokens", "thinking_tokens"):
        out[col] = int(np.nansum(_col(df, col)))
    lat_s = _col(df, "latency_s")
    lat_s = lat_s[np.isfinite(lat_s)]
    if lat_s.size:
        out["latency_p50"] = float(np.percentile(lat_s, 50))
        out["latency_p95"] = float(np.percentile(lat_s, 95))

    a = point_arrays(df, threshold)
    out["invalid_rate"] = float((~a["valid"]).mean())
    out["n_valid"] = int(a["valid"].sum())
    k = a["known"]
    out["n_unknown_truth"] = int((~k).sum())
    if not k.any():
        return out

    w = a["w"][k]
    t = a["t_land"][k]
    valid = a["valid"][k]
    pred = a["pred_land"][k]
    correct = a["correct"][k]
    p = a["p"][k]
    sw = float(w.sum())
    if sw <= 0:
        return out

    acc = float((w * correct).sum() / sw)
    out["acc_area"] = acc
    out["acc_unweighted"] = float(correct.mean())
    out["acc_valid_only"] = _ratio(float((w * correct)[valid].sum()), float(w[valid].sum()))
    baseline = float((w * ~t).sum() / sw)  # answer Water everywhere
    out["baseline_area"] = baseline
    out["skill"] = None if baseline >= 1.0 else float((acc - baseline) / (1.0 - baseline))
    out["land_share_true"] = float((w * t).sum() / sw)
    out["land_share_pred"] = float((w * pred).sum() / sw)

    tp = float((w * (pred & t)).sum())
    fp = float((w * (pred & ~t)).sum())
    fn = float((w * (~pred & t)).sum())
    out["precision"] = _ratio(tp, tp + fp)
    out["recall"] = _ratio(tp, tp + fn)
    out["f1"] = _ratio(2 * tp, 2 * tp + fp + fn)
    out["iou"] = _ratio(tp, tp + fp + fn)

    if probabilistic and valid.any():
        pv = np.clip(p[valid], 0.0, 1.0)
        yv = t[valid].astype(float)
        wv = w[valid]
        swv = float(wv.sum())
        if swv > 0:
            out["brier"] = float((wv * (pv - yv) ** 2).sum() / swv)
            pc = np.clip(pv, LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
            ll = -(yv * np.log(pc) + (1.0 - yv) * np.log(1.0 - pc))
            out["log_loss"] = float((wv * ll).sum() / swv)
            bins = calibration_bins(pv, yv, wv)
            out["ece"] = float(sum(b["weight"] * abs(b["mean_p"] - b["frac_land"]) for b in bins) / swv)
            out["calibration_bins"] = bins

    # Coastline error on the eval grid.
    lat = a["lat"]
    lon = a["lon"]
    step = float(step_deg) if step_deg else infer_step(lat, lon)
    plc: Placement = placement or infer_placement(lat, step)
    pred_grid = values_to_grid(
        lat, lon, np.where(a["valid"], a["pred_land"].astype(float), np.nan), step, plc
    )
    truth_grid = values_to_grid(lat, lon, a["truth"], step, plc)
    out["coastline_error_deg"] = coastline_error(
        np.nan_to_num(pred_grid, nan=0.0) > 0.5,
        np.nan_to_num(truth_grid, nan=0.0) > 0.5,
        step,
        pred_known=np.isfinite(pred_grid),
        truth_known=np.isfinite(truth_grid),
    )
    out["step_deg"] = step
    out["placement"] = plc
    return out
