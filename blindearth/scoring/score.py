"""Score one stored run and save the metrics back to the store.

Works on partial runs: only the points stored so far are scored, and ``n_points`` /
``n_points_total`` / ``partial`` record how much of the grid that is.

If the mask passed in is not the one the run's eval spec was planned with (different hash), or
the stored points have no truth, truth is recomputed from the mask with the spec's truth rule, so
old runs can be re-scored against a new ground-truth image without new API calls.

Saved/returned dict: every key of ``cell_metrics`` at the top level, plus
``regions`` (label -> accuracy), ``acc_area_ci`` [est, lo, hi], ``image`` (``image_metrics`` dict
or None), and run context (``run_id``, ``variant``, ``extraction_mode``, ``mask_hash``,
``mask_source``, ``n_points_total``, ``partial``, ``cost_usd``, ``scored_at``). NaN becomes None
so the dict is JSON-safe.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from blindearth.scoring.bootstrap import acc_area_stat, block_bootstrap_ci
from blindearth.scoring.cell_metrics import cell_metrics
from blindearth.scoring.image_metrics import image_metrics
from blindearth.scoring.regions import region_accuracy
from blindearth.types import ExtractionMode, Point

if TYPE_CHECKING:
    from blindearth.evalspec.masks import Mask
    from blindearth.store.db import Store


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    return x


def _mode_value(mode: Any) -> str:
    return str(getattr(mode, "value", mode))


def is_probabilistic(mode: Any, df: pd.DataFrame) -> bool:
    """Logprobs runs, and sampling runs with more than one sample per point."""
    m = _mode_value(mode)
    if m == ExtractionMode.LOGPROBS.value:
        return True
    if m == ExtractionMode.SAMPLE.value and "n_samples" in df.columns and len(df):
        ns = pd.to_numeric(df["n_samples"], errors="coerce").to_numpy(float)
        return bool(np.nanmax(ns) > 1) if np.isfinite(ns).any() else False
    return False


def _recompute_truth(df: pd.DataFrame, mask: Any, spec: Any) -> pd.DataFrame:
    from blindearth.evalspec.truth import cell_truth

    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(float)
    w = pd.to_numeric(df["weight"], errors="coerce").to_numpy(float) if "weight" in df else np.cos(np.deg2rad(lat))
    w = np.where(np.isfinite(w), w, np.cos(np.deg2rad(lat)))
    pts = [Point(int(i), float(a), float(o), float(ww)) for i, a, o, ww in zip(df["idx"], lat, lon, w)]
    t = np.asarray(cell_truth(mask, pts, spec.grid, spec.mask.truth_rule), dtype=float)
    t[t < 0] = np.nan
    df = df.copy()
    df["truth"] = t
    return df


def score_run(
    store: "Store",
    run_id: str,
    mask: "Mask | None" = None,
    threshold: float = 0.5,
    with_images: bool = True,
) -> dict:
    """Compute cell, region, bootstrap and (optionally) image metrics; save and return them."""
    run = store.get_run(run_id)
    run_id = run.id  # get_run accepts a unique prefix; key everything by the full id
    spec, spec_mask_hash, spec_mask_source = store.get_eval_spec(run.spec_id)
    df = store.load_points(run_id)
    if mask is None:
        from blindearth.evalspec.masks import load_mask

        mask = load_mask(spec.mask)

    if len(df):
        truth = pd.to_numeric(df["truth"], errors="coerce") if "truth" in df else None
        if mask.hash != spec_mask_hash or truth is None or not truth.notna().any():
            df = _recompute_truth(df, mask, spec)

    step = float(spec.grid.step_deg)
    placement = spec.grid.placement
    probabilistic = is_probabilistic(run.extraction_mode, df)

    metrics: dict[str, Any] = cell_metrics(
        df, threshold, probabilistic, step_deg=step, placement=placement
    )
    metrics["regions"] = region_accuracy(df, threshold)
    metrics["regions_method"] = getattr(metrics["regions"], "method", None)
    est, lo, hi = block_bootstrap_ci(df, acc_area_stat(threshold)) if len(df) else (None, None, None)
    metrics["acc_area_ci"] = [est, lo, hi]
    metrics["image"] = (
        image_metrics(
            df, mask, step, placement, threshold,
            truth_rule=spec.mask.truth_rule, probabilistic=probabilistic,
        )
        if with_images and len(df)
        else None
    )

    n_total = int(getattr(run, "n_points_total", 0) or 0)
    metrics.update(
        run_id=run_id,
        variant=run.variant,
        extraction_mode=_mode_value(run.extraction_mode),
        run_status=_mode_value(run.status),
        mask_hash=mask.hash,
        mask_source=getattr(mask, "source", spec_mask_source),
        truth_rule=spec.mask.truth_rule,
        n_points_total=n_total,
        partial=bool(n_total and len(df) < n_total) or _mode_value(run.status) != "complete",
        cost_usd=run.cost_usd,
        thinking=bool(getattr(run, "thinking", False)),
        forced_thinking=bool(getattr(run, "forced_thinking", False)),
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
    metrics = _jsonable(metrics)
    store.save_metrics(run_id, mask.hash, threshold, metrics)
    return metrics
