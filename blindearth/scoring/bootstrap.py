"""Block bootstrap confidence intervals over 10° x 10° blocks of cells.

Neighbouring cells are correlated, so whole blocks are resampled with replacement (as many blocks
as are present) and the percentile interval is taken.

Fast path: a stat that is a ratio of per-point sums (``RatioStat``, e.g. area-weighted accuracy) is
bootstrapped fully vectorized: per-block sums are computed once and each resample is a gather +
sum over a (n_boot, n_blocks) index matrix. Any other callable falls back to one call per resample
on ``df.iloc[...]`` (indices are built vectorized, the stat itself is not).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from blindearth.scoring.cell_metrics import point_arrays


@dataclass
class RatioStat:
    """stat(df) = sum(num(df)) / sum(den(df)); bootstrapped on the vectorized fast path."""

    parts: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]]

    def __call__(self, df: pd.DataFrame) -> float:
        num, den = self.parts(df)
        d = float(np.sum(den))
        return float("nan") if d <= 0 else float(np.sum(num) / d)


def acc_area_stat(threshold: float = 0.5) -> RatioStat:
    """Area-weighted accuracy (invalid = wrong, unknown truth dropped) as a RatioStat."""

    def parts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        a = point_arrays(df, threshold)
        w = np.where(a["known"], a["w"], 0.0)
        return w * a["correct"], w

    return RatioStat(parts)


def acc_unweighted_stat(threshold: float = 0.5) -> RatioStat:
    def parts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        a = point_arrays(df, threshold)
        k = a["known"].astype(float)
        return k * a["correct"], k

    return RatioStat(parts)


def block_ids(lat: np.ndarray, lon: np.ndarray, block_deg: float = 10.0) -> np.ndarray:
    """Integer block id per point (row-major over block_deg x block_deg tiles, lon wraps)."""
    n_r = int(math.ceil(180.0 / block_deg - 1e-9))
    n_c = int(math.ceil(360.0 / block_deg - 1e-9))
    r = np.clip(np.floor((90.0 - np.asarray(lat, float)) / block_deg).astype(np.int64), 0, n_r - 1)
    c = np.mod(np.floor((np.asarray(lon, float) + 180.0) / block_deg).astype(np.int64), n_c)
    return r * n_c + c


def _percentile_ci(boot: np.ndarray, alpha: float) -> tuple[float, float]:
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def _ratio_boot(num: np.ndarray, den: np.ndarray, blocks: np.ndarray, n_boot: int,
                rng: np.random.Generator) -> np.ndarray:
    _, inv = np.unique(blocks, return_inverse=True)
    nb = int(inv.max()) + 1
    num_b = np.bincount(inv, weights=num, minlength=nb)
    den_b = np.bincount(inv, weights=den, minlength=nb)
    pick = rng.integers(0, nb, size=(n_boot, nb))
    nsum = num_b[pick].sum(axis=1)
    dsum = den_b[pick].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dsum > 0, nsum / dsum, np.nan)


def _generic_boot(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], blocks: np.ndarray,
                  n_boot: int, rng: np.random.Generator) -> np.ndarray:
    _, inv = np.unique(blocks, return_inverse=True)
    nb = int(inv.max()) + 1
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=nb)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, nb, size=nb)
        lengths = counts[chosen]
        total = int(lengths.sum())
        base = np.repeat(starts[chosen], lengths)
        within = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        out[i] = stat(df.iloc[order[base + within]])
    return out


def block_bootstrap_ci(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    *,
    block_deg: float = 10.0,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(estimate, lo, hi): percentile interval from resampling blocks with replacement."""
    if df is None or len(df) == 0:
        return float("nan"), float("nan"), float("nan")
    est = float(stat(df))
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(float)
    blocks = block_ids(lat, lon, block_deg)
    rng = np.random.default_rng(seed)
    if isinstance(stat, RatioStat):
        num, den = stat.parts(df)
        boot = _ratio_boot(np.asarray(num, float), np.asarray(den, float), blocks, n_boot, rng)
    else:
        boot = _generic_boot(df, stat, blocks, n_boot, rng)
    lo, hi = _percentile_ci(boot, alpha)
    return est, lo, hi


def _aligned(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = df_a.drop_duplicates("idx", keep="last").set_index("idx")
    b = df_b.drop_duplicates("idx", keep="last").set_index("idx")
    common = a.index.intersection(b.index).sort_values()
    a = a.loc[common]
    b = b.loc[common]
    for col in ("lat", "lon"):
        va = pd.to_numeric(a[col], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(b[col], errors="coerce").to_numpy(float)
        if va.size and not np.allclose(va, vb, atol=1e-6, equal_nan=True):
            raise ValueError(f"runs are on different grids: {col} differs for the same idx")
    ta = pd.to_numeric(a["truth"], errors="coerce").to_numpy(float) if "truth" in a else None
    tb = pd.to_numeric(b["truth"], errors="coerce").to_numpy(float) if "truth" in b else None
    if ta is not None and tb is not None:
        both = np.isfinite(ta) & np.isfinite(tb)
        if np.any(ta[both] != tb[both]):
            raise ValueError("runs were scored against different truth (mask or truth rule differs)")
    return a.reset_index(), b.reset_index()


def paired_difference(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    threshold: float = 0.5,
    block_deg: float = 10.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Area-weighted accuracy of A minus B on the points both runs have, with a paired block CI.

    Raises ValueError if the runs are on different grids or truths. ``significant`` is True only
    when the 95% interval excludes zero.
    """
    a, b = _aligned(df_a, df_b)
    n = len(a)
    if n == 0:
        return {"diff": None, "lo": None, "hi": None, "significant": False, "n_paired": 0,
                "acc_a": None, "acc_b": None}
    pa = point_arrays(a, threshold)
    pb = point_arrays(b, threshold)
    known = pa["known"] & pb["known"]
    w = np.where(known, pa["w"], 0.0)
    num = w * (pa["correct"].astype(float) - pb["correct"].astype(float))
    den = w
    sw = float(den.sum())
    if sw <= 0:
        return {"diff": None, "lo": None, "hi": None, "significant": False, "n_paired": n,
                "acc_a": None, "acc_b": None}
    diff = float(num.sum() / sw)
    rng = np.random.default_rng(seed)
    boot = _ratio_boot(num, den, block_ids(pa["lat"], pa["lon"], block_deg), n_boot, rng)
    lo, hi = _percentile_ci(boot, 0.05)
    significant = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    return {
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "significant": significant,
        "n_paired": n,
        "acc_a": float((w * pa["correct"]).sum() / sw),
        "acc_b": float((w * pb["correct"]).sum() / sw),
    }


def repeat_spread(dfs: list[pd.DataFrame], threshold: float = 0.5) -> dict:
    """Run-to-run spread of area-weighted accuracy over repeats of one configuration."""
    stat = acc_area_stat(threshold)
    vals = np.array([stat(d) for d in dfs if d is not None and len(d)], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "values": [float(v) for v in vals],
    }
