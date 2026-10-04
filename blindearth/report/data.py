"""Load a saved comparison into report-ready RunViews.

Reports read only from the store: points, cached metrics and run records. Metrics that are not
cached (or are stale because a partial run has grown) are computed with
`scoring.score.score_run`, which also saves them, so the next report is free.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from blindearth.types import EvalSpec, ExtractionMode, ModelSpec, Point, RunRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from blindearth.evalspec.masks import Mask
    from blindearth.providers.registry import Registry
    from blindearth.store.db import Store

# Metrics that need a real probability per point. A comparison that mixes extraction modes
# never ranks on these (spec: fairness rules).
PROB_METRICS = ("brier", "log_loss", "ece")

# Canonical ordering for the effort axis; unknown provider-native levels sort after these.
EFFORT_ORDER = ("default", "off", "none", "minimal", "low", "medium", "high", "xhigh", "max")

POINT_COLUMNS = (
    "idx", "lat", "lon", "weight", "truth", "p_land", "n_valid", "n_samples", "validity_mass",
    "answer_text", "finish_reason", "latency_s", "input_tokens", "output_tokens",
    "thinking_tokens", "error",
)


class ComparisonError(ValueError):
    """A comparison that cannot be reported fairly (e.g. runs from different eval specs)."""


# --------------------------------------------------------------------------- variant parsing


def parse_variant(variant: str) -> tuple[str, dict[str, str]]:
    """`opus-5-5@effort=low,temperature=1.0` -> ("opus-5-5", {"effort": "low", ...})."""
    model_id, sep, rest = variant.partition("@")
    cfg: dict[str, str] = {}
    if sep and rest:
        for part in rest.split(","):
            k, eq, v = part.partition("=")
            if eq:
                cfg[k.strip()] = v.strip()
    return model_id, cfg


def effort_sort_key(effort: str) -> tuple[int, str]:
    try:
        return (EFFORT_ORDER.index(effort), effort)
    except ValueError:
        return (len(EFFORT_ORDER), effort)


# --------------------------------------------------------------------------- per-point helpers


def point_weights(df: pd.DataFrame) -> np.ndarray:
    if "weight" in df.columns and df["weight"].notna().all():
        return df["weight"].to_numpy(dtype=float)
    return np.cos(np.radians(df["lat"].to_numpy(dtype=float)))


def _truth_and_pred(df: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (has_truth, truth_is_land, pred_is_land_or_nan) as arrays aligned with df."""
    truth = pd.to_numeric(df["truth"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["p_land"], errors="coerce").to_numpy(dtype=float)
    has_truth = ~np.isnan(truth)
    pred = np.where(np.isnan(p), np.nan, (p > threshold).astype(float))
    return has_truth, truth == 1, pred


def correct_array(df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Boolean per row: thresholded prediction equals truth. Invalid (NaN p_land) is wrong."""
    has_truth, truth_land, pred = _truth_and_pred(df, threshold)
    valid = ~np.isnan(pred)
    return has_truth & valid & ((pred == 1) == truth_land)


def area_accuracy(df: pd.DataFrame, threshold: float = 0.5) -> float:
    """Area-weighted accuracy with invalid answers counted wrong (the headline number)."""
    if len(df) == 0:
        return float("nan")
    has_truth, _, _ = _truth_and_pred(df, threshold)
    w = point_weights(df) * has_truth
    total = w.sum()
    if total <= 0:
        return float("nan")
    return float((w * correct_array(df, threshold)).sum() / total)


def error_breakdown(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """Area-weighted shares of correct, false Land, false Water and invalid points (sum to 1)."""
    out = {"correct": float("nan"), "false_land": float("nan"), "false_water": float("nan"),
           "invalid": float("nan"), "n_false_land": 0, "n_false_water": 0, "n_invalid": 0}
    if len(df) == 0:
        return out
    has_truth, truth_land, pred = _truth_and_pred(df, threshold)
    w = point_weights(df) * has_truth
    total = w.sum()
    if total <= 0:
        return out
    invalid = has_truth & np.isnan(pred)
    false_land = has_truth & (pred == 1) & ~truth_land
    false_water = has_truth & (pred == 0) & truth_land
    correct = correct_array(df, threshold)
    out.update(
        correct=float((w * correct).sum() / total),
        false_land=float((w * false_land).sum() / total),
        false_water=float((w * false_water).sum() / total),
        invalid=float((w * invalid).sum() / total),
        n_false_land=int(false_land.sum()),
        n_false_water=int(false_water.sum()),
        n_invalid=int(invalid.sum()),
    )
    return out


def is_probabilistic(mode: str, df: pd.DataFrame) -> bool:
    """True when p_land carries more than a 0/1 answer (logprobs, or sampling with n > 1)."""
    if mode == ExtractionMode.LOGPROBS.value:
        return True
    if mode == ExtractionMode.SAMPLE.value and "n_samples" in df.columns and len(df):
        n = pd.to_numeric(df["n_samples"], errors="coerce").dropna()
        return bool(len(n) and n.max() > 1)  # same rule as scoring.score.is_probabilistic
    return False


# --------------------------------------------------------------------------- RunView


@dataclass
class RunView:
    """Everything a report needs for one run, loaded from the store."""

    run: RunRecord
    model: ModelSpec
    df: pd.DataFrame
    metrics: dict
    regions: dict
    acc_ci: tuple[float, float, float]

    # Convenience accessors (derived, not stored).
    @property
    def label(self) -> str:
        base = self.run.variant or self.model.id
        return f"{base} #{self.run.repeat_idx + 1}" if self.run.repeat_idx else base

    @property
    def model_key(self) -> str:
        return parse_variant(self.run.variant)[0] or self.model.id

    @property
    def config(self) -> dict[str, str]:
        return parse_variant(self.run.variant)[1]

    @property
    def effort(self) -> str:
        return self.config.get("effort", "default")

    @property
    def vendor(self) -> str:
        return self.model.vendor or self.model.provider or "unknown"

    @property
    def mode(self) -> str:
        m = self.run.extraction_mode
        return m.value if isinstance(m, ExtractionMode) else str(m)

    @property
    def forced_thinking(self) -> bool:
        return bool(self.run.forced_thinking or self.model.forced_thinking)

    @property
    def thinking(self) -> bool:
        return bool(self.run.thinking or self.forced_thinking)

    @property
    def acc(self) -> float:
        v = self.metrics.get("acc_area")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return self.acc_ci[0]
        return float(v)

    @property
    def probabilistic(self) -> bool:
        flag = self.metrics.get("probabilistic")  # set by scoring.cell_metrics
        if isinstance(flag, bool):
            return flag
        return is_probabilistic(self.mode, self.df)

    @property
    def status(self) -> str:
        s = self.run.status
        return s.value if hasattr(s, "value") else str(s)


# --------------------------------------------------------------------------- loading


def _normalize_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in POINT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if df["weight"].isna().any():
        df["weight"] = np.cos(np.radians(df["lat"].astype(float)))
    return df.sort_values("idx").reset_index(drop=True)


def _fill_truth(df: pd.DataFrame, mask: "Mask", spec: EvalSpec, *, force: bool = False) -> pd.DataFrame:
    """Fill missing truth from the mask; recompute all of it when `force` (mask changed),
    matching what `scoring.score.score_run` does so maps and metrics agree."""
    truth = pd.to_numeric(df["truth"], errors="coerce")
    if len(df) == 0 or (not force and not truth.isna().any()):
        return df
    from blindearth.evalspec.truth import cell_truth

    points = [Point(int(r.idx), float(r.lat), float(r.lon), float(r.weight))
              for r in df[["idx", "lat", "lon", "weight"]].itertuples(index=False)]
    computed = np.asarray(cell_truth(mask, points, spec.grid, spec.mask.truth_rule), dtype=float)
    computed[computed < 0] = np.nan
    computed = pd.Series(computed, index=df.index)
    df = df.copy()
    df["truth"] = computed if force else truth.fillna(computed)
    return df


def _resolve_model(store: "Store", run: RunRecord, registry: "Registry | None") -> ModelSpec:
    """ModelSpec for a run. The store contract has no model getter, so try one if the store
    offers it, then the registry, then build a minimal spec from the run record."""
    model: ModelSpec | None = None
    getter = getattr(store, "get_model", None)
    if callable(getter):
        try:
            got = getter(run.model_id)
            if isinstance(got, ModelSpec):
                model = got
            elif isinstance(got, (tuple, list)):
                model = next((g for g in got if isinstance(g, ModelSpec)), None)
        except Exception:  # noqa: BLE001 - optional API, fall through
            model = None

    reg_model: ModelSpec | None = None
    if registry is not None:
        for ref in (run.model_id, parse_variant(run.variant)[0]):
            try:
                reg_model = registry.model(ref)
                break
            except Exception:  # noqa: BLE001
                continue

    if model is None:
        model = reg_model
    elif reg_model is not None:
        # Fill report-only fields the store may not keep.
        model = dataclasses.replace(
            model,
            vendor=model.vendor or reg_model.vendor,
            release_date=model.release_date or reg_model.release_date,
            forced_thinking=model.forced_thinking or reg_model.forced_thinking,
        )

    if model is None:
        provider, _, mid = run.model_id.rpartition("/")
        model = ModelSpec(
            id=parse_variant(run.variant)[0] or mid or run.model_id,
            provider=provider or "unknown",
            name=run.resolved_model_version or mid or run.model_id,
            forced_thinking=run.forced_thinking,
        )
    return model


def _metrics_fresh(metrics: dict | None, df: pd.DataFrame) -> bool:
    if not metrics:
        return False
    n = metrics.get("n_points")
    return n is None or int(n) == len(df)


def _ci_from_metrics(metrics: dict) -> tuple[float, float, float] | None:
    ci = metrics.get("acc_area_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 3 and all(v is not None for v in ci):
        return (float(ci[0]), float(ci[1]), float(ci[2]))
    return None


def load_comparison(
    store: "Store",
    comparison: str,
    threshold: float = 0.5,
    *,
    registry: "Registry | None" = None,
    n_boot: int = 1000,
) -> tuple[dict, list[RunView], "Mask"]:
    """Load a comparison's runs, enforce the fairness rules and attach metrics.

    Returns (info, views, mask). `info` is the stored comparison dict plus: eval_spec, spec_id,
    mask_hash, mask_source, modes, mixed_modes, threshold, warnings.
    Raises ComparisonError if the runs do not share one eval spec.
    """
    comp = store.get_comparison(comparison)
    run_ids = list(comp.get("run_ids") or [])
    if not run_ids:
        raise ComparisonError(f"comparison {comparison!r} has no runs")
    runs = [store.get_run(rid) for rid in run_ids]

    # Fairness rule 1: one eval spec (mask, grid, prompt, coord format) per comparison.
    spec_ids = sorted({r.spec_id for r in runs})
    if len(spec_ids) != 1:
        detail = ", ".join(f"{r.variant}->{r.spec_id[:10]}" for r in runs)
        raise ComparisonError(
            f"comparison {comparison!r} mixes {len(spec_ids)} eval specs; scores are only "
            f"comparable under one mask, grid, prompt and coordinate format ({detail})"
        )
    spec_id = spec_ids[0]
    spec, mask_hash, mask_source = store.get_eval_spec(spec_id)

    from blindearth.evalspec.masks import load_mask
    from blindearth.scoring.bootstrap import acc_area_stat, block_bootstrap_ci
    from blindearth.scoring.regions import region_accuracy
    from blindearth.scoring.score import score_run

    mask = load_mask(spec.mask)
    warnings: list[str] = []
    mask_changed = bool(mask_hash) and mask.hash != mask_hash
    if mask_changed:
        warnings.append(
            f"Mask on disk ({mask.hash[:12]}) differs from the mask the runs were planned with "
            f"({mask_hash[:12]}); truth and scores below use the mask on disk."
        )

    views: list[RunView] = []
    for run in runs:
        df = _fill_truth(_normalize_points(store.load_points(run.id)), mask, spec,
                         force=mask_changed)
        metrics = store.load_metrics(run.id, mask.hash, threshold)
        if not _metrics_fresh(metrics, df):
            metrics = score_run(store, run.id, mask=mask, threshold=threshold)
        metrics = dict(metrics or {})

        regions = metrics.get("regions")
        if not isinstance(regions, dict) or not regions:
            regions = region_accuracy(df, threshold) if len(df) else {}
            metrics["regions_method"] = getattr(regions, "method", None)

        ci = _ci_from_metrics(metrics)
        if ci is None:
            if len(df):
                ci = block_bootstrap_ci(
                    df, acc_area_stat(threshold),
                    block_deg=10.0, n_boot=n_boot, seed=run.seed or 0,
                )
                ci = (float(ci[0]), float(ci[1]), float(ci[2]))
            else:
                ci = (float("nan"),) * 3
        if metrics.get("acc_area") is None:
            metrics["acc_area"] = area_accuracy(df, threshold)

        views.append(RunView(run=run, model=_resolve_model(store, run, registry), df=df,
                             metrics=metrics, regions=dict(regions), acc_ci=ci))

    # Fairness rule 2: mixed extraction modes are flagged; ranking uses thresholded accuracy.
    modes = sorted({v.mode for v in views})
    mixed = len(modes) > 1
    if mixed:
        warnings.append(
            "Runs use different extraction modes (" + ", ".join(modes) + "); the leaderboard "
            "ranks on thresholded accuracy only and probability metrics are not compared."
        )

    info: dict[str, Any] = dict(comp)
    info.update(
        eval_spec=spec,
        spec_id=spec_id,
        mask_hash=mask_hash or mask.hash,
        mask_hash_loaded=mask.hash,
        mask_source=mask_source or mask.source,
        modes=modes,
        mixed_modes=mixed,
        threshold=threshold,
        warnings=warnings,
    )
    return info, views, mask


__all__ = [
    "ComparisonError", "RunView", "load_comparison", "parse_variant", "area_accuracy",
    "correct_array", "error_breakdown", "point_weights", "is_probabilistic", "effort_sort_key",
    "PROB_METRICS", "EFFORT_ORDER",
]
