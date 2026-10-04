"""Self-contained HTML report for one comparison.

One file, no network: plotly.js is inlined once, every map is a base64 PNG data URI, and all
interactivity (map toggles, leaderboard sort/filter, diff picker, theme) is vanilla JS.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import itertools
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from blindearth import __version__ as _report_version
from blindearth.report import charts, mapgrid
from blindearth.report.data import (
    PROB_METRICS,
    RunView,
    error_breakdown,
    load_comparison,
)
from blindearth.scoring.bootstrap import paired_difference

if TYPE_CHECKING:  # pragma: no cover
    from blindearth.evalspec.masks import Mask
    from blindearth.providers.registry import Registry
    from blindearth.store.db import Store

TEMPLATE_DIR = Path(__file__).parent / "templates"
MAX_DIFF_RUNS = 16  # pairs precomputed for at most this many runs: 16 -> 120 pairs

# Sequential blue ramp (reference palette) for the region heat table, light -> dark.
HEAT_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


# --------------------------------------------------------------------------- formatting


def _isnum(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not (
        isinstance(x, (float, np.floating)) and math.isnan(x))


def fmt_pct(x: Any, digits: int = 1) -> str:
    return f"{float(x) * 100:.{digits}f}%" if _isnum(x) else "—"


def fmt_num(x: Any, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}" if _isnum(x) else "—"


def fmt_usd(x: Any) -> str:
    if not _isnum(x):
        return "—"
    x = float(x)
    return f"${x:,.2f}" if x >= 1 else f"${x:.4f}"


def fmt_int(x: Any) -> str:
    return f"{int(x):,}" if _isnum(x) else "—"


def fmt_s(x: Any) -> str:
    return f"{float(x):.2f} s" if _isnum(x) else "—"


def _sortval(x: Any) -> str:
    """data-v attribute for numeric sort; blanks sort last."""
    return repr(float(x)) if _isnum(x) else ""


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# --------------------------------------------------------------------------- paired differences


class PairCache:
    """Memoized `paired_difference` per unordered pair; diff is A − B for the order asked."""

    def __init__(self, views: list[RunView], threshold: float, n_boot: int):
        self.views, self.threshold, self.n_boot = views, threshold, n_boot
        self._cache: dict[tuple[int, int], dict] = {}

    def get(self, i: int, j: int) -> dict:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in self._cache:
            try:
                res = dict(paired_difference(
                    self.views[a].df, self.views[b].df, threshold=self.threshold,
                    n_boot=self.n_boot, seed=0,
                ))
            except Exception as exc:  # noqa: BLE001 - shown in the report instead of failing it
                res = {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                       "significant": False, "n_paired": 0, "error": str(exc)}
            self._cache[(a, b)] = res
        res = dict(self._cache[(a, b)])
        for k in ("diff", "lo", "hi"):  # paired_difference returns None when nothing pairs
            if not _isnum(res.get(k)):
                res[k] = float("nan")
        if (i, j) != (a, b):
            res["diff"], res["lo"], res["hi"] = -res["diff"], -res["hi"], -res["lo"]
            res["acc_a"], res["acc_b"] = res.get("acc_b"), res.get("acc_a")
        return res


def rank_with_ties(views: list[RunView], pairs: PairCache) -> dict[int, dict]:
    """Rank by area-weighted accuracy (thresholded; never probability metrics).

    A run joins the current tie group when its paired-difference interval against the group's
    leader includes zero. Returns idx -> {rank, tied, vs_leader}.
    """
    def key(i: int) -> float:
        a = views[i].acc
        return -a if _isnum(a) else math.inf

    order = sorted(range(len(views)), key=key)
    out: dict[int, dict] = {}
    leader = None
    for pos, i in enumerate(order):
        if not _isnum(views[i].acc):
            out[i] = {"rank": pos + 1, "tied": False, "vs_leader": None}
            continue
        if leader is None:
            out[i] = {"rank": 1, "tied": False, "vs_leader": None}
            leader = i
            continue
        pdiff = pairs.get(leader, i)
        if not pdiff.get("significant", False) and not pdiff.get("error"):
            out[i] = {"rank": out[leader]["rank"], "tied": True, "vs_leader": pdiff,
                      "leader": leader}
            out[leader]["tied"] = True
        else:
            out[i] = {"rank": pos + 1, "tied": False, "vs_leader": pdiff}
            leader = i
    return out


# --------------------------------------------------------------------------- sections


def _header(info: dict, views: list[RunView], mask: "Mask") -> dict:
    spec = info["eval_spec"]
    starts = sorted(v.run.started_at for v in views if v.run.started_at)
    ends = sorted(v.run.ended_at for v in views if v.run.ended_at)
    costs = [v.run.cost_usd for v in views]
    known = [c for c in costs if _isnum(c)]
    versions = sorted({v.run.runner_version for v in views if v.run.runner_version})
    g = spec.grid
    return {
        "name": info.get("name") or info.get("id"),
        "comparison_id": info.get("id"),
        "created_at": info.get("created_at"),
        "spec_id": info.get("spec_id"),
        "mask_id": spec.mask.id,
        "mask_source": info.get("mask_source"),
        "mask_hash": info.get("mask_hash"),
        "mask_hash_loaded": info.get("mask_hash_loaded"),
        "mask_shape": "×".join(str(s) for s in np.asarray(mask.data).shape[::-1]),
        "truth_rule": spec.mask.truth_rule,
        "grid": f"{g.step_deg:g}°, {g.placement.replace('_', ' ')}",
        "subset": "all points" if not g.subset_frac else f"{g.subset_frac * 100:g}% (seed {g.seed})",
        "coord_format": spec.coord_format,
        "prompt_id": spec.prompt.id,
        "prompt": spec.prompt.template,
        "system_prompt": spec.prompt.system_prompt,
        "threshold": info.get("threshold", 0.5),
        "run_dates": (f"{starts[0][:16].replace('T', ' ')} → {ends[-1][:16].replace('T', ' ')} UTC"
                      if starts and ends else (starts[0] if starts else "—")),
        "runner_versions": ", ".join(versions) or "—",
        "total_spend": fmt_usd(sum(known)) if known else "—",
        "spend_note": (f"{len(costs) - len(known)} run(s) without a price" if len(known) < len(costs)
                       else ""),
        "n_runs": len(views),
        "modes": ", ".join(info.get("modes", [])),
        "mixed_modes": info.get("mixed_modes", False),
        "warnings": info.get("warnings", []),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "report_version": _report_version,
    }


def _map_images(views: list[RunView], spec, mask: "Mask", threshold: float, name: str) -> dict:
    step, placement = spec.grid.step_deg, spec.grid.placement
    out: dict[str, dict[str, str]] = {}
    for mode in ("binary", "probability", "error"):
        out[mode] = {}
        for coast in (False, True):
            png = mapgrid.render_map_grid(
                views, mode=mode, step_deg=step, placement=placement, mask=mask,
                coastline=coast, title=None, ncols=4, threshold=threshold)
            out[mode]["coast" if coast else "plain"] = data_uri(png)
    return out


def _leaderboard(views: list[RunView], info: dict, ranks: dict[int, dict]) -> list[dict]:
    rows = []
    mixed = info.get("mixed_modes", False)
    for i, v in enumerate(views):
        m = v.metrics
        r = ranks[i]
        rank_label = (f"={r['rank']}" if r["tied"] else str(r["rank"]))
        tie_note = ""
        vs = r.get("vs_leader")
        if r["tied"] and vs is not None and vs.get("error"):
            tie_note = f"no paired comparison with {views[r['leader']].label}: {vs['error']}"
        elif r["tied"] and vs is not None and _isnum(vs.get("diff")):
            lead = views[r["leader"]].label
            tie_note = (f"vs {lead}: Δ {vs['diff']*100:+.2f} pp "
                        f"[{vs['lo']*100:+.2f}, {vs['hi']*100:+.2f}], interval includes 0")
        elif r["tied"]:
            tie_note = "tied with a lower-ranked run: paired-difference interval includes 0"
        prob_ok = v.probabilistic and not mixed
        rows.append({
            "i": i, "rank": r["rank"], "rank_label": rank_label, "tied": r["tied"],
            "tie_note": tie_note,
            "label": v.label, "model": v.model.name, "resolved": v.run.resolved_model_version,
            "vendor": v.vendor, "mode": v.mode, "thinking": v.thinking,
            "forced": v.forced_thinking, "effort": v.effort, "status": v.status,
            "mixed_flag": mixed,
            "acc": v.acc, "lo": v.acc_ci[1], "hi": v.acc_ci[2],
            "acc_valid": m.get("acc_valid_only"), "skill": m.get("skill"), "f1": m.get("f1"),
            "invalid": m.get("invalid_rate"),
            "brier": m.get("brier") if prob_ok else None,
            "ece": m.get("ece") if prob_ok else None,
            "cost": v.run.cost_usd,
            "p50": m.get("latency_p50"), "p95": m.get("latency_p95"),
            "n_done": len(v.df), "n_total": v.run.n_points_total or len(v.df),
        })
    rows.sort(key=lambda r: (r["rank"], -(r["acc"] if _isnum(r["acc"]) else -1)))
    return rows


def _error_cards(views: list[RunView], spec, threshold: float) -> list[dict]:
    cards = []
    for v in views:
        b = error_breakdown(v.df, threshold)
        rgb = mapgrid.grid_rgb(v.df, spec.grid.step_deg, spec.grid.placement, "error", threshold)
        cards.append({"label": v.label, "forced": v.forced_thinking,
                      "img": data_uri(mapgrid.rgb_png(rgb, width=540)),
                      "false_land": b["false_land"], "false_water": b["false_water"],
                      "invalid": b["invalid"], "n_false_land": b["n_false_land"],
                      "n_false_water": b["n_false_water"], "n_invalid": b["n_invalid"]})
    return cards


def _diff_data(views: list[RunView], spec, threshold: float, pairs: PairCache,
               max_runs: int) -> tuple[dict, str]:
    order = sorted(range(len(views)),
                   key=lambda i: -(views[i].acc if _isnum(views[i].acc) else -1))
    chosen = sorted(order[:max_runs])
    note = ("" if len(views) <= max_runs else
            f"Diff pairs precomputed for the top {max_runs} runs by accuracy "
            f"({len(views) - max_runs} run(s) left out to keep the file small).")
    step, placement = spec.grid.step_deg, spec.grid.placement
    runs = []
    for i in chosen:
        v = views[i]
        rgb = mapgrid.grid_rgb(v.df, step, placement, "binary", threshold)
        runs.append({"i": i, "label": v.label, "acc": _f(v.acc) or 0.0,
                     "img": data_uri(mapgrid.rgb_png(rgb, width=540))})
    pair_data = {}
    for a, b in itertools.combinations(chosen, 2):
        rgb, stats = mapgrid.disagreement_rgb(views[a].df, views[b].df, step, placement, threshold)
        pdiff = pairs.get(a, b)
        pair_data[f"{a}-{b}"] = {
            "img": data_uri(mapgrid.rgb_png(rgb, width=720)),
            "diff": _f(pdiff.get("diff")), "lo": _f(pdiff.get("lo")), "hi": _f(pdiff.get("hi")),
            "significant": bool(pdiff.get("significant", False)),
            "n_paired": int(pdiff.get("n_paired") or 0), "error": pdiff.get("error"),
            **stats,
        }
    return {"runs": runs, "pairs": pair_data}, note


def _f(x: Any) -> float | None:
    return float(x) if _isnum(x) else None


def _region_order(name: str) -> tuple[int, str]:
    n = name.lower()
    if "antarc" in n:
        return (3, n)
    if "ocean" in n or "sea" in n:
        return (1, n)
    if "polar" in n or "arctic" in n:
        return (2, n)
    return (0, n)


def _heat(x: Any, lo: float, hi: float) -> tuple[str, str]:
    if not _isnum(x):
        return ("transparent", "inherit")
    t = 0.0 if hi <= lo else (float(x) - lo) / (hi - lo)
    k = int(round(np.clip(t, 0, 1) * (len(HEAT_RAMP) - 1)))
    return HEAT_RAMP[k], ("#ffffff" if k >= 7 else "#0b0b0b")


def _region_table(views: list[RunView]) -> dict:
    cols: list[str] = []
    for v in views:
        for k in v.regions:
            if k not in cols:
                cols.append(k)
    cols.sort(key=_region_order)
    vals = [float(x) for v in views for x in v.regions.values() if _isnum(x)]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    rows = []
    for v in sorted(views, key=lambda v: -(v.acc if _isnum(v.acc) else -1)):
        cells = []
        for c in cols:
            x = v.regions.get(c)
            bg, fg = _heat(x, lo, hi)
            cells.append({"text": fmt_pct(x), "bg": bg, "fg": fg, "v": _sortval(x)})
        rows.append({"label": v.label, "acc": v.acc, "forced": v.forced_thinking, "cells": cells})
    methods = sorted({str(m) for v in views if (m := (v.metrics or {}).get("regions_method"))})
    return {"cols": cols, "rows": rows, "lo": lo, "hi": hi, "methods": methods,
            "legend": [HEAT_RAMP[0], HEAT_RAMP[len(HEAT_RAMP) // 2], HEAT_RAMP[-1]]}


def _health(views: list[RunView]) -> list[dict]:
    rows = []
    for v in views:
        df = v.df
        err = df["error"].notna() & (df["error"].astype(str) != "")
        fr = df["finish_reason"].astype(str).str.lower()
        truncated = fr.isin(["length", "max_tokens", "max_output_tokens"])
        vm = pd.to_numeric(df["validity_mass"], errors="coerce")
        lat = pd.to_numeric(df["latency_s"], errors="coerce").dropna()
        invalid = pd.to_numeric(df["p_land"], errors="coerce").isna() & ~err
        rows.append({
            "label": v.label, "status": v.status,
            "done": len(df), "total": v.run.n_points_total or len(df),
            "invalid_rate": v.metrics.get("invalid_rate",
                                          float(invalid.mean()) if len(df) else None),
            "n_invalid": int(invalid.sum()), "n_errors": int(err.sum()),
            "n_truncated": int(truncated.sum()),
            "low_mass": float((vm < 0.5).mean()) if vm.notna().any() else None,
            "p50": float(lat.median()) if len(lat) else None,
            "p95": float(lat.quantile(0.95)) if len(lat) else None,
            "in_tok": v.run.totals.input_tokens, "out_tok": v.run.totals.output_tokens,
            "think_tok": v.run.totals.thinking_tokens, "cost": v.run.cost_usd,
            "started": (v.run.started_at or "")[:16].replace("T", " "),
            "ended": (v.run.ended_at or "")[:16].replace("T", " "),
            "resolved": v.run.resolved_model_version or "—",
            "runner": v.run.runner_version or "—",
            "top_error": (df.loc[err, "error"].astype(str).str[:80].value_counts().index[0]
                          if err.any() else ""),
        })
    return rows


def _caveats(spec, info: dict, views: list[RunView]) -> list[str]:
    km = spec.grid.step_deg * 111.2
    out = [
        f"At {spec.grid.step_deg:g}°, a cell is about {km:,.0f} km wide at the equator, so coastal "
        f"cells are ambiguous and the truth rule ({spec.mask.truth_rule.replace('_', ' ')}) matters.",
        "Coordinate tables (GeoNames, OpenStreetMap, Natural Earth) are probably in the training "
        "data, so scores measure what survived compression, not spatial reasoning.",
        "Thinking and non-thinking runs are not like for like, and local quantized models are not "
        "the full-precision model.",
        "Model aliases drift; the resolved version is stored with every run and shown under Run "
        "health.",
        f"Scores are only comparable with reports that share this mask ({info.get('mask_source')}, "
        f"hash {str(info.get('mask_hash'))[:12]}) and truth rule.",
        "Invalid answers (refusals, explanations, truncation) count as wrong in the headline "
        "accuracy; accuracy on valid answers only is shown beside it.",
        "Intervals are 95% block bootstraps over 10° × 10° blocks; two runs are called different "
        "only when the paired-difference interval excludes zero.",
    ]
    if info.get("mixed_modes"):
        out.append("This comparison mixes extraction modes; ranking uses thresholded accuracy "
                   "only and probability metrics are not compared across modes.")
    if any(v.status != "complete" for v in views):
        out.append("Some runs are partial; their scores cover only the points done so far.")
    return out


# --------------------------------------------------------------------------- render


def _env():
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      autoescape=select_autoescape(["html", "j2"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(pct=fmt_pct, num=fmt_num, usd=fmt_usd, int=fmt_int, secs=fmt_s,
                       sv=_sortval)
    return env


def render_report(info: dict, views: list[RunView], mask: "Mask", *, threshold: float = 0.5,
                  max_diff_runs: int = MAX_DIFF_RUNS, n_boot: int = 1000) -> str:
    """Build the report HTML string from already-loaded views (no store access)."""
    from plotly.offline import get_plotlyjs

    spec = info["eval_spec"]
    pairs = PairCache(views, threshold, n_boot)
    ranks = rank_with_ties(views, pairs)
    diff, diff_note = _diff_data(views, spec, threshold, pairs, max_diff_runs)

    figs = {
        "effort": charts.fig_html(charts.effort_curve(views), "fig-effort"),
        "cost": charts.fig_html(charts.cost_vs_accuracy(views), "fig-cost"),
        "lineage": charts.fig_html(charts.lineage(views), "fig-lineage"),
        "calibration": charts.fig_html(charts.calibration(views), "fig-calibration"),
        "errors": charts.fig_html(charts.error_split(views, threshold), "fig-errors")
        if views else "",
        "latency": charts.fig_html(charts.latency_histogram(views), "fig-latency"),
    }
    vendors = sorted({v.vendor for v in views})
    modes = sorted({v.mode for v in views})
    ctx = {
        "h": _header(info, views, mask),
        "maps": _map_images(views, spec, mask, threshold, info.get("name", "")),
        "leaderboard": _leaderboard(views, info, ranks),
        "vendors": vendors,
        "modes": modes,
        "any_forced": any(v.forced_thinking for v in views),
        "mixed": info.get("mixed_modes", False),
        "prob_metrics": PROB_METRICS,
        "error_cards": _error_cards(views, spec, threshold),
        "diff_json": json.dumps(diff, separators=(",", ":")).replace("</", "<\\/"),
        "diff_note": diff_note,
        "diff_legend": mapgrid.DIFF_LEGEND,
        "error_legend": mapgrid.ERROR_LEGEND,
        "regions": _region_table(views),
        "health": _health(views),
        "caveats": _caveats(spec, info, views),
        "figs": figs,
        "palette": {"light": charts.SERIES_LIGHT, "dark": charts.SERIES_DARK},
        "plotly_js": get_plotlyjs(),
        "n_prob": sum(1 for v in views if v.probabilistic),
        "rgb": lambda c: "rgb({},{},{})".format(*c),
        "esc": _html.escape,
    }
    return _env().get_template("report.html.j2").render(**ctx)


def build_report(store: "Store", comparison: str, out_html: Path, *, threshold: float = 0.5,
                 registry: "Registry | None" = None, max_diff_runs: int = MAX_DIFF_RUNS,
                 n_boot: int = 1000) -> Path:
    """Load a saved comparison and write its self-contained HTML report. Returns the path."""
    info, views, mask = load_comparison(store, comparison, threshold, registry=registry,
                                        n_boot=n_boot)
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(info, views, mask, threshold=threshold,
                                 max_diff_runs=max_diff_runs, n_boot=n_boot), encoding="utf-8")
    return out


__all__ = ["build_report", "render_report", "rank_with_ties", "PairCache", "data_uri"]
