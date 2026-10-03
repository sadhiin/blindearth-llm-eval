"""Plotly figures for the HTML report.

Every figure is built against a light/dark palette pair. Each trace carries `meta={"slot": k}`
(k = categorical slot, or "ink"/"muted" for reference marks) so the report's theme script can
restyle colours in place when the viewer switches theme; text never uses series colours.

Colour follows the entity: `entity_colours` assigns one fixed slot per model key, in the order
models first appear, and every chart uses that mapping. Past eight entities the remainder share
a neutral grey and are identified by legend and hover only (no generated hues).
"""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from blindearth.report.data import effort_sort_key, error_breakdown, point_weights

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

    from blindearth.report.data import RunView

# Reference categorical palette (light, dark), fixed slot order.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OVERFLOW = "#898781"
INK = {"light": "#0b0b0b", "dark": "#ffffff"}
MUTED = {"light": "#898781", "dark": "#898781"}
GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}
AXIS = {"light": "#c3c2b7", "dark": "#383835"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Error categories: status-like colours matching the PNG maps (never used for series).
ERR_COLOURS = {"false_land": "#d7301f", "false_water": "#fd8d3c", "invalid": "#762a83"}


def entity_colours(views: Iterable["RunView"]) -> dict[str, int | None]:
    """model key -> categorical slot index (None = overflow grey)."""
    slots: dict[str, int | None] = OrderedDict()
    for v in views:
        if v.model_key not in slots:
            n = len(slots)
            slots[v.model_key] = n if n < len(SERIES_LIGHT) else None
    return slots


def slot_colour(slot: int | str | None, theme: str = "light") -> str:
    if slot == "ink":
        return INK[theme]
    if slot == "muted":
        return MUTED[theme]
    if slot is None:
        return OVERFLOW
    pal = SERIES_LIGHT if theme == "light" else SERIES_DARK
    return pal[int(slot)]


def _layout(fig: "go.Figure", *, xtitle: str, ytitle: str, height: int = 420) -> "go.Figure":
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=30, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK["light"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=xtitle, gridcolor=GRID["light"], linecolor=AXIS["light"],
                     zeroline=False, showline=True)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID["light"], linecolor=AXIS["light"],
                     zeroline=False, showline=True)
    return fig


def _ci_err(v: "RunView") -> tuple[float, float]:
    est, lo, hi = v.acc_ci
    acc = v.acc
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in (acc, lo, hi)):
        return 0.0, 0.0
    return max(0.0, (hi - acc) * 100), max(0.0, (acc - lo) * 100)


def _pct(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x * 100


# --------------------------------------------------------------------------- effort curve


def effort_curve(views: list["RunView"]) -> "go.Figure | None":
    """Accuracy against effort level, one line per model. None when nothing was swept."""
    import plotly.graph_objects as go

    slots = entity_colours(views)
    by_model: dict[str, dict[str, list["RunView"]]] = defaultdict(lambda: defaultdict(list))
    for v in views:
        by_model[v.model_key][v.effort].append(v)
    swept = {m: e for m, e in by_model.items() if len(e) >= 2}
    if not swept:
        return None
    levels = sorted({e for efforts in swept.values() for e in efforts}, key=effort_sort_key)
    fig = go.Figure()
    for model, efforts in swept.items():
        xs, ys, up, dn, hover = [], [], [], [], []
        for lvl in sorted(efforts, key=effort_sort_key):
            runs = efforts[lvl]
            accs = [r.acc for r in runs]
            acc = float(np.nanmean(accs))
            lo = float(np.nanmean([r.acc_ci[1] for r in runs]))
            hi = float(np.nanmean([r.acc_ci[2] for r in runs]))
            xs.append(lvl)
            ys.append(acc * 100)
            up.append(max(0.0, (hi - acc) * 100))
            dn.append(max(0.0, (acc - lo) * 100))
            hover.append(f"{model} @ {lvl}<br>{acc*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"
                         + (f"<br>{len(runs)} repeats" if len(runs) > 1 else ""))
        c = slot_colour(slots[model])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=model, meta={"slot": slots[model]},
            line=dict(color=c, width=2), marker=dict(color=c, size=9),
            error_y=dict(type="data", array=up, arrayminus=dn, color=c, thickness=1.2, width=4),
            hovertext=hover, hoverinfo="text",
        ))
    _layout(fig, xtitle="Effort", ytitle="Area-weighted accuracy (%)")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=levels)
    return fig


# --------------------------------------------------------------------------- cost vs accuracy


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated by any other (lower-or-equal cost and higher accuracy).

    `points` are (cost, accuracy). Returned in increasing cost order.
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    frontier: list[int] = []
    best = -math.inf
    for i in order:
        cost, acc = points[i]
        if cost is None or acc is None or math.isnan(cost) or math.isnan(acc):
            continue
        if acc > best:
            frontier.append(i)
            best = acc
    return frontier


def cost_vs_accuracy(views: list["RunView"]) -> "go.Figure | None":
    import plotly.graph_objects as go

    slots = entity_colours(views)
    priced = [v for v in views if v.run.cost_usd is not None and not math.isnan(v.acc)]
    if not priced:
        return None
    pts = [(float(v.run.cost_usd), float(v.acc)) for v in priced]
    front = pareto_frontier(pts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[pts[i][0] for i in front], y=[pts[i][1] * 100 for i in front],
        mode="lines", name="Pareto frontier", meta={"slot": "muted"},
        line=dict(color=MUTED["light"], width=2, dash="dot", shape="hv"), hoverinfo="skip",
    ))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(priced):
        groups[v.model_key].append(i)
    front_set = set(front)
    for model, idxs in groups.items():
        c = slot_colour(slots[model])
        err = [_ci_err(priced[i]) for i in idxs]
        fig.add_trace(go.Scatter(
            x=[pts[i][0] for i in idxs], y=[pts[i][1] * 100 for i in idxs],
            mode="markers", name=model, meta={"slot": slots[model]},
            marker=dict(color=c, size=[13 if i in front_set else 9 for i in idxs],
                        symbol=["star" if priced[i].thinking else "circle" for i in idxs],
                        line=dict(width=2, color="rgba(252,252,251,0.9)")),
            error_y=dict(type="data", array=[e[0] for e in err], arrayminus=[e[1] for e in err],
                         color=c, thickness=1, width=3),
            hovertext=[f"{priced[i].label}<br>${pts[i][0]:.2f} · {pts[i][1]*100:.1f}%"
                       + ("<br>on Pareto frontier" if i in front_set else "") for i in idxs],
            hoverinfo="text",
        ))
    _layout(fig, xtitle="Cost per run (USD)", ytitle="Area-weighted accuracy (%)")
    costs = [p[0] for p in pts if p[0] > 0]
    if costs and max(costs) / max(min(costs), 1e-9) > 20:
        fig.update_xaxes(type="log")
    return fig


# --------------------------------------------------------------------------- lineage


def lineage(views: list["RunView"]) -> "go.Figure | None":
    """Accuracy against release date, one line per vendor through each model's best run."""
    import plotly.graph_objects as go

    dated = [v for v in views if v.model.release_date and not math.isnan(v.acc)]
    if not dated:
        return None
    vendors = list(OrderedDict.fromkeys(v.vendor for v in dated))
    fig = go.Figure()
    for k, vendor in enumerate(vendors):
        slot = k if k < len(SERIES_LIGHT) else None
        c = slot_colour(slot)
        vv = [v for v in dated if v.vendor == vendor]
        best: dict[str, "RunView"] = {}
        for v in vv:
            if v.model_key not in best or v.acc > best[v.model_key].acc:
                best[v.model_key] = v
        line_pts = sorted(best.values(), key=lambda v: v.model.release_date or "")
        fig.add_trace(go.Scatter(
            x=[v.model.release_date for v in line_pts], y=[v.acc * 100 for v in line_pts],
            mode="lines", name=vendor, meta={"slot": slot}, line=dict(color=c, width=2),
            hoverinfo="skip", legendgroup=vendor,
        ))
        fig.add_trace(go.Scatter(
            x=[v.model.release_date for v in vv], y=[v.acc * 100 for v in vv],
            mode="markers", name=vendor, meta={"slot": slot}, legendgroup=vendor,
            showlegend=False,
            marker=dict(color=c, size=10, symbol=["star" if v.thinking else "circle" for v in vv],
                        line=dict(width=2, color="rgba(252,252,251,0.9)")),
            hovertext=[f"{v.label}<br>released {v.model.release_date}<br>{v.acc*100:.1f}%"
                       for v in vv], hoverinfo="text",
        ))
    _layout(fig, xtitle="Release date", ytitle="Area-weighted accuracy (%)")
    fig.update_xaxes(type="date")
    return fig


# --------------------------------------------------------------------------- calibration


def reliability_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Area-weighted reliability table: per P(Land) bin, mean predicted vs observed land share."""
    p = pd.to_numeric(df["p_land"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["truth"], errors="coerce").to_numpy(float)
    w = point_weights(df)
    ok = ~np.isnan(p) & ~np.isnan(t)
    p, t, w = p[ok], t[ok], w[ok]
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        sel = b == k
        if not sel.any():
            continue
        ws = w[sel].sum() or 1.0
        rows.append({
            "bin_lo": edges[k], "bin_hi": edges[k + 1],
            "mean_p": float((w[sel] * p[sel]).sum() / ws),
            "observed": float((w[sel] * t[sel]).sum() / ws),
            "count": int(sel.sum()),
        })
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "mean_p", "observed", "count"])


def _bins_from_metrics(metrics: dict) -> pd.DataFrame | None:
    """`cell_metrics` calibration_bins [{lo, hi, mean_p, frac_land, weight, n}] -> our frame."""
    try:
        rows = [{"bin_lo": b["lo"], "bin_hi": b["hi"], "mean_p": b["mean_p"],
                 "observed": b["frac_land"], "count": int(b["n"])}
                for b in metrics["calibration_bins"]]
    except (KeyError, TypeError, ValueError):
        return None
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "mean_p", "observed", "count"])


def calibration(views: list["RunView"], n_bins: int = 10) -> "go.Figure | None":
    import plotly.graph_objects as go

    slots = entity_colours(views)
    prob = [v for v in views if v.probabilistic and len(v.df)]
    if not prob:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration",
                             meta={"slot": "muted"}, hoverinfo="skip",
                             line=dict(color=MUTED["light"], width=1.5, dash="dash")))
    for v in prob:
        rb = _bins_from_metrics(v.metrics) if v.metrics.get("calibration_bins") else None
        if rb is None or rb.empty:
            rb = reliability_bins(v.df, n_bins)
        if rb.empty:
            continue
        c = slot_colour(slots[v.model_key])
        ece = v.metrics.get("ece")
        name = v.label + (f" (ECE {ece:.3f})" if isinstance(ece, (int, float)) else "")
        sizes = np.clip(np.sqrt(rb["count"].to_numpy()) / 2.5, 8, 22)
        fig.add_trace(go.Scatter(
            x=rb["mean_p"] * 100, y=rb["observed"] * 100, mode="lines+markers", name=name,
            meta={"slot": slots[v.model_key]}, line=dict(color=c, width=2),
            marker=dict(color=c, size=sizes, line=dict(width=2, color="rgba(252,252,251,0.9)")),
            hovertext=[f"{v.label}<br>P(Land) {r.bin_lo:.1f}–{r.bin_hi:.1f}<br>"
                       f"predicted {r.mean_p*100:.1f}% · observed {r.observed*100:.1f}%<br>{r.count} cells"
                       for r in rb.itertuples()],
            hoverinfo="text",
        ))
    _layout(fig, xtitle="Mean predicted P(Land) (%)", ytitle="Observed land share (%)", height=460)
    fig.update_xaxes(range=[-2, 102])
    fig.update_yaxes(range=[-2, 102], scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------- errors / health


def error_split(views: list["RunView"], threshold: float = 0.5) -> "go.Figure":
    """Horizontal stacked bars: area share of false Land, false Water and invalid per run."""
    import plotly.graph_objects as go

    labels = [v.label for v in views]
    br = [error_breakdown(v.df, threshold) for v in views]
    fig = go.Figure()
    for key, name in (("false_land", "False Land"), ("false_water", "False Water"),
                      ("invalid", "Invalid")):
        fig.add_trace(go.Bar(
            y=labels, x=[_pct(b[key]) for b in br], name=name, orientation="h",
            meta={"fixed": ERR_COLOURS[key]},
            marker=dict(color=ERR_COLOURS[key], line=dict(width=1, color="rgba(252,252,251,0.9)")),
            hovertemplate="%{y}<br>" + name + ": %{x:.2f}% of area<extra></extra>",
        ))
    _layout(fig, xtitle="Share of area wrong (%)", ytitle="", height=max(260, 34 * len(views) + 110))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def latency_histogram(views: list["RunView"]) -> "go.Figure | None":
    import plotly.graph_objects as go

    slots = entity_colours(views)
    fig = go.Figure()
    any_data = False
    for v in views:
        lat = pd.to_numeric(v.df["latency_s"], errors="coerce").dropna()
        lat = lat[lat > 0]
        if lat.empty:
            continue
        any_data = True
        c = slot_colour(slots[v.model_key])
        fig.add_trace(go.Histogram(
            x=lat, name=v.label, meta={"slot": slots[v.model_key]}, opacity=0.55,
            marker=dict(color=c), nbinsx=60,
            hovertemplate=v.label + "<br>%{x} s: %{y} calls<extra></extra>",
        ))
    if not any_data:
        return None
    _layout(fig, xtitle="Latency per point (s)", ytitle="Points", height=380)
    fig.update_layout(barmode="overlay")
    lats = np.concatenate([pd.to_numeric(v.df["latency_s"], errors="coerce").dropna().to_numpy()
                           for v in views])
    lats = lats[lats > 0]
    if lats.size and np.percentile(lats, 99) / max(np.percentile(lats, 1), 1e-6) > 50:
        fig.update_xaxes(type="log")
    return fig


def fig_html(fig: "go.Figure | None", div_id: str) -> str:
    """Figure -> embeddable <div> (plotly.js is inlined once by the template)."""
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})


__all__ = [
    "effort_curve", "cost_vs_accuracy", "pareto_frontier", "lineage", "calibration",
    "reliability_bins", "error_split", "latency_histogram", "fig_html", "entity_colours",
    "slot_colour", "SERIES_LIGHT", "SERIES_DARK",
]
