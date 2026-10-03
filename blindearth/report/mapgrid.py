"""Map rendering: small-multiple map grids (PNG) and single-map RGB helpers.

The grid follows "How 12 blind Claudes see the Earth": black water, white land, one panel per
run titled `name — 97.8%`, with a star for forced-thinking models. Uses matplotlib's object API
with the Agg canvas only, so it never touches a GUI backend.
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pandas as pd

from blindearth.types import Placement

if TYPE_CHECKING:  # pragma: no cover
    from blindearth.evalspec.masks import Mask
    from blindearth.report.data import RunView

MapMode = Literal["binary", "probability", "error"]

STAR = "★"
DASH = "—"

# Categorical colours (RGB uint8).
WATER = (0, 0, 0)
LAND = (255, 255, 255)
INVALID = (128, 128, 128)          # answered but unparsable: counted wrong
NOT_DONE = (38, 38, 46)            # not evaluated yet (partial run)
CORRECT_LAND = (120, 200, 120)     # green, lighter on land so the continents stay readable
CORRECT_WATER = (28, 110, 60)      # green, darker on water
FALSE_LAND = (215, 48, 39)         # red: said Land, truth is water
FALSE_WATER = (253, 141, 60)       # orange: said Water, truth is land
INVALID_ERR = (118, 42, 131)       # purple: invalid in error mode
A_ONLY = (44, 123, 182)            # diff: only A correct
B_ONLY = (253, 174, 97)            # diff: only B correct
NEITHER = (171, 61, 168)           # diff: disagree and neither correct
AGREE_LAND = (200, 200, 200)
AGREE_WATER = (24, 24, 24)

ERROR_LEGEND = [
    ("Correct (land)", CORRECT_LAND),
    ("Correct (water)", CORRECT_WATER),
    ("False Land", FALSE_LAND),
    ("False Water", FALSE_WATER),
    ("Invalid", INVALID_ERR),
    ("Not evaluated", NOT_DONE),
]
BINARY_LEGEND = [("Land", LAND), ("Water", WATER), ("Invalid", INVALID), ("Not evaluated", NOT_DONE)]
DIFF_LEGEND = [
    ("Agree: land", AGREE_LAND),
    ("Agree: water", AGREE_WATER),
    ("Only A correct", A_ONLY),
    ("Only B correct", B_ONLY),
    ("Disagree, neither correct", NEITHER),
]


# --------------------------------------------------------------------------- grid geometry


def grid_shape(step_deg: float) -> tuple[int, int]:
    from blindearth.scoring.render import grid_shape as _gs

    return _gs(step_deg)


def _origin(step_deg: float, placement: Placement) -> tuple[float, float]:
    if placement == "cell_corner":
        return 90.0, -180.0
    return 90.0 - step_deg / 2.0, -180.0 + step_deg / 2.0


def cell_index(lat: np.ndarray, lon: np.ndarray, step_deg: float,
               placement: Placement) -> tuple[np.ndarray, np.ndarray]:
    from blindearth.scoring.render import grid_index

    return grid_index(np.asarray(lat, float), np.asarray(lon, float), step_deg, placement)


def df_to_grid(df: pd.DataFrame, step_deg: float, placement: Placement,
               column: str = "p_land") -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one column of a points table. -> (values float (H, W) NaN-filled, present bool).

    Geometry comes from `scoring.render`; `present` separates invalid answers (present, NaN)
    from points not evaluated yet.
    """
    from blindearth.scoring.render import values_to_grid

    h, w = grid_shape(step_deg)
    if len(df) == 0:
        return np.full((h, w), np.nan), np.zeros((h, w), bool)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(float)
    values = values_to_grid(lat, lon, pd.to_numeric(df[column], errors="coerce").to_numpy(float),
                            step_deg, placement)
    present = values_to_grid(lat, lon, np.ones(len(df)), step_deg, placement) == 1
    return values, present


# --------------------------------------------------------------------------- RGB rendering


def _paint(shape: tuple[int, int], layers: Sequence[tuple[np.ndarray, tuple[int, int, int]]],
           base: tuple[int, int, int] = NOT_DONE) -> np.ndarray:
    rgb = np.empty(shape + (3,), np.uint8)
    rgb[...] = base
    for where, colour in layers:
        rgb[where] = colour
    return rgb


def probability_rgb(p: np.ndarray, present: np.ndarray | None = None,
                    cmap: str = "magma") -> np.ndarray:
    """P(Land) grid -> RGB (magma: dark water, bright land). NaN shows as invalid/not done."""
    from matplotlib import colormaps

    p = np.asarray(p, float)
    lut = (colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    idx = np.clip(np.nan_to_num(p, nan=0.0) * 255, 0, 255).astype(int)
    rgb = lut[idx]
    nan = np.isnan(p)
    if present is None:
        rgb[nan] = NOT_DONE
    else:
        rgb[nan & present] = INVALID
        rgb[~present] = NOT_DONE
    return rgb


def grid_rgb(df: pd.DataFrame, step_deg: float, placement: Placement, mode: MapMode,
             threshold: float = 0.5) -> np.ndarray:
    """One run's map as RGB uint8 (H, W, 3) at grid resolution."""
    p, present = df_to_grid(df, step_deg, placement, "p_land")
    if mode == "probability":
        return probability_rgb(p, present)
    valid = present & ~np.isnan(p)
    pred_land = valid & (np.nan_to_num(p, nan=0.0) > threshold)
    pred_water = valid & ~pred_land
    invalid = present & np.isnan(p)
    if mode == "binary":
        return _paint(p.shape, [(pred_water, WATER), (pred_land, LAND), (invalid, INVALID)])
    if mode != "error":
        raise ValueError(f"unknown map mode {mode!r}")
    # Error codes from scoring.render (ERR_CORRECT / FALSE_LAND / FALSE_WATER / INVALID).
    # The palette here is the report's: two greens (so continents stay readable), red for
    # false Land, orange for false Water, purple for invalid.
    from blindearth.scoring.render import (
        ERR_CORRECT, ERR_FALSE_LAND, ERR_FALSE_WATER, ERR_INVALID, points_to_grid,
    )

    err = points_to_grid(df, step_deg, placement, "error", threshold)
    return _paint(p.shape, [
        (present & np.isnan(err), INVALID),  # present but truth unknown
        ((err == ERR_CORRECT) & pred_land, CORRECT_LAND),
        ((err == ERR_CORRECT) & ~pred_land, CORRECT_WATER),
        (err == ERR_FALSE_LAND, FALSE_LAND),
        (err == ERR_FALSE_WATER, FALSE_WATER),
        (err == ERR_INVALID, INVALID_ERR),
    ])


def disagreement_rgb(df_a: pd.DataFrame, df_b: pd.DataFrame, step_deg: float,
                     placement: Placement, threshold: float = 0.5) -> tuple[np.ndarray, dict]:
    """Where A and B differ, coloured by who is right. -> (RGB, area-weighted counts)."""
    pa, presa = df_to_grid(df_a, step_deg, placement, "p_land")
    pb, presb = df_to_grid(df_b, step_deg, placement, "p_land")
    ta, _ = df_to_grid(df_a, step_deg, placement, "truth")
    tb, _ = df_to_grid(df_b, step_deg, placement, "truth")
    truth = np.where(np.isnan(ta), tb, ta)
    both = presa & presb

    def answer(p: np.ndarray) -> np.ndarray:  # 1 land, 0 water, -1 invalid
        return np.where(np.isnan(p), -1, (np.nan_to_num(p, nan=0.0) > threshold).astype(int))

    aa, ab = answer(pa), answer(pb)
    ca = (aa >= 0) & (aa == truth)
    cb = (ab >= 0) & (ab == truth)
    agree = both & (aa == ab)
    differ = both & (aa != ab)
    rgb = _paint(pa.shape, [
        (agree & (aa == 1), AGREE_LAND),
        (agree & (aa != 1), AGREE_WATER),
        (differ & ca & ~cb, A_ONLY),
        (differ & cb & ~ca, B_ONLY),
        (differ & ~ca & ~cb, NEITHER),
    ])
    h, _ = pa.shape
    lat0, _ = _origin(step_deg, placement)
    lats = lat0 - step_deg * np.arange(h)
    w = np.repeat(np.cos(np.radians(lats))[:, None], pa.shape[1], axis=1)
    wb = (w * both).sum() or 1.0
    stats = {
        "n_both": int(both.sum()),
        "n_disagree": int(differ.sum()),
        "share_disagree_area": float((w * differ).sum() / wb),
        "n_only_a": int((differ & ca & ~cb).sum()),
        "n_only_b": int((differ & cb & ~ca).sum()),
    }
    return rgb, stats


def upscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)


def rgb_png(rgb: np.ndarray, *, width: int | None = 720) -> bytes:
    """Encode an RGB array as PNG, nearest-neighbour upscaled to about `width` pixels."""
    from PIL import Image

    if width and rgb.shape[1] < width:
        rgb = upscale(rgb, max(1, width // rgb.shape[1]))
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def live_rgb(grid: np.ndarray, *, binary: bool = False, threshold: float = 0.5,
             scale: int = 4) -> np.ndarray:
    """RGB for a live map from `scoring.render.points_to_grid` output (NaN = not yet done)."""
    g = np.asarray(grid, float)
    if binary:
        nan = np.isnan(g)
        rgb = _paint(g.shape, [(~nan & (np.nan_to_num(g) <= threshold), WATER),
                               (~nan & (np.nan_to_num(g) > threshold), LAND)])
    else:
        rgb = probability_rgb(g)
    return upscale(rgb, scale)


# --------------------------------------------------------------------------- map grid


def panel_title(view: "RunView", max_len: int = 40) -> str:
    name = view.label
    if len(name) > max_len:
        name = name[: max_len - 1] + "…"
    acc = view.acc
    acc_s = "n/a" if acc is None or (isinstance(acc, float) and math.isnan(acc)) else f"{acc * 100:.1f}%"
    star = f" {STAR}" if view.forced_thinking else ""
    return f"{name}{star} {DASH} {acc_s}"


def _downsample_mask(mask: "Mask", max_w: int = 1440) -> np.ndarray:
    data = np.asarray(mask.data, bool)
    f = max(1, int(math.ceil(data.shape[1] / max_w)))
    return data[::f, ::f].astype(float)


def render_map_grid(
    views: list["RunView"],
    *,
    mode: MapMode = "binary",
    step_deg: float,
    placement: Placement,
    mask: "Mask | None" = None,
    coastline: bool = False,
    title: str | None = None,
    ncols: int = 4,
    threshold: float = 0.5,
    panel_width_in: float = 4.0,
    dpi: int = 110,
) -> bytes:
    """Small multiples, one map per run. Returns PNG bytes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    n = max(1, len(views))
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    panel_h = panel_width_in / 2.0
    title_h = 0.38
    top_in = 0.65 if title else 0.12
    bottom_in = 0.6
    fig_w = ncols * panel_width_in + 0.2
    fig_h = top_in + nrows * (panel_h + title_h) + bottom_in
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    fig.subplots_adjust(
        left=0.1 / fig_w, right=1 - 0.1 / fig_w,
        top=1 - (top_in + title_h) / fig_h, bottom=bottom_in / fig_h,
        wspace=0.04, hspace=title_h / panel_h,
    )
    axes = fig.subplots(nrows, ncols, squeeze=False)

    coast = _downsample_mask(mask) if (coastline and mask is not None) else None
    coast_colour = {"binary": "#22c4e0", "probability": "#22c4e0", "error": "#1d1d1d"}[mode]

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= len(views):
            continue
        v = views[i]
        rgb = grid_rgb(v.df, step_deg, placement, mode, threshold)
        ax.imshow(rgb, extent=(-180, 180, -90, 90), interpolation="nearest", aspect="equal")
        if coast is not None:
            ax.contour(coast, levels=[0.5], colors=[coast_colour], linewidths=0.45,
                       extent=(-180, 180, -90, 90), origin="upper")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title(panel_title(v), fontsize=10, color="#111111", pad=4)

    if not views:
        axes.flat[0].text(0, 0, "No runs", ha="center", va="center", fontsize=14)
        axes.flat[0].set_xlim(-180, 180)
        axes.flat[0].set_ylim(-90, 90)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1 - 0.12 / fig_h, va="top")

    legend_y = 0.18 / fig_h
    if mode == "probability":
        cax = fig.add_axes((0.3, legend_y + 0.12 / fig_h, 0.4, 0.12 / fig_h))
        sm = ScalarMappable(norm=Normalize(0, 1), cmap="magma")
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("P(Land)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        items = ERROR_LEGEND if mode == "error" else BINARY_LEGEND
        handles = [Patch(facecolor=np.array(c) / 255, edgecolor="#777777", label=lab)
                   for lab, c in items]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, legend_y * 0.3))
    if any(v.forced_thinking for v in views):
        fig.text(1 - 0.12 / fig_w, 0.06 / fig_h, f"{STAR} forced thinking", ha="right",
                 va="bottom", fontsize=8, color="#444444")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()


__all__ = [
    "render_map_grid", "grid_rgb", "disagreement_rgb", "probability_rgb", "live_rgb", "rgb_png",
    "df_to_grid", "grid_shape", "cell_index", "panel_title", "upscale",
    "ERROR_LEGEND", "BINARY_LEGEND", "DIFF_LEGEND",
]
