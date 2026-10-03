"""Map rendering tests on synthetic points (no store, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blindearth.report import mapgrid
from blindearth.report.data import RunView
from blindearth.types import ExtractionMode, ModelSpec, RunRecord

STEP = 10.0


def make_df(step: float = STEP, placement: str = "cell_center", flip_every: int = 0,
            invalid_every: int = 0, seed: int = 0) -> pd.DataFrame:
    off = step / 2 if placement == "cell_center" else 0.0
    lats = np.arange(90 - off, -90, -step)[: int(180 / step)]
    lons = np.arange(-180 + off, 180, step)[: int(360 / step)]
    rows = []
    rng = np.random.default_rng(seed)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            idx = i * len(lons) + j
            truth = int((abs(lat) < 40 and -20 < lon < 40) or lat < -65)
            p = float(truth)
            if flip_every and idx % flip_every == 0:
                p = 1.0 - p
            if invalid_every and idx % invalid_every == 1:
                p = np.nan
            rows.append({"idx": idx, "lat": lat, "lon": lon, "weight": np.cos(np.radians(lat)),
                         "truth": truth, "p_land": p if np.isnan(p) else
                         float(np.clip(p * 0.9 + 0.05 + rng.normal(0, 0.01), 0, 1)),
                         "n_valid": 4, "n_samples": 4, "latency_s": 0.3, "error": None})
    return pd.DataFrame(rows)


def make_view(df: pd.DataFrame, variant: str = "m@effort=low", acc: float = 0.9,
              forced: bool = False) -> RunView:
    run = RunRecord(id="r-" + variant, run_hash="h", spec_id="s", model_id="p/m", config_id="c",
                    variant=variant, extraction_mode=ExtractionMode.SAMPLE, forced_thinking=forced)
    model = ModelSpec(id=variant.split("@")[0], provider="p", name="m", forced_thinking=forced)
    return RunView(run=run, model=model, df=df, metrics={"acc_area": acc}, regions={},
                   acc_ci=(acc, acc - 0.01, acc + 0.01))


def test_grid_shape_and_index():
    assert mapgrid.grid_shape(2.0) == (90, 180)
    assert mapgrid.grid_shape(10.0) == (18, 36)
    r, c = mapgrid.cell_index(np.array([89.0, -89.0]), np.array([-179.0, 179.0]), 2.0, "cell_center")
    assert list(r) == [0, 89] and list(c) == [0, 179]
    r, c = mapgrid.cell_index(np.array([90.0, -88.0]), np.array([-180.0, 178.0]), 2.0, "cell_corner")
    assert list(r) == [0, 89] and list(c) == [0, 179]


@pytest.mark.parametrize("placement", ["cell_center", "cell_corner"])
def test_df_to_grid_roundtrip(placement):
    df = make_df(placement=placement)
    vals, present = mapgrid.df_to_grid(df, STEP, placement, "truth")
    assert present.all()
    # idx is row-major north->south, west->east, so the flat grid equals truth in idx order
    np.testing.assert_array_equal(vals.ravel(), df.sort_values("idx")["truth"].to_numpy(float))


def test_binary_colours_and_missing():
    df = make_df(invalid_every=7)
    part = df.iloc[: len(df) // 2]
    rgb = mapgrid.grid_rgb(part, STEP, "cell_center", "binary")
    assert rgb.shape == (18, 36, 3) and rgb.dtype == np.uint8
    colours = {tuple(c) for c in rgb.reshape(-1, 3)}
    assert mapgrid.WATER in colours and mapgrid.LAND in colours
    assert mapgrid.INVALID in colours
    assert tuple(rgb[-1, -1]) == mapgrid.NOT_DONE  # southern half not evaluated


def test_error_mode_splits_false_land_and_water():
    df = make_df(flip_every=5)
    rgb = mapgrid.grid_rgb(df, STEP, "cell_center", "error")
    flat = [tuple(c) for c in rgb.reshape(-1, 3)]
    flipped = df[df["idx"] % 5 == 0]
    n_fl = int((flipped["truth"] == 0).sum())
    n_fw = int((flipped["truth"] == 1).sum())
    assert flat.count(mapgrid.FALSE_LAND) == n_fl
    assert flat.count(mapgrid.FALSE_WATER) == n_fw
    assert flat.count(mapgrid.CORRECT_LAND) + flat.count(mapgrid.CORRECT_WATER) == len(df) - n_fl - n_fw


def test_probability_rgb_extremes():
    p = np.array([[0.0, 1.0, np.nan]])
    rgb = mapgrid.probability_rgb(p)
    assert rgb[0, 0].sum() < rgb[0, 1].sum()
    assert tuple(rgb[0, 2]) == mapgrid.NOT_DONE


def test_disagreement_stats():
    a = make_df()
    b = make_df(flip_every=9)
    rgb, stats = mapgrid.disagreement_rgb(a, b, STEP, "cell_center")
    assert stats["n_disagree"] == int((a["idx"] % 9 == 0).sum())
    assert stats["n_only_a"] == stats["n_disagree"] and stats["n_only_b"] == 0
    assert 0 < stats["share_disagree_area"] < 1
    assert rgb.shape == (18, 36, 3)


def test_panel_title_and_star():
    v = make_view(make_df(), variant="fable-5", acc=0.978, forced=True)
    t = mapgrid.panel_title(v)
    assert t.startswith("fable-5 ★ — 97.8%")


@pytest.mark.parametrize("mode", ["binary", "probability", "error"])
@pytest.mark.parametrize("coast", [False, True])
def test_render_map_grid_png(mode, coast):
    views = [make_view(make_df(flip_every=k + 3, seed=k), variant=f"m{k}", acc=0.9 - k / 100,
                       forced=(k == 1)) for k in range(5)]
    land = np.zeros((180, 360), bool)
    land[50:130, 160:220] = True
    mask = SimpleNamespace(data=land, source="synthetic", hash="0" * 64)
    png = mapgrid.render_map_grid(views, mode=mode, step_deg=STEP, placement="cell_center",
                                  mask=mask, coastline=coast, title="How 5 blind models see the Earth",
                                  ncols=3)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 5000


def test_render_map_grid_empty():
    png = mapgrid.render_map_grid([], step_deg=STEP, placement="cell_center")
    assert png[:4] == b"\x89PNG"


def test_rgb_png_and_live():
    grid = np.full((18, 36), np.nan)
    grid[:5] = 0.9
    rgb = mapgrid.live_rgb(grid, scale=4)
    assert rgb.shape == (72, 144, 3)
    png = mapgrid.rgb_png(rgb, width=None)
    assert png[:4] == b"\x89PNG"
