from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from blindearth.scoring.image_metrics import downsample_mask, image_metrics, label_wrapped
from blindearth.scoring.render import upsample_to

STEP = 10.0


@dataclass
class FakeMask:  # duck-typed stand-in for evalspec.masks.Mask
    data: np.ndarray
    source: str = "synthetic"
    hash: str = "synthetic"


def truth_grid() -> np.ndarray:
    g = np.zeros((18, 36), dtype=float)
    g[3:8, 5:12] = 1  # "northern continent"
    g[10:15, 20:24] = 1  # "southern continent"
    g[12, 30] = 1  # small island
    return g


def grid_df(grid: np.ndarray, p: np.ndarray | None = None) -> pd.DataFrame:
    nr, nc = grid.shape
    lats = 90 - STEP / 2 - STEP * np.arange(nr)
    lons = -180 + STEP / 2 + STEP * np.arange(nc)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    vals = grid if p is None else p
    return pd.DataFrame(
        {"idx": np.arange(LA.size), "lat": LA.ravel(), "lon": LO.ravel(),
         "weight": np.cos(np.deg2rad(LA.ravel())), "truth": grid.ravel().astype(int),
         "p_land": vals.ravel().astype(float)}
    )


def block_mask(grid: np.ndarray, shape=(180, 360)) -> FakeMask:
    return FakeMask(upsample_to(grid > 0.5, shape, binary=True))


def test_perfect_at_grid_matches_ceiling():
    tg = truth_grid()
    m = image_metrics(grid_df(tg), block_mask(tg), STEP, "cell_center",
                      work_shape=(180, 360), truth_grid=tg)
    assert m["pixel_iou"] == pytest.approx(1.0)
    assert m["dice"] == pytest.approx(1.0)
    assert m["boundary_f@2"] == pytest.approx(1.0)
    assert m["contour_mean_deg"] == pytest.approx(0.0)
    assert m["spurious_specks"] == 0
    assert m["n_components_pred"] == m["n_components_true"] == 3
    assert m["share_of_ceiling"]["pixel_iou"] == pytest.approx(1.0)
    assert m["share_of_ceiling"]["ssim"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(1.0)
    assert m["work_shape"] == [180, 360]


def test_worse_prediction_scores_below_ceiling():
    tg = truth_grid()
    pred = np.roll(tg, 2, axis=1)
    pred[1, 30] = 1  # spurious speck over water
    m = image_metrics(grid_df(tg, pred), block_mask(tg), STEP, "cell_center",
                      work_shape=(180, 360), truth_grid=tg)
    assert m["pixel_iou"] < 1.0
    assert m["share_of_ceiling"]["pixel_iou"] < 1.0
    assert m["contour_mean_deg"] > 0
    assert m["contour_mean_km"] == pytest.approx(m["contour_mean_deg"] * 111.195, rel=1e-3)
    assert m["spurious_specks"] >= 1
    assert m["ceiling"]["pixel_iou"] == pytest.approx(1.0)


def test_seam_landmass_is_one_component():
    mask = np.zeros((180, 360), dtype=bool)
    mask[80:100, :5] = True
    mask[80:100, 355:] = True
    labels, n = label_wrapped(mask)
    assert n == 1
    zeros = np.zeros((18, 36))
    m = image_metrics(grid_df(zeros), FakeMask(mask), STEP, "cell_center",
                      work_shape=(180, 360), truth_grid=zeros)
    assert m["n_components_true"] == 1
    assert len(m["per_landmass_iou"]) == 1
    assert list(m["per_landmass_iou"].values())[0] == 0.0


def test_prob_rmse_only_when_probabilistic():
    tg = truth_grid()
    df = grid_df(tg, np.where(tg > 0, 0.8, 0.2))
    m = image_metrics(df, block_mask(tg), STEP, "cell_center", work_shape=(180, 360),
                      truth_grid=tg, probabilistic=True)
    assert m["prob_rmse"] is not None and 0 < m["prob_rmse"] < 0.5
    m2 = image_metrics(df, block_mask(tg), STEP, "cell_center", work_shape=(180, 360),
                       truth_grid=tg, probabilistic=False)
    assert m2["prob_rmse"] is None


def test_partial_run_coverage():
    tg = truth_grid()
    df = grid_df(tg).sample(frac=0.5, random_state=0)
    m = image_metrics(df, block_mask(tg), STEP, "cell_center", work_shape=(180, 360), truth_grid=tg)
    assert 0.4 < m["coverage"] < 0.6


def test_downsample_mask_majority_and_nearest():
    big = np.zeros((360, 720), dtype=bool)
    big[:180, :] = True
    small = downsample_mask(big, (180, 360))
    assert small.shape == (180, 360)
    assert small[:90].all() and not small[90:].any()
    odd = downsample_mask(big, (100, 200))
    assert odd.shape == (100, 200)
    unchanged = downsample_mask(big, (1800, 3600))
    assert unchanged.shape == big.shape
