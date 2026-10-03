from __future__ import annotations

import numpy as np
import pandas as pd

from blindearth.scoring.render import (
    ERR_CORRECT,
    ERR_FALSE_LAND,
    ERR_FALSE_WATER,
    ERR_INVALID,
    OVERLAY_COLORS,
    error_map_rgb,
    error_overlay,
    grid_shape,
    points_to_grid,
    render_to_shape,
    upsample_to,
)


def test_grid_shape():
    assert grid_shape(2.0, "cell_center") == (90, 180)
    assert grid_shape(2.0, "cell_corner") == (90, 180)
    assert grid_shape(5.0, "cell_center") == (36, 72)


def test_orientation_cell_center():
    df = pd.DataFrame({"lat": [89.0, -89.0], "lon": [-179.0, 179.0], "p_land": [0.8, 0.2]})
    g = points_to_grid(df, 2.0, "cell_center", "p_land")
    assert g.shape == (90, 180)
    assert g[0, 0] == 0.8
    assert g[89, 179] == 0.2
    assert np.isnan(g[1, 1])
    b = points_to_grid(df, 2.0, "cell_center", "binary")
    assert b[0, 0] == 1.0 and b[89, 179] == 0.0


def test_orientation_cell_corner():
    df = pd.DataFrame({"lat": [90.0, -88.0], "lon": [-180.0, 178.0], "p_land": [1.0, 0.0]})
    g = points_to_grid(df, 2.0, "cell_corner", "p_land")
    assert g[0, 0] == 1.0 and g[89, 179] == 0.0


def test_error_codes():
    df = pd.DataFrame(
        {"lat": [89.0, 89.0, 89.0, 89.0], "lon": [-179.0, -177.0, -175.0, -173.0],
         "p_land": [0.9, 0.9, 0.1, np.nan], "truth": [1, 0, 1, 1]}
    )
    e = points_to_grid(df, 2.0, "cell_center", "error")
    assert list(e[0, :4]) == [ERR_CORRECT, ERR_FALSE_LAND, ERR_FALSE_WATER, ERR_INVALID]
    rgb = error_map_rgb(e)
    assert rgb.shape == (90, 180, 3) and rgb.dtype == np.uint8


def test_nearest_upsample_is_blocky():
    g = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
    up = upsample_to(g, (4, 8), binary=True)
    assert up.shape == (4, 8) and up.dtype == bool
    assert up[:2, :2].all() and not up[:2, 2:].any()
    assert up[2:, 6:].all()


def test_bilinear_upsample():
    g = np.full((3, 6), 0.4)
    up = upsample_to(g, (30, 60), binary=False)
    assert up.dtype == np.float32
    assert np.allclose(up, 0.4, atol=1e-6)
    g2 = np.zeros((4, 8))
    g2[:, 0] = 1.0
    up2 = upsample_to(g2, (40, 80), binary=False)
    assert up2.min() >= 0 and up2.max() <= 1
    # longitude wraps: the far east edge blends with column 0
    assert up2[20, -1] > 0


def test_corner_render_shift():
    g = np.zeros((2, 4), dtype=bool)
    g[0, 0] = True  # point at (90N, 180W)
    up = render_to_shape(g, (4, 8), 90.0, "cell_corner", binary=True)
    # nearest-neighbour region of the corner point straddles the seam
    assert up[0, 0] and up[0, -1]


def test_error_overlay_colors():
    pred = np.array([[True, True], [False, False]])
    mask = np.array([[True, False], [True, False]])
    o = error_overlay(pred, mask)
    assert tuple(o[0, 0]) == OVERLAY_COLORS["land"]
    assert tuple(o[0, 1]) == OVERLAY_COLORS["false_land"]
    assert tuple(o[1, 0]) == OVERLAY_COLORS["false_water"]
    assert tuple(o[1, 1]) == OVERLAY_COLORS["water"]
