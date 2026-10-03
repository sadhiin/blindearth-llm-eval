import numpy as np
import pytest

from blindearth.evalspec.grid import make_grid
from blindearth.evalspec.masks import Mask
from blindearth.evalspec.truth import cell_truth, majority_in_bounds, sample_points
from blindearth.types import GridSpec, Point


def _one_degree_mask(fn):
    """180 x 360 mask, 1 deg pixels; fn(lat_center, lon_center) -> bool."""
    lat = 90 - (np.arange(180) + 0.5)
    lon = -180 + (np.arange(360) + 0.5)
    data = fn(lat[:, None], lon[None, :])
    return Mask.from_array(np.broadcast_to(data, (180, 360)).copy(), "synthetic")


def test_cell_center_samples_point():
    mask = _one_degree_mask(lambda la, lo: (la > 0) & (la < 30) & (lo > 0) & (lo < 60))
    grid = GridSpec(step_deg=2)
    pts = make_grid(grid)
    truth = cell_truth(mask, pts, grid, "cell_center")
    assert truth.dtype == np.int8 and truth.shape == (len(pts),)
    for p, t in zip(pts, truth):
        assert t == int(0 < p.lat < 30 and 0 < p.lon < 60)


def test_majority_rule_on_half_cells():
    # Land west of lon 0.5 within 10..20 N: the 2-degree cell centred on lon 1 is 1 of 2 px land
    # (tie -> water); the cell centred on -1 is all land.
    mask = _one_degree_mask(lambda la, lo: (la > 10) & (la < 20) & (lo < 0.6))
    grid = GridSpec(step_deg=2)
    pts = [Point(0, 15.0, 1.0, 1.0), Point(1, 15.0, -1.0, 1.0), Point(2, 15.0, 3.0, 1.0)]
    truth = cell_truth(mask, pts, grid, "majority")
    assert truth.tolist() == [0, 1, 0]


def test_majority_three_quarters_is_land():
    data = np.zeros((180, 360), bool)
    data[0:2, 0:2] = True
    data[1, 1] = False  # 3 of 4 pixels in the top-left 2-degree cell
    mask = Mask.from_array(data, "t")
    grid = GridSpec(step_deg=2)
    pts = [Point(0, 89.0, -179.0, 0.0)]
    assert cell_truth(mask, pts, grid, "majority").tolist() == [1]


def test_seam_wraps_columns():
    data = np.zeros((180, 360), bool)
    data[:, 0] = True  # -180..-179
    data[:, 359] = True  # 179..180
    mask = Mask.from_array(data, "seam")
    # A box straddling the antimeridian: 179..181 -> cols 359 and 0 (wrapped), both land.
    assert majority_in_bounds(mask, 179.0, 0.0, 181.0, 2.0) == 1
    assert majority_in_bounds(mask, -181.0, 0.0, -179.0, 2.0) == 1
    # lon 180 samples the -180 column.
    assert sample_points(mask, np.array([10.0]), np.array([180.0])).tolist() == [1]


def test_poles_and_corner_placement():
    data = np.zeros((180, 360), bool)
    data[-1, :] = True  # southernmost row is land (Antarctica)
    mask = Mask.from_array(data, "pole")
    grid = GridSpec(step_deg=2, placement="cell_corner")
    pts = make_grid(grid)
    center = cell_truth(mask, pts, grid, "cell_center")
    assert center[0] == 0  # 90 N clamps to row 0
    assert sample_points(mask, np.array([-90.0]), np.array([0.0])).tolist() == [1]
    # Majority over the southernmost corner-placed cells (-88 .. -90): 1 of 2 rows land -> tie.
    maj = cell_truth(mask, pts, grid, "majority")
    assert maj.max() == 0


def test_tiny_cell_falls_back_to_center_pixel():
    data = np.zeros((18, 36), bool)  # 10 deg pixels
    data[8, 18] = True  # lat 0..10, lon 0..10
    mask = Mask.from_array(data, "coarse")
    # The 1..3 box contains no pixel centre (centres sit at 5, 15, ...): falls back to the pixel
    # holding the box's middle.
    assert majority_in_bounds(mask, 1.0, 1.0, 3.0, 3.0) == 1
    assert majority_in_bounds(mask, 11.0, 1.0, 13.0, 3.0) == 0


def test_unknown_rule():
    mask = Mask.from_array(np.zeros((18, 36), bool), "z")
    with pytest.raises(ValueError):
        cell_truth(mask, [Point(0, 0.0, 0.0, 1.0)], GridSpec(), "mode")  # type: ignore[arg-type]
