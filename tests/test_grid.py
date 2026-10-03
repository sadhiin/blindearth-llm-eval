import math

import pytest

from blindearth.evalspec.grid import cell_bounds, grid_dims, make_grid, stratified_order
from blindearth.types import GridSpec, Point


@pytest.mark.parametrize("step,n", [(1, 64_800), (2, 16_200), (4, 4_050), (5, 2_592), (3, 7_200)])
def test_point_counts(step, n):
    assert len(make_grid(GridSpec(step_deg=step))) == n


def test_center_placement_extremes_and_order():
    pts = make_grid(GridSpec(step_deg=2))
    assert (pts[0].lat, pts[0].lon) == (89.0, -179.0)
    assert (pts[1].lat, pts[1].lon) == (89.0, -177.0)
    assert (pts[180].lat, pts[180].lon) == (87.0, -179.0)
    assert (pts[-1].lat, pts[-1].lon) == (-89.0, 179.0)
    assert [p.idx for p in pts] == list(range(len(pts)))


def test_corner_placement_extremes():
    pts = make_grid(GridSpec(step_deg=2, placement="cell_corner"))
    assert (pts[0].lat, pts[0].lon) == (90.0, -180.0)
    assert (pts[-1].lat, pts[-1].lon) == (-88.0, 178.0)
    assert pts[0].weight == 0.0


def test_weights_are_cos_lat():
    for p in make_grid(GridSpec(step_deg=5))[::97]:
        assert p.weight == pytest.approx(math.cos(math.radians(p.lat)))


def test_bad_step_rejected():
    with pytest.raises(ValueError):
        grid_dims(7)
    with pytest.raises(ValueError):
        make_grid(GridSpec(step_deg=0))


def test_custom_fractional_step():
    assert grid_dims(0.5) == (360, 720)


def test_subset_seeded_and_stable():
    full = {p.idx: p for p in make_grid(GridSpec(step_deg=2))}
    a = make_grid(GridSpec(step_deg=2, subset_frac=0.1, seed=1))
    b = make_grid(GridSpec(step_deg=2, subset_frac=0.1, seed=1))
    c = make_grid(GridSpec(step_deg=2, subset_frac=0.1, seed=2))
    assert len(a) == 1620
    assert a == b
    assert a != c
    assert [p.idx for p in a] == sorted(p.idx for p in a)
    for p in a:
        assert full[p.idx] == p


def test_subset_bad_fraction():
    with pytest.raises(ValueError):
        make_grid(GridSpec(subset_frac=0.0))
    with pytest.raises(ValueError):
        make_grid(GridSpec(subset_frac=1.5))


def test_stratified_order_is_seeded_permutation():
    pts = make_grid(GridSpec(step_deg=4))
    o1 = stratified_order(pts, seed=3)
    o2 = stratified_order(pts, seed=3)
    o3 = stratified_order(pts, seed=4)
    assert o1 == o2
    assert o1 != o3
    assert sorted(p.idx for p in o1) == [p.idx for p in pts]


def _covers_globe(prefix):
    hemis = {(p.lat > 0, p.lon > 0) for p in prefix}
    sextants = {int((p.lon + 180) // 60) for p in prefix}
    bands = {int((p.lat + 90) // 45) for p in prefix}
    return len(hemis) == 4 and len(sextants) == 6 and len(bands) == 4


def test_stratified_prefix_covers_globe():
    pts = make_grid(GridSpec(step_deg=2))
    order = stratified_order(pts, seed=0)
    prefix = order[: len(order) // 100]  # first 1%
    assert _covers_globe(prefix)
    assert not _covers_globe(pts[: len(pts) // 100])  # row-major would not


def test_stratified_order_on_subset_and_tiny_inputs():
    pts = make_grid(GridSpec(step_deg=2, subset_frac=0.1, seed=0))
    order = stratified_order(pts, seed=0)
    assert sorted(p.idx for p in order) == [p.idx for p in pts]
    assert stratified_order([], 0) == []
    one = [Point(0, 1.0, 1.0, 1.0)]
    assert stratified_order(one, 0) == one


def test_cell_bounds():
    p = Point(0, 89.0, -179.0, 1.0)
    assert cell_bounds(p, 2, "cell_center") == (-180.0, 88.0, -178.0, 90.0)
    q = Point(0, 90.0, -180.0, 0.0)
    assert cell_bounds(q, 2, "cell_corner") == (-180.0, 88.0, -178.0, 90.0)
    r = Point(0, -88.0, 178.0, 1.0)
    assert cell_bounds(r, 2, "cell_corner") == (178.0, -90.0, 180.0, -88.0)
