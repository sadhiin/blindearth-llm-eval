from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from blindearth.scoring.bootstrap import (
    acc_area_stat,
    block_bootstrap_ci,
    block_ids,
    paired_difference,
    repeat_spread,
)


def make_df(error_rate: float = 0.0, seed: int = 0, step: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = 90 - step / 2 - step * np.arange(int(180 / step))
    lons = -180 + step / 2 + step * np.arange(int(360 / step))
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    lat, lon = LA.ravel(), LO.ravel()
    truth = ((np.abs(lat) < 40) & (np.abs(lon) < 70)).astype(int)
    flip = rng.random(lat.size) < error_rate
    p = np.where(flip, 1 - truth, truth).astype(float)
    return pd.DataFrame(
        {"idx": np.arange(lat.size), "lat": lat, "lon": lon,
         "weight": np.cos(np.deg2rad(lat)), "truth": truth, "p_land": p}
    )


def test_block_ids_10deg():
    ids = block_ids(np.array([89.0, -89.0, 85.0]), np.array([-179.0, 179.0, -171.0]))
    assert ids[0] == 0
    assert ids[1] == 17 * 36 + 35
    assert ids[2] == 0


def test_constant_stat_has_zero_width_ci():
    est, lo, hi = block_bootstrap_ci(make_df(0.0), acc_area_stat())
    assert est == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_ci_brackets_estimate():
    est, lo, hi = block_bootstrap_ci(make_df(0.1), acc_area_stat())
    assert lo <= est <= hi
    assert hi - lo > 0


def test_generic_callable_path_matches_fast_path_roughly():
    df = make_df(0.1)
    stat = acc_area_stat()
    fast = block_bootstrap_ci(df, stat, n_boot=200, seed=1)
    slow = block_bootstrap_ci(df, lambda d: stat(d), n_boot=200, seed=1)
    assert fast[0] == pytest.approx(slow[0])
    assert abs(fast[1] - slow[1]) < 0.02 and abs(fast[2] - slow[2]) < 0.02


def test_fast_path_is_fast():
    df = make_df(0.1)
    t0 = time.perf_counter()
    block_bootstrap_ci(df, acc_area_stat(), n_boot=1000)
    assert time.perf_counter() - t0 < 2.0


def test_paired_identical_runs_not_significant():
    df = make_df(0.1)
    r = paired_difference(df, df.copy())
    assert r["diff"] == pytest.approx(0.0)
    assert r["significant"] is False
    assert r["n_paired"] == len(df)


def test_paired_clear_difference_is_significant():
    r = paired_difference(make_df(0.0), make_df(0.2, seed=3))
    assert r["diff"] > 0.1
    assert r["lo"] > 0 and r["significant"] is True


def test_paired_aligns_on_idx_with_partial_runs():
    a = make_df(0.05)
    b = make_df(0.05, seed=2).sample(frac=0.5, random_state=0)
    r = paired_difference(a, b)
    assert r["n_paired"] == len(b)


def test_paired_rejects_different_grids():
    with pytest.raises(ValueError):
        paired_difference(make_df(0.0, step=2.0), make_df(0.0, step=4.0))


def test_repeat_spread():
    dfs = [make_df(0.1, seed=s) for s in range(3)]
    r = repeat_spread(dfs)
    assert r["n"] == 3
    assert r["min"] <= r["mean"] <= r["max"]
    assert r["std"] >= 0
