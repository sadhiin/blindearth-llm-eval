from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blindearth.scoring.cell_metrics import cell_metrics


def grid_df(truth_fn, p_fn, step: float = 2.0) -> pd.DataFrame:
    lats = 90 - step / 2 - step * np.arange(int(180 / step))
    lons = -180 + step / 2 + step * np.arange(int(360 / step))
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    lat, lon = LA.ravel(), LO.ravel()
    truth = truth_fn(lat, lon).astype(int)
    p = p_fn(lat, lon, truth)
    return pd.DataFrame(
        {
            "idx": np.arange(lat.size),
            "lat": lat,
            "lon": lon,
            "weight": np.cos(np.deg2rad(lat)),
            "truth": truth,
            "p_land": p,
            "latency_s": np.full(lat.size, 0.5),
            "input_tokens": np.full(lat.size, 50),
            "output_tokens": np.full(lat.size, 1),
            "thinking_tokens": np.zeros(lat.size, dtype=int),
        }
    )


def box(lat, lon):
    return (np.abs(lat) < 30) & (np.abs(lon) < 60)


def test_perfect_prediction():
    df = grid_df(box, lambda la, lo, t: t.astype(float))
    m = cell_metrics(df)
    assert m["acc_area"] == pytest.approx(1.0)
    assert m["acc_unweighted"] == pytest.approx(1.0)
    assert m["skill"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["iou"] == pytest.approx(1.0)
    assert m["coastline_error_deg"] == pytest.approx(0.0)
    assert m["invalid_rate"] == 0.0
    assert m["n_points"] == 16200
    assert m["input_tokens"] == 16200 * 50
    assert m["latency_p50"] == pytest.approx(0.5)


def test_all_water_equals_baseline():
    df = grid_df(box, lambda la, lo, t: np.zeros(t.size))
    m = cell_metrics(df)
    assert m["acc_area"] == pytest.approx(m["baseline_area"])
    assert m["skill"] == pytest.approx(0.0)
    assert m["recall"] == 0.0
    assert m["precision"] is None
    assert m["iou"] == 0.0
    assert m["coastline_error_deg"] is None  # no predicted boundary


def test_baseline_is_area_weighted_water_share():
    df = grid_df(box, lambda la, lo, t: t.astype(float))
    w = df["weight"].to_numpy()
    expected = (w * (df["truth"].to_numpy() == 0)).sum() / w.sum()
    assert cell_metrics(df)["baseline_area"] == pytest.approx(expected)


def test_invalid_counts_as_wrong():
    df = grid_df(box, lambda la, lo, t: t.astype(float))
    df.loc[df.index[:1000], "p_land"] = np.nan
    m = cell_metrics(df)
    assert m["acc_area"] < 1.0
    assert m["acc_valid_only"] == pytest.approx(1.0)
    assert m["invalid_rate"] == pytest.approx(1000 / 16200)
    assert m["n_valid"] == 15200


def test_probabilistic_scores():
    df = grid_df(box, lambda la, lo, t: np.where(t == 1, 0.9, 0.1))
    m = cell_metrics(df, probabilistic=True)
    assert m["brier"] == pytest.approx(0.01)
    assert m["ece"] == pytest.approx(0.1)
    assert m["log_loss"] == pytest.approx(-np.log(0.9))
    assert m["calibration_bins"]
    m2 = cell_metrics(df, probabilistic=False)
    assert m2["brier"] is None and m2["log_loss"] is None and m2["ece"] is None


def test_threshold_changes_prediction():
    df = grid_df(box, lambda la, lo, t: np.where(t == 1, 0.6, 0.1))
    assert cell_metrics(df, threshold=0.5)["acc_area"] == pytest.approx(1.0)
    assert cell_metrics(df, threshold=0.7)["acc_area"] < 1.0


def test_coastline_shift_positive_and_small():
    df = grid_df(box, lambda la, lo, t: box(la, lo - 2).astype(float))
    ce = cell_metrics(df)["coastline_error_deg"]
    assert ce is not None and 0 < ce <= 2.0 + 1e-9


def test_coastline_wraps_across_seam():
    # Truth: land column just east of 180°W; prediction: land column just west of 180°E.
    t_fn = lambda la, lo: (lo == -179) & (np.abs(la) < 20)  # noqa: E731
    df = grid_df(t_fn, lambda la, lo, t: ((lo == 179) & (np.abs(la) < 20)).astype(float))
    ce = cell_metrics(df, step_deg=2.0, placement="cell_center")["coastline_error_deg"]
    assert ce is not None and ce < 3.0  # would be ~hundreds of degrees without wrap handling


def test_partial_run_and_empty():
    df = grid_df(box, lambda la, lo, t: t.astype(float)).sample(frac=0.3, random_state=0)
    m = cell_metrics(df)
    assert m["n_points"] == len(df)
    assert m["acc_area"] == pytest.approx(1.0)
    empty = cell_metrics(df.iloc[:0])
    assert empty["n_points"] == 0 and empty["acc_area"] is None
