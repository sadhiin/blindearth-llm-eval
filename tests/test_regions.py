from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blindearth.scoring.regions import region_accuracy, region_labels

LAND = [
    (48.8, 2.3, "Europe"),  # Paris
    (40.0, -100.0, "North America"),
    (15.0, -88.0, "North America"),  # Central America
    (-15.0, -60.0, "South America"),
    (0.0, 20.0, "Africa"),
    (30.0, 31.0, "Africa"),  # Cairo
    (24.0, 45.0, "Asia"),  # Riyadh
    (39.0, 35.0, "Asia"),  # Anatolia
    (35.0, 100.0, "Asia"),
    (-25.0, 135.0, "Oceania"),
    (20.0, -157.0, "Oceania"),  # Hawaii
    (64.5, -18.0, "Europe"),  # Iceland
    (62.0, -45.0, "North America"),  # southern Greenland
    (-80.0, 0.0, "Antarctica"),
    (75.0, -40.0, "Arctic"),
]

WATER = [
    (0.0, -30.0, "Atlantic Ocean"),
    (25.0, -90.0, "Atlantic Ocean"),  # Gulf of Mexico
    (12.0, -95.0, "Pacific Ocean"),  # Pacific off Central America
    (0.0, -150.0, "Pacific Ocean"),
    (15.0, 115.0, "Pacific Ocean"),  # South China Sea
    (-20.0, 80.0, "Indian Ocean"),
    (20.0, 38.0, "Indian Ocean"),  # Red Sea
    (35.0, 18.0, "Mediterranean & Black Sea"),
    (43.0, 35.0, "Mediterranean & Black Sea"),
    (42.0, 51.0, "Inland waters"),  # Caspian
    (58.0, 20.0, "Atlantic Ocean"),  # Baltic
    (-65.0, 0.0, "Southern Ocean"),
    (80.0, 0.0, "Arctic"),
]


@pytest.mark.parametrize("lat,lon,expected", LAND)
def test_land_labels(lat, lon, expected):
    assert region_labels(np.array([lat]), np.array([lon]), np.array([1]))[0] == expected


@pytest.mark.parametrize("lat,lon,expected", WATER)
def test_water_labels(lat, lon, expected):
    assert region_labels(np.array([lat]), np.array([lon]), np.array([0]))[0] == expected


def test_unknown_truth():
    assert region_labels(np.array([0.0]), np.array([0.0]), np.array([np.nan]))[0] == "Unknown"


def test_region_accuracy_counts_invalid_as_wrong():
    rows = [(la, lo, 1) for la, lo, _ in LAND] + [(la, lo, 0) for la, lo, _ in WATER]
    df = pd.DataFrame(rows, columns=["lat", "lon", "truth"])
    df["weight"] = np.cos(np.deg2rad(df["lat"]))
    df["p_land"] = df["truth"].astype(float)
    df.loc[df["lat"] == 48.8, "p_land"] = np.nan  # invalid answer in Europe
    acc = region_accuracy(df)
    assert acc["Europe"] == pytest.approx(
        np.cos(np.deg2rad(64.5)) / (np.cos(np.deg2rad(64.5)) + np.cos(np.deg2rad(48.8)))
    )
    assert acc["Africa"] == pytest.approx(1.0)
    assert acc["Antarctica"] == pytest.approx(1.0)
    assert "Pacific Ocean" in acc
