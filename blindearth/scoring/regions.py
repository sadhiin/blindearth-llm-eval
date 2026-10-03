"""Region labels (continent, ocean, polar band) for region accuracy.

Self-contained approximation, no downloads and no polygon data. Each point is labelled from its
(lat, lon) and its *true* class:

- Polar bands first: lat >= 66.5 (Arctic Circle) -> ``"Arctic"`` (land and water together);
  lat < -60 -> ``"Antarctica"`` for land, ``"Southern Ocean"`` for water. Antarctica is its own
  label because mask conventions for ice shelves differ.
- True land -> continent from hand-tuned lat/lon boxes and a few straight lines:
  Greenland goes to North America; the Africa/Asia split follows a straight line along the Red
  Sea axis; Europe/Asia uses the Urals (60°E), the Ural river region (lon > 50, lat < 52),
  the Caucasus (lat < 43.5, lon > 39.5) and Anatolia (lon > 26.5, lat < 41; lon > 29.5,
  lat < 42.2); Central America/South America splits near the Panama–Colombia border; Oceania
  is Australia and everything with lon >= 130 south of 22°N, plus the central/eastern Pacific
  (lon < -120, lat < 30, which includes Hawaii). Land left over (remote islands) goes to the
  nearest of a few continent anchor points by great-circle distance.
- True water -> ``"Mediterranean & Black Sea"`` (boxes), ``"Inland waters"`` (Caspian and other
  lakes in Eurasia), ``"Indian Ocean"`` (20°E–147°E south of 31°N, excluding the South China
  Sea / Indonesian seas north of 8°S and the Arafura/Coral side east of 132°E north of 30°S),
  ``"Atlantic Ocean"`` (70°W–20°E, the Gulf of Mexico and Caribbean north-east of a straight
  line along the Central American isthmus, the Baltic, Drake Passage split at 67.3°W) and
  ``"Pacific Ocean"`` for the rest.

Cells near these hand-drawn borders (Sinai, Panama, Bosporus, Wallace line, small islands) can be
misassigned; at a 2° grid that is a handful of cells and region figures are indicative only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from blindearth.scoring.cell_metrics import point_arrays

ARCTIC_LAT = 66.5
ANTARCTIC_LAT = -60.0

CONTINENTS = ("North America", "South America", "Europe", "Africa", "Asia", "Oceania")
OCEANS = (
    "Atlantic Ocean",
    "Pacific Ocean",
    "Indian Ocean",
    "Mediterranean & Black Sea",
    "Inland waters",
)
POLAR = ("Arctic", "Antarctica", "Southern Ocean")

# (lat, lon, label) anchors for land not caught by a box.
_ANCHORS = [
    (45.0, -100.0, "North America"),
    (-15.0, -60.0, "South America"),
    (50.0, 15.0, "Europe"),
    (5.0, 20.0, "Africa"),
    (40.0, 90.0, "Asia"),
    (-25.0, 135.0, "Oceania"),
    (-15.0, -150.0, "Oceania"),
    (-5.0, 160.0, "Oceania"),
]


class _Assigner:
    def __init__(self, n: int):
        self.out = np.full(n, "", dtype=object)
        self.free = np.ones(n, dtype=bool)

    def __call__(self, cond: np.ndarray, label: str) -> None:
        m = cond & self.free
        self.out[m] = label
        self.free &= ~m


def _nearest_anchor(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la = np.deg2rad(lat)[:, None]
    lo = np.deg2rad(lon)[:, None]
    alat = np.deg2rad(np.array([a[0] for a in _ANCHORS]))[None, :]
    alon = np.deg2rad(np.array([a[1] for a in _ANCHORS]))[None, :]
    cosd = np.sin(la) * np.sin(alat) + np.cos(la) * np.cos(alat) * np.cos(lo - alon)
    best = np.argmax(cosd, axis=1)
    labels = np.array([a[2] for a in _ANCHORS], dtype=object)
    return labels[best]


def land_continent(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Continent of land points at any latitude (no polar band; lat < -60 -> Antarctica)."""
    lat = np.asarray(lat, dtype=float)
    lon = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    a = _Assigner(lat.size)
    a(lat < ANTARCTIC_LAT, "Antarctica")
    # Greenland (Iceland lies east of 30°W below 67°N and stays with Europe).
    a((lat >= 59) & (lon >= -75) & (lon <= -10) & ((lat >= 67) | (lon <= -30)), "North America")
    # Oceania: Australia/NZ, New Guinea and Micronesia/Melanesia, central/eastern Pacific.
    a(((lat < -10) & (lon > 110)) | ((lon >= 130) & (lat < 22)) | ((lon < -120) & (lat < 30)), "Oceania")
    # South America (Panama/Costa Rica excluded).
    a((lat < 12.5) & (lon > -92) & (lon < -34) & ~((lat > 7) & (lon < -77.2)), "South America")
    # North America, incl. Central America, Caribbean and the western Aleutians.
    na = (lat >= 7) & (lon >= -170) & (lon <= -50) & ~((lat > 55) & (lon < -168.5))
    a(na | ((lon >= 172) & (lat > 50) & (lat < 56)), "North America")
    # Africa: west of the Red Sea axis, south of the Mediterranean.
    red_sea_axis = 34.0 + (28.0 - lat) * 0.604
    africa = (
        (lon >= -26) & (lon <= 52) & (lat >= -36) & (lat < 35.95)
        & ~((lon > 11.6) & (lat > 33.4))
        & ~((lat >= 12.5) & (lat <= 30) & (lon > red_sea_axis))
        & ~((lat > 30) & (lon > 34.2))
    )
    a(africa, "Africa")
    europe = (
        (lon >= -32) & (lon <= 60) & (lat >= 34.5)
        & ~((lon > 26.5) & (lat < 41.0))
        & ~((lon > 29.5) & (lat < 42.2))
        & ~((lon > 39.5) & (lat < 43.5))
        & ~((lon > 50) & (lat < 52))
    )
    a(europe, "Europe")
    a((lon >= 26) | ((lon < -168) & (lat > 55)), "Asia")
    if a.free.any():
        a.out[a.free] = _nearest_anchor(lat[a.free], lon[a.free])
    return a.out


def water_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Ocean/sea of water points between 60°S and 66.5°N (see module docstring)."""
    lat = np.asarray(lat, dtype=float)
    lon = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    a = _Assigner(lat.size)
    med = (lat >= 30) & (lat <= 46) & (lon >= -5.6) & (lon <= 36.5) & ((lon >= 0) | (lat < 38))
    black = (lat >= 40.5) & (lat <= 47.5) & (lon >= 27) & (lon <= 42)
    a(med | black, "Mediterranean & Black Sea")
    a((lat >= 36.5) & (lat <= 47.5) & (lon >= 46.5) & (lon <= 55.5), "Inland waters")  # Caspian
    a((lat >= 50) & (lon >= 20) & (lon < 31), "Atlantic Ocean")  # Baltic
    a((lat >= 60) & (lon >= 31) & (lon < 100), "Arctic")  # White Sea and Arctic fringe below 66.5
    a((lat >= 31) & (lon >= 20) & (lon < 100), "Inland waters")
    indian = (
        (lon >= 20) & (lon <= 147) & (lat < 31)
        & ~((lon > 99) & (lat > -8))
        & ((lon < 132) | (lat < -30))
    )
    a(indian, "Indian Ocean")
    isthmus = 17.0 - (lon + 92.0) * 0.654  # Central American isthmus, (17N,92W)-(8.5N,79W)
    atlantic = (
        ((lon >= -70) & (lon < 20) & ~((lat < -54) & (lon < -67.3)))
        | ((lon >= -100) & (lon < -70) & (lat >= 8) & (lat > isthmus))
    )
    a(atlantic, "Atlantic Ocean")
    a(np.ones(lat.size, dtype=bool), "Pacific Ocean")
    return a.out


def region_labels(lat: np.ndarray, lon: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """String label per point (object array). Unknown truth -> ``"Unknown"``."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    t = np.asarray(truth, dtype=float)
    out = np.full(lat.size, "Unknown", dtype=object)
    known = np.isfinite(t)
    land = known & (t > 0.5)
    water = known & ~(t > 0.5)
    arctic = lat >= ARCTIC_LAT
    south = lat < ANTARCTIC_LAT
    out[known & arctic] = "Arctic"
    out[land & south] = "Antarctica"
    out[water & south] = "Southern Ocean"
    mid = ~arctic & ~south
    m = land & mid
    if m.any():
        out[m] = land_continent(lat[m], lon[m])
    m = water & mid
    if m.any():
        out[m] = water_region(lat[m], lon[m])
    return out


def region_accuracy(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """Area-weighted accuracy per region label (invalid answers count as wrong)."""
    if df is None or len(df) == 0:
        return {}
    a = point_arrays(df, threshold)
    labels = region_labels(a["lat"], a["lon"], a["truth"])
    k = a["known"]
    out: dict[str, float] = {}
    for lab in pd.unique(labels[k]):
        m = k & (labels == lab)
        sw = float(a["w"][m].sum())
        if sw > 0:
            out[str(lab)] = float((a["w"][m] * a["correct"][m]).sum() / sw)
    order = {n: i for i, n in enumerate(CONTINENTS + POLAR + OCEANS)}
    return dict(sorted(out.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))
