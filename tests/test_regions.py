from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blindearth.scoring import regions
from blindearth.scoring.regions import region_accuracy, region_labels


@pytest.fixture(autouse=True)
def _boxes_by_default(monkeypatch):
    """Box method unless a test opts in; never touch the network."""
    monkeypatch.setenv(regions.ENV_MODE, "boxes")
    regions.clear_cache()
    yield
    regions.clear_cache()

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


def test_box_method_recorded():
    df = pd.DataFrame({"lat": [0.0], "lon": [-30.0], "truth": [0], "weight": [1.0], "p_land": [0.0]})
    acc = region_accuracy(df)
    assert acc == {"Atlantic Ocean": 1.0}
    assert acc.method == regions.METHOD_BOXES
    assert region_accuracy(df.iloc[:0]) == {}


# --------------------------------------------------------------------------- polygon method


def _fake_index():
    pytest.importorskip("shapely")
    from shapely.geometry import box

    land = (
        [box(-20, -35, 50, 35), box(0, 0, 10, 10), box(100, -10, 110, 0), box(60, 60, 70, 70)],
        [
            {"FEATURECLA": "Continent", "REGION": "Africa"},
            {"FEATURECLA": "Island", "REGION": "Europe"},  # overlaps Africa; Continent wins
            {"FEATURECLA": "Island group", "REGION": "Oceania"},
            {"FEATURECLA": "Island", "REGION": "Seven seas (open ocean)"},  # unmapped: dropped
        ],
    )
    water = (
        [box(-80, 0, 0, 60), box(-10, 30, 10, 45), box(-180, -80, 180, -60)],
        [
            {"featurecla": "ocean", "name": "North Atlantic Ocean"},
            {"featurecla": "sea", "name": "Mediterranean Sea"},  # inside Atlantic; smallest wins
            {"featurecla": "ocean", "name": "SOUTHERN OCEAN"},
        ],
    )
    return regions.build_index(land, water)


@pytest.fixture
def poly(monkeypatch):
    idx = _fake_index()
    monkeypatch.setenv(regions.ENV_MODE, "auto")
    monkeypatch.setattr(regions, "_get_index", lambda mode, cache_dir=None: idx)
    regions.clear_cache()
    return idx


POLY_LAND = [
    (5.0, 5.0, "Africa"),  # in Africa continent and a smaller "Europe" island
    (-5.0, 105.0, "Oceania"),
    (36.5, 0.0, "Africa"),  # just outside: nearest polygon within NEAREST_MAX_DEG
    (45.0, -100.0, "North America"),  # far from any polygon: box fallback
    (65.0, 65.0, "Asia"),  # only an unmapped polygon here: box fallback
]
POLY_WATER = [
    (20.0, -40.0, "Atlantic Ocean"),
    (40.0, 0.0, "Mediterranean & Black Sea"),
    (-70.0, 0.0, "Southern Ocean"),
    (61.5, -40.0, "Atlantic Ocean"),  # coastal miss, nearest
    (0.0, 170.0, "Inland waters"),  # no marine polygon nearby
]


@pytest.mark.parametrize("lat,lon,expected", POLY_LAND)
def test_polygon_land(poly, lat, lon, expected):
    labels, method = regions.region_labels_with_method(np.array([lat]), np.array([lon]), np.array([1]))
    assert method == regions.METHOD_POLYGONS
    assert labels[0] == expected


@pytest.mark.parametrize("lat,lon,expected", POLY_WATER)
def test_polygon_water(poly, lat, lon, expected):
    assert region_labels(np.array([lat]), np.array([lon]), np.array([0]))[0] == expected


def test_polygon_unknown_and_lon_wrap(poly):
    labels = region_labels(np.array([20.0, 20.0]), np.array([0.0, 320.0]), np.array([np.nan, 0]))
    assert list(labels) == ["Unknown", "Atlantic Ocean"]


def test_polygon_accuracy_records_method(poly):
    import json

    df = pd.DataFrame({"lat": [5.0, 20.0], "lon": [5.0, -40.0], "truth": [1, 0],
                       "weight": [1.0, 1.0], "p_land": [1.0, 1.0]})
    acc = region_accuracy(df)
    assert acc == {"Africa": 1.0, "Atlantic Ocean": 0.0}
    assert acc.method == regions.METHOD_POLYGONS
    assert json.loads(json.dumps(acc)) == {"Africa": 1.0, "Atlantic Ocean": 0.0}


def test_labels_cached_per_grid(poly, monkeypatch):
    calls = []
    real = regions._polygon_labels

    def spy(*a):
        calls.append(1)
        return real(*a)

    monkeypatch.setattr(regions, "_polygon_labels", spy)
    lat, lon = np.array([5.0, 20.0]), np.array([5.0, -40.0])
    a = region_labels(lat, lon, np.array([1, 0]))
    a[0] = "mutated"  # callers get a copy
    b = region_labels(lat, lon, np.array([1.0, 0.0]))
    assert len(calls) == 1 and b[0] == "Africa"
    region_labels(lat, lon, np.array([0, 0]))
    assert len(calls) == 2


def test_falls_back_to_boxes_when_unavailable(monkeypatch):
    monkeypatch.setenv(regions.ENV_MODE, "auto")
    monkeypatch.setattr(regions, "_get_index", lambda mode, cache_dir=None: None)
    regions.clear_cache()
    labels, method = regions.region_labels_with_method(np.array([20.0]), np.array([-157.0]), np.array([1]))
    assert method == regions.METHOD_BOXES and labels[0] == "Oceania"


def test_load_index_offline(monkeypatch, tmp_path):
    from blindearth.evalspec import masks

    def boom(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr(masks, "_fetch", boom)
    regions._load_index.cache_clear()
    assert regions._load_index(str(tmp_path), True) is None


def test_load_index_cached_mode_never_downloads(monkeypatch, tmp_path):
    from blindearth.evalspec import masks

    def boom(*a, **k):
        raise AssertionError("must not download")

    monkeypatch.setattr(masks, "_fetch", boom)
    regions._load_index.cache_clear()
    assert regions._load_index(str(tmp_path), False) is None


def test_boxes_mode_skips_data(monkeypatch):
    monkeypatch.setenv(regions.ENV_MODE, "boxes")
    monkeypatch.setattr(regions, "_load_index", lambda *a: pytest.fail("loaded"))
    assert regions.region_method() == regions.METHOD_BOXES


def test_pytest_default_is_boxes(monkeypatch):
    monkeypatch.delenv(regions.ENV_MODE, raising=False)
    assert regions._mode() == "boxes"  # PYTEST_CURRENT_TEST is set while a test runs


# --------------------------------------------------------------------------- shapefile/dbf reading


def _shp(polys: list[list[tuple[float, float]]]) -> bytes:
    import struct

    recs = b""
    for i, ring in enumerate(polys, 1):
        pts = np.asarray(ring, dtype="<f8")
        content = struct.pack("<i4d", 5, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        content += struct.pack("<ii", 1, len(pts)) + struct.pack("<i", 0) + pts.tobytes()
        recs += struct.pack(">ii", i, len(content) // 2) + content
    header = struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", (100 + len(recs)) // 2)
    header += struct.pack("<ii4d4d", 1000, 5, -180, -90, 180, 90, 0, 0, 0, 0)
    return header + recs


def _dbf(fields: list[str], rows: list[list[str]], width: int = 40) -> bytes:
    import struct

    header_len = 32 + 32 * len(fields) + 1
    rec_len = 1 + width * len(fields)
    out = struct.pack("<B3BIHH20x", 3, 126, 1, 1, len(rows), header_len, rec_len)
    for f in fields:
        out += f.encode().ljust(11, b"\0") + b"C" + b"\0" * 4 + bytes([width, 0]) + b"\0" * 14
    out += b"\r"
    for r in rows:
        out += b" " + b"".join(v.encode().ljust(width) for v in r)
    return out + b"\x1a"


def _write_zip(path, stem, polys, fields, rows):
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(stem + ".shp", _shp(polys))
        zf.writestr(stem + ".dbf", _dbf(fields, rows))
        zf.writestr(stem + ".VERSION.txt", "5.1.0")


def _square(x0, y0, x1, y1):  # clockwise outer ring, closed
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]


def test_read_dbf_fields_and_deleted():
    buf = bytearray(_dbf(["FEATURECLA", "REGION"], [["Continent", "Africa"], ["Island", "Europe"]]))
    assert regions._read_dbf(bytes(buf)) == [
        {"FEATURECLA": "Continent", "REGION": "Africa"},
        {"FEATURECLA": "Island", "REGION": "Europe"},
    ]
    header_len = 32 + 32 * 2 + 1
    buf[header_len + (1 + 40 * 2)] = ord("*")  # delete the second record
    assert regions._read_dbf(bytes(buf))[1] is None


def test_marine_and_land_groups():
    assert regions.marine_group({"name": "Hudson  Bay"}) == "Arctic Ocean"
    assert regions.marine_group({"name": "Caspian Sea"}) == "Inland waters"
    assert regions.marine_group({"name": "Some North Atlantic bight"}) == "Atlantic Ocean"
    assert regions.marine_group({"name": "Unknown Sea"}) is None
    assert regions.land_group({"REGION": "North America"}) == "North America"
    assert regions.land_group({"REGION": "Seven seas (open ocean)"}) is None


def test_load_index_from_cached_archives(monkeypatch, tmp_path):
    """End to end from zipped shapefiles already in the cache dir (no network)."""
    pytest.importorskip("shapely")
    from blindearth.evalspec import masks

    dl = tmp_path / "downloads"
    dl.mkdir()
    _write_zip(dl / f"{regions._NE_LAND}.zip", regions._NE_LAND,
               [_square(-20, -35, 50, 35), _square(100, -10, 110, 0)],
               ["FEATURECLA", "NAME", "REGION"],
               [["Continent", "AFRICA", "Africa"], ["Island group", "MELANESIA", "Oceania"]])
    _write_zip(dl / f"{regions._NE_MARINE}.zip", regions._NE_MARINE,
               [_square(-80, 0, 0, 60), _square(40, -40, 100, 20)],
               ["featurecla", "name"],
               [["ocean", "North Atlantic Ocean"], ["ocean", "INDIAN OCEAN"]])
    monkeypatch.setattr(masks, "urllib", None)  # any download attempt would crash
    regions._load_index.cache_clear()
    idx = regions._load_index(str(tmp_path), False)
    assert idx is not None
    labels = regions._polygon_labels(
        idx, np.array([0.0, -5.0, 20.0, -10.0]), np.array([10.0, 105.0, -40.0, 70.0]),
        np.array([1, 1, 0, 0], dtype=np.int8),
    )
    assert list(labels) == ["Africa", "Oceania", "Atlantic Ocean", "Indian Ocean"]
