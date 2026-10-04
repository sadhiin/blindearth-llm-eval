"""Region labels (continent, ocean) for region accuracy.

Two methods; the one used is recorded (``region_labels_with_method``, ``RegionAccuracy.method``)
so reports can show it.

**Natural Earth polygons** (``method == "natural-earth-110m"``, preferred). Two Natural Earth
1:110m physical layers are downloaded once into the mask cache dir (``<cache>/downloads``, see
``blindearth.evalspec.masks``) and indexed with a shapely STRtree:

- land: ``ne_110m_geography_regions_polys`` — every feature has a ``REGION`` field (Africa,
  Antarctica, Asia, Europe, North America, Oceania, South America). Where features overlap,
  ``FEATURECLA == "Continent"`` wins, then the smallest polygon.
- water: ``ne_110m_geography_marine_polys`` — oceans and major seas by ``name``, grouped to
  Atlantic / Pacific / Indian / Arctic / Southern Ocean, ``"Mediterranean & Black Sea"`` and
  ``"Inland waters"`` (Caspian). Baffin Bay, Hudson Bay and the Beaufort Sea count as Arctic
  (IHO S-23); Weddell and Ross seas as Southern Ocean. Where features overlap the smallest wins.

Each point is labelled by its *true* class: land points against land polygons, water points
against marine polygons. A point in no polygon (coastal cells, 110m generalisation) takes the
nearest polygon within ``NEAREST_MAX_DEG``; beyond that, land falls back to the box method below
(remote islands) and water is ``"Inland waters"`` (lakes). There is no polar band in this method:
Arctic land goes to its continent and Antarctic land is ``"Antarctica"``.

The data is used when it can be loaded. ``$BLINDEARTH_REGIONS`` controls this: ``auto``
(default; download if missing), ``cached`` (use only files already in the cache), ``boxes``
(never). Under pytest the default is ``boxes`` so tests never touch the network.

**Lat/lon boxes** (``method == "lat-lon-boxes"``, fallback when the data or shapely is
unavailable). Self-contained approximation, no polygon data. Each point is labelled from its
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

import functools
import logging
import os
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blindearth.scoring.cell_metrics import point_arrays

log = logging.getLogger(__name__)

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

# Fixed label set of the polygon method.
POLY_LAND = CONTINENTS + ("Antarctica",)
POLY_WATER = (
    "Atlantic Ocean",
    "Pacific Ocean",
    "Indian Ocean",
    "Arctic Ocean",
    "Southern Ocean",
    "Mediterranean & Black Sea",
    "Inland waters",
)

METHOD_POLYGONS = "natural-earth-110m"
METHOD_BOXES = "lat-lon-boxes"
ENV_MODE = "BLINDEARTH_REGIONS"  # auto | cached | boxes
NEAREST_MAX_DEG = 3.0

_NE_BASE_URL = "https://naciscdn.org/naturalearth/110m/physical/"
_NE_LAND = "ne_110m_geography_regions_polys"
_NE_MARINE = "ne_110m_geography_marine_polys"

# marine_polys ``name`` (lower-cased) -> label. Unlisted names fall back to keywords.
_MARINE_GROUPS: dict[str, str] = {
    **dict.fromkeys(
        ("arctic ocean", "beaufort sea", "baffin bay", "hudson bay"), "Arctic Ocean"
    ),
    **dict.fromkeys(("southern ocean", "weddell sea", "ross sea"), "Southern Ocean"),
    **dict.fromkeys(
        ("north atlantic ocean", "south atlantic ocean", "caribbean sea", "gulf of mexico",
         "labrador sea"),
        "Atlantic Ocean",
    ),
    **dict.fromkeys(
        ("north pacific ocean", "south pacific ocean", "philippine sea", "tasman sea",
         "south china sea", "coral sea", "sea of okhotsk", "sea of japan", "gulf of alaska"),
        "Pacific Ocean",
    ),
    **dict.fromkeys(
        ("indian ocean", "bay of bengal", "arabian sea", "red sea", "persian gulf"),
        "Indian Ocean",
    ),
    **dict.fromkeys(("mediterranean sea", "black sea"), "Mediterranean & Black Sea"),
    "caspian sea": "Inland waters",
}
_REGION_GROUPS: dict[str, str] = {x.lower(): x for x in POLY_LAND}


class RegionAccuracy(dict):
    """``dict[str, float]`` (label -> accuracy) that also carries ``method``, the labelling
    method used (``METHOD_POLYGONS`` or ``METHOD_BOXES``; ``None`` for an empty input)."""

    def __init__(self, *args: Any, method: str | None = None, **kw: Any):
        super().__init__(*args, **kw)
        self.method = method

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


def _box_labels(lat: np.ndarray, lon: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Box method. ``cls``: 1 land, 0 water, -1 unknown."""
    out = np.full(lat.size, "Unknown", dtype=object)
    known = cls >= 0
    land = cls == 1
    water = cls == 0
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


# --------------------------------------------------------------------------- Natural Earth data


def _read_dbf(buf: bytes) -> list[dict[str, str] | None]:
    """Minimal dBASE III reader: one dict of stripped strings per record (None if deleted)."""
    if len(buf) < 32:
        raise ValueError("not a dBASE file (.dbf)")
    n_rec, header_len, rec_len = struct.unpack("<IHH", buf[4:12])
    fields: list[tuple[str, int]] = []
    pos = 32
    while pos + 32 <= header_len and buf[pos] != 0x0D:
        name = buf[pos : pos + 11].split(b"\0", 1)[0].decode("ascii", "replace")
        fields.append((name, buf[pos + 16]))
        pos += 32
    out: list[dict[str, str] | None] = []
    for i in range(n_rec):
        start = header_len + i * rec_len
        rec = buf[start : start + rec_len]
        if len(rec) < rec_len:
            raise ValueError("truncated .dbf")
        if rec[:1] == b"*":
            out.append(None)
            continue
        row: dict[str, str] = {}
        p = 1
        for name, width in fields:
            row[name] = rec[p : p + width].decode("utf-8", "replace").strip()
            p += width
        out.append(row)
    return out


def _zip_member(names: list[str], filename: str) -> str:
    for n in names:
        if n.rsplit("/", 1)[-1].lower() == filename.lower():
            return n
    raise ValueError(f"{filename} not found in archive")


def _rings_to_geom(rings: list[np.ndarray]):
    """Shapefile rings -> shapely geometry (even-odd rule, so holes cut out). None if empty."""
    from shapely.geometry import Polygon

    geom = None
    for r in rings:
        p = Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        geom = p if geom is None else geom.symmetric_difference(p)
    return geom


def _read_layer(zip_path: Path, stem: str) -> tuple[list[Any], list[dict[str, str] | None]]:
    """(geometries, attribute rows) of a zipped shapefile, aligned by record."""
    from blindearth.evalspec.masks import read_shp_polygons

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        shp = zf.read(_zip_member(names, stem + ".shp"))
        dbf = zf.read(_zip_member(names, stem + ".dbf"))
    rows = _read_dbf(dbf)
    shapes = list(read_shp_polygons(shp))
    if len(shapes) != len(rows):
        raise ValueError(f"{stem}: {len(shapes)} polygon records but {len(rows)} attribute rows")
    return [_rings_to_geom(rings) for _, rings in shapes], rows


def _field(row: dict[str, str], name: str) -> str:
    for k, v in row.items():
        if k.lower() == name:
            return v
    return ""


def land_group(row: dict[str, str]) -> str | None:
    """Continent label of a geography_regions_polys row (``REGION`` field), or None."""
    return _REGION_GROUPS.get(" ".join(_field(row, "region").split()).lower())


def marine_group(row: dict[str, str]) -> str | None:
    """Ocean label of a geography_marine_polys row (``name`` field), or None."""
    name = " ".join(_field(row, "name").split()).lower()
    if name in _MARINE_GROUPS:
        return _MARINE_GROUPS[name]
    for key, label in (("atlantic", "Atlantic Ocean"), ("pacific", "Pacific Ocean"),
                       ("indian", "Indian Ocean"), ("arctic", "Arctic Ocean"),
                       ("southern", "Southern Ocean"), ("antarctic", "Southern Ocean")):
        if key in name:
            return label
    return None


@dataclass
class _LayerIndex:
    tree: Any  # shapely.STRtree
    labels: np.ndarray  # object, per tree geometry
    prio: np.ndarray  # float, lower wins when a point is in several polygons


@dataclass
class _PolyIndex:
    land: _LayerIndex
    water: _LayerIndex


def _make_layer(geoms: list[Any], labels: list[str | None], prio: list[float]) -> _LayerIndex:
    from shapely import STRtree

    keep = [i for i, (g, lab) in enumerate(zip(geoms, labels)) if g is not None and lab]
    if not keep:
        raise ValueError("no usable polygons")
    return _LayerIndex(
        tree=STRtree([geoms[i] for i in keep]),
        labels=np.array([labels[i] for i in keep], dtype=object),
        prio=np.array([prio[i] for i in keep], dtype=float),
    )


def build_index(land: tuple[list[Any], list[dict[str, str] | None]],
                water: tuple[list[Any], list[dict[str, str] | None]]) -> _PolyIndex:
    """Index (geometries, rows) of the land (regions) and water (marine) layers."""
    lg, lr = land
    l_labels = [land_group(r) if r else None for r in lr]
    # Continent polygons first, then the smallest feature (e.g. an island group).
    l_prio = [
        (0.0 if r and _field(r, "featurecla").lower() == "continent" else 1e6)
        + (g.area if g is not None else 0.0)
        for g, r in zip(lg, lr)
    ]
    wg, wr = water
    w_labels = [marine_group(r) if r else None for r in wr]
    w_prio = [g.area if g is not None else 0.0 for g in wg]
    return _PolyIndex(_make_layer(lg, l_labels, l_prio), _make_layer(wg, w_labels, w_prio))


def _ne_source(stem: str):
    from blindearth.evalspec.masks import _Source

    return _Source(
        id=stem,
        label=f"Natural Earth 1:110m {stem}",
        url=_NE_BASE_URL + stem + ".zip",
        sha256=None,  # no upstream sha256: trust-on-first-use via masks' checksums.json
        filename=stem + ".zip",
        layers=(),
        min_size=10_000,  # rejects HTML error pages / truncated files
    )


@functools.lru_cache(maxsize=4)
def _load_index(cache_dir: str, allow_download: bool) -> _PolyIndex | None:
    """Load (downloading once if allowed) and index the polygons; None when unavailable."""
    try:
        import shapely  # noqa: F401

        from blindearth.evalspec.masks import _fetch

        cache = Path(cache_dir)
        paths: dict[str, Path] = {}
        for stem in (_NE_LAND, _NE_MARINE):
            src = _ne_source(stem)
            if not allow_download and not (cache / "downloads" / src.filename).exists():
                log.info("region polygons not cached (%s); using lat/lon boxes", src.filename)
                return None
            paths[stem] = _fetch(src, cache)
        return build_index(_read_layer(paths[_NE_LAND], _NE_LAND),
                           _read_layer(paths[_NE_MARINE], _NE_MARINE))
    except Exception as exc:  # noqa: BLE001 - offline, missing shapely, bad archive
        log.warning("Natural Earth region polygons unavailable (%s); using lat/lon boxes", exc)
        return None


def _mode() -> str:
    mode = os.environ.get(ENV_MODE, "").strip().lower()
    if mode in ("auto", "cached", "boxes"):
        return mode
    return "boxes" if "PYTEST_CURRENT_TEST" in os.environ else "auto"


def _get_index(mode: str, cache_dir: str | None = None) -> _PolyIndex | None:
    if mode == "boxes":
        return None
    if cache_dir is None:
        from blindearth.evalspec.masks import default_cache_dir

        cache_dir = str(default_cache_dir())
    return _load_index(cache_dir, mode == "auto")


def _layer_labels(idx: _LayerIndex, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Label per point from the containing polygon, else the nearest within NEAREST_MAX_DEG.
    Object array; None where neither applies."""
    import shapely

    out = np.full(lat.size, None, dtype=object)
    if lat.size == 0:
        return out
    pts = shapely.points(lon, lat)
    hits = idx.tree.query(pts, predicate="intersects")
    if hits.size:
        pi, ti = hits
        order = np.lexsort((idx.prio[ti], pi))
        pi, ti = pi[order], ti[order]
        _, first = np.unique(pi, return_index=True)
        out[pi[first]] = idx.labels[ti[first]]
    miss = np.flatnonzero(np.array([x is None for x in out], dtype=bool))
    if miss.size:
        near = idx.tree.query_nearest(pts[miss], max_distance=NEAREST_MAX_DEG, all_matches=False)
        if near.size:
            out[miss[near[0]]] = idx.labels[near[1]]
    return out


def _polygon_labels(idx: _PolyIndex, lat: np.ndarray, lon: np.ndarray, cls: np.ndarray) -> np.ndarray:
    out = np.full(lat.size, "Unknown", dtype=object)
    for value, layer in ((1, idx.land), (0, idx.water)):
        m = np.flatnonzero(cls == value)
        if not m.size:
            continue
        lab = _layer_labels(layer, lat[m], lon[m])
        miss = np.array([x is None for x in lab], dtype=bool)
        if miss.any():
            lab[miss] = (land_continent(lat[m][miss], lon[m][miss]) if value == 1
                         else "Inland waters")
        out[m] = lab
    return out


@functools.lru_cache(maxsize=32)
def _labels_cached(lat_b: bytes, lon_b: bytes, cls_b: bytes, mode: str,
                   cache_dir: str | None) -> tuple[np.ndarray, str]:
    lat = np.frombuffer(lat_b, dtype=float)
    lon = np.frombuffer(lon_b, dtype=float)
    cls = np.frombuffer(cls_b, dtype=np.int8)
    idx = _get_index(mode, cache_dir)
    if idx is not None:
        labels, method = _polygon_labels(idx, lat, lon, cls), METHOD_POLYGONS
    else:
        labels, method = _box_labels(lat, lon, cls), METHOD_BOXES
    labels.setflags(write=False)
    return labels, method


def clear_cache() -> None:
    """Forget cached labels and loaded polygons (e.g. after changing ``$BLINDEARTH_REGIONS``)."""
    _labels_cached.cache_clear()
    _load_index.cache_clear()


def region_method(cache_dir: str | Path | None = None) -> str:
    """The labelling method region functions would use now (may load/download the data)."""
    idx = _get_index(_mode(), None if cache_dir is None else str(cache_dir))
    return METHOD_POLYGONS if idx is not None else METHOD_BOXES


def region_labels_with_method(lat: np.ndarray, lon: np.ndarray, truth: np.ndarray, *,
                              cache_dir: str | Path | None = None) -> tuple[np.ndarray, str]:
    """``(labels, method)``. Labels per grid point are cached by (lat, lon, truth class)."""
    lat = np.ascontiguousarray(lat, dtype=float).ravel()
    lon = ((np.ascontiguousarray(lon, dtype=float).ravel() + 180.0) % 360.0) - 180.0
    t = np.asarray(truth, dtype=float).ravel()
    cls = np.where(np.isfinite(t), (t > 0.5).astype(np.int8), np.int8(-1)).astype(np.int8)
    labels, method = _labels_cached(
        lat.tobytes(), np.ascontiguousarray(lon).tobytes(), cls.tobytes(), _mode(),
        None if cache_dir is None else str(cache_dir),
    )
    return labels.copy(), method


def region_labels(lat: np.ndarray, lon: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """String label per point (object array). Unknown truth -> ``"Unknown"``."""
    return region_labels_with_method(lat, lon, truth)[0]


_ORDER = {
    n: i
    for i, n in enumerate(
        dict.fromkeys(CONTINENTS + POLAR + ("Arctic Ocean",) + OCEANS + POLY_WATER)
    )
}


def region_accuracy(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """Area-weighted accuracy per region label (invalid answers count as wrong).

    Returns a :class:`RegionAccuracy` (a plain ``dict[str, float]`` with a ``method``
    attribute naming the labelling method used).
    """
    if df is None or len(df) == 0:
        return RegionAccuracy()
    a = point_arrays(df, threshold)
    labels, method = region_labels_with_method(a["lat"], a["lon"], a["truth"])
    k = a["known"]
    out: dict[str, float] = {}
    for lab in pd.unique(labels[k]):
        m = k & (labels == lab)
        sw = float(a["w"][m].sum())
        if sw > 0:
            out[str(lab)] = float((a["w"][m] * a["correct"][m]).sum() / sw)
    return RegionAccuracy(
        sorted(out.items(), key=lambda kv: (_ORDER.get(kv[0], 99), kv[0])), method=method
    )
