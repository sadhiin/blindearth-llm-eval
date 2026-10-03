"""Ground-truth land masks.

A mask is a ``bool`` array of shape (H, W), equirectangular, row 0 = 90 N, column 0 = 180 W,
True = land. Pixel ``(i, j)`` covers lat ``[90 - (i+1)*180/H, 90 - i*180/H]`` and lon
``[-180 + j*360/W, -180 + (j+1)*360/W]``.

Built-in sources (``MaskSpec.id``):

- ``natural-earth-land``: Natural Earth 1:10m land polygons, downloaded on first use from a pinned
  URL into the cache dir (``~/.cache/blindearth`` or ``$BLINDEARTH_CACHE``) and rasterized at
  ``MaskSpec.resolution_km`` (default 1 km, about 40076 x 20038 pixels, ~800 MB as bool, ~100 MB
  packed on disk). Rasterization runs in horizontal strips so peak memory stays near the size of
  the output array.
- ``gshhg``: GSHHG 2.3.7 full-resolution shorelines. Land = L1 (continents/islands) + L5
  (Antarctica ice front), minus L2 (lakes), plus L3 (islands in lakes), minus L4 (ponds).
- ``modis-mod44w``: MODIS MOD44W water mask. NASA serves it only as sinusoidal HDF tiles behind an
  Earthdata login, so there is no stable single-file URL and the download is a MANUAL step:

      1. Download the MOD44W v6.1 tiles for one year from LP DAAC (https://lpdaac.usgs.gov/).
      2. Mosaic and reproject the ``water_mask`` layer to a global EPSG:4326 GeoTIFF, e.g.
         ``gdalwarp -t_srs EPSG:4326 -te -180 -90 180 90 -tr 0.0025 0.0025 -r mode \
           HDF4_EOS:EOS_GRID:"<tile>.hdf":MOD44W_250m_GRID:water_mask ... mod44w_global.tif``
      3. Put it at ``<cache_dir>/modis-mod44w/mod44w_global.tif`` or set ``MaskSpec.path``.

  Pixel value 0 is land; 1 (water) and any fill/nodata value are water.
- ``upload``: a user image (PNG, JPEG, or GeoTIFF) at ``MaskSpec.path``, prepared by
  :func:`prepare_upload` and downsampled to ``resolution_km`` only when it is finer than that.

Pinned checksums are placeholders (``None``) until someone records them from a trusted download;
while a checksum is ``None`` the file's sha256 is logged instead of verified. Either way the
mask's own content hash (``Mask.hash``) goes into the spec hash, so a changed download can never
be mixed silently with old runs.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from blindearth.types import MaskSpec

log = logging.getLogger(__name__)

EARTH_EQUATOR_KM = 40075.017
MERCATOR_MAX_LAT = 85.0511287798066
_STRIP_ROWS = 2048
_HASH_STRIP_ROWS = 1024  # multiple of 8, so strip-wise packbits == whole-array packbits
_PREVIEW_MAX_WIDTH = 2048
_MAX_UPLOAD_PIXELS = 1_600_000_000  # 40000 x 40000; PIL's bomb guard otherwise stops at ~179M


# --------------------------------------------------------------------------- Mask


def mask_hash(data: np.ndarray) -> str:
    """sha256 over the shape and ``np.packbits`` of the mask (row-major), computed in strips."""
    if data.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {data.shape}")
    h = hashlib.sha256()
    h.update(f"blindearth-mask:{data.shape[0]}x{data.shape[1]}:".encode())
    for r0 in range(0, data.shape[0], _HASH_STRIP_ROWS):
        strip = np.ascontiguousarray(data[r0 : r0 + _HASH_STRIP_ROWS], dtype=bool)
        h.update(np.packbits(strip, axis=None).tobytes())
    return h.hexdigest()


@dataclass
class Mask:
    data: np.ndarray  # bool (H, W)
    source: str  # human label printed in reports
    hash: str  # sha256 of packed bits + shape

    @classmethod
    def from_array(cls, data: np.ndarray, source: str) -> Mask:
        arr = np.asarray(data)
        if arr.dtype != bool:
            arr = arr.astype(bool)
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
        return cls(data=arr, source=source, hash=mask_hash(arr))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    def land_fraction(self, area_weighted: bool = True) -> float:
        """Share of land pixels; cos(lat)-weighted by default (about 0.29 for Earth)."""
        h = self.data.shape[0]
        lat_c = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
        w = np.cos(np.radians(lat_c)) if area_weighted else np.ones(h)
        row_frac = self.data.mean(axis=1)
        return float((row_frac * w).sum() / w.sum())


# --------------------------------------------------------------------------- helpers


def default_cache_dir() -> Path:
    return Path(os.environ.get("BLINDEARTH_CACHE", "~/.cache/blindearth")).expanduser()


def shape_for_resolution(resolution_km: float) -> tuple[int, int]:
    """(H, W) of a global equirectangular grid with ~``resolution_km`` pixels at the equator."""
    res = float(resolution_km)
    if not math.isfinite(res) or res <= 0:
        raise ValueError(f"resolution_km must be > 0, got {resolution_km!r}")
    w = 2 * max(180, int(round(EARTH_EQUATOR_KM / res / 2.0)))
    if w > 200_000:
        raise ValueError(f"resolution_km={res} would need a {w}-pixel-wide mask; use >= 0.2 km")
    return w // 2, w


def resample_mask(data: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Resample a bool mask to ``out_shape``.

    Downsampling on an axis is an area majority: each output pixel takes the mean of the input
    pixels whose index range it covers (edges rounded down to whole input pixels) and is land when
    that mean is > 0.5 (ties go to water). Upsampling is nearest neighbour. Work runs in row strips
    so memory stays near the size of the output.
    """
    h_in, w_in = data.shape
    h, w = int(out_shape[0]), int(out_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"bad output shape {out_shape}")
    if (h, w) == (h_in, w_in):
        return data
    if h > h_in or w > w_in:
        # Nearest-neighbour on the growing axes first, then majority on the shrinking ones.
        rows = np.arange(h_in) if h <= h_in else ((np.arange(h) + 0.5) * h_in / h).astype(np.int64)
        cols = np.arange(w_in) if w <= w_in else ((np.arange(w) + 0.5) * w_in / w).astype(np.int64)
        grown = data[np.ix_(rows, cols)]
        if grown.shape == (h, w):
            return grown
        return resample_mask(grown, (h, w))

    row_edges = np.floor(np.arange(h + 1) * (h_in / h)).astype(np.int64)
    col_edges = np.floor(np.arange(w + 1) * (w_in / w)).astype(np.int64)
    row_edges[-1], col_edges[-1] = h_in, w_in
    row_counts = np.diff(row_edges)
    col_counts = np.diff(col_edges)
    col_starts = col_edges[:-1]
    out = np.empty((h, w), dtype=bool)
    step = max(1, _STRIP_ROWS // max(1, h_in // h))
    for i0 in range(0, h, step):
        i1 = min(h, i0 + step)
        rs, re_ = row_edges[i0], row_edges[i1]
        block = data[rs:re_]
        if block.dtype != bool:
            block = block.astype(bool)
        block = block.view(np.uint8)
        row_sum = np.add.reduceat(block, row_edges[i0:i1] - rs, axis=0, dtype=np.uint32)
        cell_sum = np.add.reduceat(row_sum, col_starts, axis=1, dtype=np.uint64)
        counts = row_counts[i0:i1, None].astype(np.uint64) * col_counts[None, :].astype(np.uint64)
        out[i0:i1] = cell_sum * 2 > counts
    return out


# --------------------------------------------------------------------------- downloads


@dataclass(frozen=True)
class _Layer:
    member: str  # path inside the zip
    burn: int  # 1 = land, 0 = water
    required: bool = True


@dataclass(frozen=True)
class _Source:
    id: str
    label: str
    url: str
    sha256: str | None  # pinned checksum; None = placeholder, not yet recorded
    filename: str
    layers: tuple[_Layer, ...]


NATURAL_EARTH = _Source(
    id="natural-earth-land",
    label="Natural Earth 1:10m land v5.1",
    url="https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
    sha256=None,  # TODO: pin after a verified download
    filename="ne_10m_land.zip",
    layers=(_Layer("ne_10m_land.shp", 1),),
)

GSHHG = _Source(
    id="gshhg",
    label="GSHHG 2.3.7 full resolution (L1+L5-L2+L3-L4)",
    url="https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip",
    sha256=None,  # TODO: pin after a verified download
    filename="gshhg-shp-2.3.7.zip",
    layers=(
        _Layer("GSHHS_shp/f/GSHHS_f_L1.shp", 1),
        _Layer("GSHHS_shp/f/GSHHS_f_L5.shp", 1, required=False),
        _Layer("GSHHS_shp/f/GSHHS_f_L2.shp", 0, required=False),
        _Layer("GSHHS_shp/f/GSHHS_f_L3.shp", 1, required=False),
        _Layer("GSHHS_shp/f/GSHHS_f_L4.shp", 0, required=False),
    ),
)

BUILTIN_SOURCES: dict[str, _Source] = {s.id: s for s in (NATURAL_EARTH, GSHHG)}
MASK_IDS: tuple[str, ...] = ("natural-earth-land", "gshhg", "modis-mod44w", "upload")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(source: _Source, cache_dir: Path) -> Path:
    """Return the cached archive, downloading it once. A file placed there by hand is used as is."""
    dest_dir = cache_dir / "downloads"
    dest = dest_dir / source.filename
    if not dest.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s from %s", source.label, source.url)
        fd, tmp_name = tempfile.mkstemp(dir=dest_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(source.url, timeout=120) as r:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            os.replace(tmp_name, dest)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    digest = _sha256_file(dest)
    if source.sha256 is None:
        log.warning("no pinned checksum for %s; downloaded file sha256=%s", source.id, digest)
    elif digest != source.sha256:
        raise ValueError(
            f"checksum mismatch for {dest}: expected {source.sha256}, got {digest}. "
            "Delete the file to re-download."
        )
    return dest


# --------------------------------------------------------------------------- shapefile reading


def read_shp_polygons(buf: bytes) -> Iterator[tuple[tuple[float, float, float, float], list[np.ndarray]]]:
    """Minimal ESRI .shp reader for (Multi)Polygon records: yields ``(bbox, rings)``.

    ``bbox`` is (xmin, ymin, xmax, ymax); each ring is an (N, 2) float64 array of lon/lat. Only
    the .shp geometry is needed, so no dbf/shx/fiona dependency. Shape types 5, 15 and 25
    (Polygon, PolygonZ, PolygonM) are read; Z/M values are ignored; null shapes are skipped.
    """
    if len(buf) < 100 or struct.unpack(">i", buf[:4])[0] != 9994:
        raise ValueError("not an ESRI shapefile (.shp)")
    file_len = struct.unpack(">i", buf[24:28])[0] * 2
    end = min(len(buf), file_len) if file_len > 100 else len(buf)
    pos = 100
    while pos + 8 <= end:
        _rec, words = struct.unpack(">ii", buf[pos : pos + 8])
        pos += 8
        content = buf[pos : pos + 2 * words]
        pos += 2 * words
        if len(content) < 4:
            continue
        stype = struct.unpack("<i", content[:4])[0]
        if stype not in (5, 15, 25):
            continue
        bbox = struct.unpack("<4d", content[4:36])
        n_parts, n_points = struct.unpack("<ii", content[36:44])
        parts = np.frombuffer(content, dtype="<i4", count=n_parts, offset=44)
        pts = np.frombuffer(
            content, dtype="<f8", count=2 * n_points, offset=44 + 4 * n_parts
        ).reshape(-1, 2)
        bounds = list(parts) + [n_points]
        rings = [pts[bounds[i] : bounds[i + 1]] for i in range(n_parts)]
        yield bbox, [r for r in rings if len(r) >= 3]


def rasterize_shapes(
    shapes: list[tuple[tuple[float, float, float, float], list[np.ndarray], int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Burn polygons into a global (H, W) bool mask, in row strips, in list order.

    Each shape is ``(bbox, rings, burn)``. All rings of one record are passed as one polygon;
    GDAL fills polygons with the even-odd rule, so holes and multi-part records both come out
    right. Later shapes overwrite earlier ones (burn 0 carves water back out). A pixel is set when
    its center is inside the polygon (``all_touched=False``).
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    h, w = shape
    dlon, dlat = 360.0 / w, 180.0 / h
    out = np.zeros((h, w), dtype=bool)
    for r0 in range(0, h, _STRIP_ROWS):
        r1 = min(h, r0 + _STRIP_ROWS)
        north = 90.0 - r0 * dlat
        south = 90.0 - r1 * dlat
        selected = [
            ({"type": "Polygon", "coordinates": [ring.tolist() for ring in rings]}, burn)
            for (bbox, rings, burn) in shapes
            if rings and bbox[1] <= north and bbox[3] >= south
        ]
        if not selected:
            continue
        strip = rasterize(
            selected,
            out_shape=(r1 - r0, w),
            transform=from_origin(-180.0, north, dlon, dlat),
            fill=0,
            all_touched=False,
            dtype="uint8",
        )
        out[r0:r1] = strip.astype(bool)
    return out


def _load_cached(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        h, w = (int(x) for x in z["shape"])
        bits = z["bits"]
    return np.unpackbits(bits, count=h * w).reshape(h, w).view(bool)


def _save_cached(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, bits=np.packbits(data, axis=None), shape=np.array(data.shape))
    os.replace(tmp, path)


def _res_tag(resolution_km: float) -> str:
    return f"{float(resolution_km):g}km".replace(".", "p")


def _builtin_vector_mask(source: _Source, resolution_km: float, cache_dir: Path) -> Mask:
    h, w = shape_for_resolution(resolution_km)
    label = f"{source.label}, rasterized at {resolution_km:g} km ({w}x{h})"
    cached = cache_dir / "masks" / f"{source.id}_{_res_tag(resolution_km)}_{w}x{h}.npz"
    data = _load_cached(cached)
    if data is None:
        archive = _fetch(source, cache_dir)
        shapes: list[tuple[tuple[float, float, float, float], list[np.ndarray], int]] = []
        with zipfile.ZipFile(archive) as zf:
            names = {Path(n).as_posix(): n for n in zf.namelist()}
            by_basename = {Path(n).name: n for n in zf.namelist()}
            for layer in source.layers:
                name = names.get(layer.member) or by_basename.get(Path(layer.member).name)
                if name is None:
                    if layer.required:
                        raise ValueError(f"{archive} has no member {layer.member}")
                    log.warning("%s: optional layer %s missing; skipped", source.id, layer.member)
                    continue
                for bbox, rings in read_shp_polygons(zf.read(name)):
                    shapes.append((bbox, rings, layer.burn))
        log.info("rasterizing %s (%d polygons) to %dx%d", source.id, len(shapes), w, h)
        data = rasterize_shapes(shapes, (h, w))
        _save_cached(cached, data)
    return Mask.from_array(data, label)


# --------------------------------------------------------------------------- rasters


def _read_global_raster(path: Path, out_shape: tuple[int, int], *, land_value: int) -> np.ndarray:
    """Read band 1 of a global EPSG:4326 raster, resampled (mode) to ``out_shape``, in strips."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    h, w = out_shape
    out = np.zeros((h, w), dtype=bool)
    with rasterio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic:
            raise ValueError(f"{path}: expected a geographic (EPSG:4326) raster, got {src.crs}")
        _check_global_bounds(src.bounds, path)
        h_src, w_src = src.height, src.width
        # When the source is coarser than requested, keep the source resolution.
        if h_src < h or w_src < w:
            h, w = h_src, w_src
            out = np.zeros((h, w), dtype=bool)
        rows_per = max(1, _STRIP_ROWS)
        for r0 in range(0, h, rows_per):
            r1 = min(h, r0 + rows_per)
            s0 = int(math.floor(r0 * h_src / h))
            s1 = int(math.ceil(r1 * h_src / h))
            band = src.read(
                1,
                window=Window(0, s0, w_src, s1 - s0),
                out_shape=(r1 - r0, w),
                resampling=Resampling.mode,
            )
            out[r0:r1] = band == land_value
    return out


def _check_global_bounds(bounds, path) -> None:
    left, bottom, right, top = bounds
    tol = 0.5
    if abs(left + 180) > tol or abs(right - 180) > tol or abs(bottom + 90) > tol or abs(top - 90) > tol:
        raise ValueError(
            f"{path}: raster must cover the whole globe (-180, -90, 180, 90); bounds are "
            f"({left:.3f}, {bottom:.3f}, {right:.3f}, {top:.3f})"
        )


def _modis_mask(spec: MaskSpec, cache_dir: Path) -> Mask:
    path = Path(spec.path).expanduser() if spec.path else cache_dir / "modis-mod44w" / "mod44w_global.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"MODIS MOD44W mask not found at {path}. It has no stable download URL: build a global "
            "EPSG:4326 GeoTIFF from the MOD44W tiles (see blindearth.evalspec.masks docstring) and "
            "place it there, or set mask.path."
        )
    h, w = shape_for_resolution(spec.resolution_km)
    data = _read_global_raster(path, (h, w), land_value=0)
    if spec.invert:
        np.logical_not(data, out=data)
    label = f"MODIS MOD44W ({path.name}) at {data.shape[1]}x{data.shape[0]}"
    return Mask.from_array(data, label)


# --------------------------------------------------------------------------- uploads


def _rgb_to_gray_u8(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Luminance as uint8, alpha-composited onto black, in strips (rgb uint8 (H, W, 3))."""
    h = rgb.shape[0]
    out = np.empty(rgb.shape[:2], dtype=np.uint8)
    for r0 in range(0, h, _STRIP_ROWS):
        r1 = min(h, r0 + _STRIP_ROWS)
        s = rgb[r0:r1].astype(np.float32)
        g = 0.299 * s[..., 0] + 0.587 * s[..., 1] + 0.114 * s[..., 2]
        if alpha is not None:
            g *= alpha[r0:r1].astype(np.float32) / 255.0
        out[r0:r1] = np.clip(np.rint(g), 0, 255).astype(np.uint8)
    return out


def _to_u8(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    if a.dtype == bool:
        return a.astype(np.uint8) * 255
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    f = a.astype(np.float64)
    vals = f[valid] if valid is not None else f
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.where(f > lo, 255, 0).astype(np.uint8) if lo > 0 else np.zeros(a.shape, np.uint8)
    g = np.nan_to_num((f - lo) / (hi - lo) * 255.0, nan=0.0)
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def _read_image_pil(path: Path) -> np.ndarray:
    from PIL import Image

    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _MAX_UPLOAD_PIXELS
    try:
        with Image.open(path) as img:
            img.load()
            mode = img.mode
            has_alpha = mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
                mode in ("P", "L", "RGB") and "transparency" in img.info
            )
            if has_alpha:
                rgba = np.asarray(img.convert("RGBA"))
                return _rgb_to_gray_u8(rgba[..., :3], rgba[..., 3])
            if mode == "1":
                return np.asarray(img, dtype=bool).astype(np.uint8) * 255
            if mode.startswith("I;16"):
                return (np.asarray(img).astype(np.uint16) >> 8).astype(np.uint8)
            if mode in ("I", "F"):
                return _to_u8(np.asarray(img))
            if mode == "L":
                return np.asarray(img, dtype=np.uint8)
            rgb = np.asarray(img.convert("RGB"))
            return _rgb_to_gray_u8(rgb, None)
    finally:
        Image.MAX_IMAGE_PIXELS = old


def _read_geotiff(path: Path, projection: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import ColorInterp

    with rasterio.open(path) as src:
        crs = src.crs
        if crs is not None:
            epsg = crs.to_epsg()
            if projection == "equirectangular":
                if not crs.is_geographic:
                    raise ValueError(
                        f"{path}: declared equirectangular but the GeoTIFF CRS is {crs}; "
                        "declare web_mercator or reproject to EPSG:4326"
                    )
                _check_global_bounds(src.bounds, path)
            elif projection == "web_mercator" and epsg not in (3857, 3785, 900913, 102100):
                raise ValueError(f"{path}: declared web_mercator but the GeoTIFF CRS is {crs}")
        interp = list(src.colorinterp)
        alpha_idx = [i + 1 for i, c in enumerate(interp) if c == ColorInterp.alpha]
        color_idx = [i + 1 for i in range(src.count) if (i + 1) not in alpha_idx]
        valid = src.dataset_mask() > 0  # nodata / alpha = transparent
        if len(color_idx) >= 3:
            rgb = np.stack([_to_u8(src.read(b), valid) for b in color_idx[:3]], axis=-1)
            alpha = src.read(alpha_idx[0]) if alpha_idx else None
            gray = _rgb_to_gray_u8(rgb, _to_u8(alpha) if alpha is not None else None)
        else:
            gray = _to_u8(src.read(color_idx[0] if color_idx else 1), valid)
        gray[~valid] = 0  # transparent / nodata flattened onto black
    return gray


def mercator_to_equirect(img: np.ndarray, out_width: int | None = None) -> np.ndarray:
    """Reproject a full-extent Web Mercator image (square, +-85.0511 deg) to equirectangular 2:1.

    Rows are mapped by nearest neighbour on the Mercator y of each output row's center latitude;
    columns are linear in longitude in both projections. Latitudes beyond +-85.0511 deg copy the
    nearest edge row (Arctic water, Antarctic land).
    """
    h_in, w_in = img.shape[:2]
    if abs(h_in - w_in) > 1:
        raise ValueError(
            f"web_mercator upload must be the full square extent; got {w_in}x{h_in} (W x H)"
        )
    w = int(out_width or w_in)
    w += w % 2
    h = w // 2
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lat = np.clip(lat, -MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
    y = np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))  # in [-pi, pi]
    rows = np.clip(np.floor((np.pi - y) / (2.0 * np.pi) * h_in), 0, h_in - 1).astype(np.int64)
    cols = np.clip(((np.arange(w) + 0.5) * w_in / w).astype(np.int64), 0, w_in - 1)
    return img[np.ix_(rows, cols)]


def _otsu_u8(gray: np.ndarray) -> float:
    """Otsu threshold on a uint8 image (skimage), using a strided sample for large images."""
    from skimage.filters import threshold_otsu

    n = gray.size
    stride = max(1, int(math.sqrt(n / 4_000_000)))
    sample = gray[::stride, ::stride]
    hist = np.bincount(sample.ravel(), minlength=256)
    occupied = np.flatnonzero(hist)
    if occupied.size < 2:
        raise ValueError(
            "uploaded image is a single uniform value; cannot binarize it (Otsu needs two levels)"
        )
    # Trim empty bins at both ends: skimage divides by the cumulative counts, and an empty
    # leading bin would give NaN class means (and a threshold of 0).
    lo, hi = int(occupied[0]), int(occupied[-1])
    return float(threshold_otsu(hist=(hist[lo : hi + 1], np.arange(lo, hi + 1))))


def _preview(land: np.ndarray) -> np.ndarray:
    step = max(1, int(math.ceil(land.shape[1] / _PREVIEW_MAX_WIDTH)))
    small = land[::step, ::step]
    rgb = np.zeros(small.shape + (3,), dtype=np.uint8)
    rgb[small] = 255
    return rgb


def prepare_upload(
    path: str | Path,
    *,
    projection: str,
    invert: bool,
    threshold: float | None,
) -> tuple[Mask, np.ndarray, float]:
    """Validate and binarize an uploaded ground-truth image.

    - Reads PNG/JPEG/... with Pillow and ``.tif``/``.tiff`` with rasterio. Colour images become
      luminance; transparent pixels (alpha, GeoTIFF nodata) are flattened onto black (water).
    - ``projection="equirectangular"``: the image must be 2:1 (+-1 pixel). For GeoTIFFs with a CRS
      it must be geographic and cover the globe. ``"web_mercator"``: the image must be the full
      square Web Mercator extent; it is reprojected to equirectangular 2:1 at the same width.
    - ``threshold`` is in [0, 1] of the 8-bit gray range (pixel > threshold*255 is land);
      ``None`` = Otsu. ``invert=True`` flips polarity (for white = water images).

    Returns ``(mask, preview_rgb_uint8, threshold_used)``. The preview is white land on black,
    decimated to at most 2048 px wide; ``threshold_used`` is in [0, 1].
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"mask upload not found: {p}")
    if projection not in ("equirectangular", "web_mercator"):
        raise ValueError(f"unknown projection {projection!r} (use equirectangular or web_mercator)")

    if p.suffix.lower() in (".tif", ".tiff"):
        gray = _read_geotiff(p, projection)
    else:
        gray = _read_image_pil(p)
    if gray.ndim != 2:
        raise ValueError(f"{p}: could not read a 2-D image (got shape {gray.shape})")

    if projection == "web_mercator":
        gray = mercator_to_equirect(gray)
    else:
        h, w = gray.shape
        if abs(w - 2 * h) > 1:
            raise ValueError(
                f"{p}: equirectangular masks must have a 2:1 aspect ratio; got {w}x{h} (W x H). "
                "If it is Web Mercator, declare projection: web_mercator."
            )

    if threshold is None:
        t_u8 = _otsu_u8(gray)
        threshold_used = t_u8 / 255.0
    else:
        threshold_used = float(threshold)
        if not (0.0 <= threshold_used <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        t_u8 = threshold_used * 255.0

    land = gray > t_u8
    if invert:
        np.logical_not(land, out=land)
    del gray

    how = "otsu" if threshold is None else "fixed"
    label = (
        f"upload:{p.name} ({projection}, {how} threshold {threshold_used:.3f}"
        f"{', inverted' if invert else ''}, {land.shape[1]}x{land.shape[0]})"
    )
    mask = Mask.from_array(land, label)
    return mask, _preview(land), threshold_used


# --------------------------------------------------------------------------- entry point


def load_mask(spec: MaskSpec, cache_dir: Path | None = None) -> Mask:
    """Load or build the mask named by ``spec``. Built-ins download/rasterize once and are cached."""
    cache = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    if spec.id in BUILTIN_SOURCES:
        mask = _builtin_vector_mask(BUILTIN_SOURCES[spec.id], spec.resolution_km, cache)
        if spec.invert:
            mask = Mask.from_array(~mask.data, mask.source + ", inverted")
        return mask
    if spec.id == "modis-mod44w":
        return _modis_mask(spec, cache)
    if spec.id == "upload":
        if not spec.path:
            raise ValueError("mask id 'upload' needs mask.path")
        mask, _preview_rgb, _t = prepare_upload(
            spec.path, projection=spec.projection, invert=spec.invert, threshold=spec.threshold
        )
        h, w = shape_for_resolution(spec.resolution_km)
        if mask.shape[0] > h:  # finer than requested: downsample with care (area majority)
            data = resample_mask(mask.data, (h, w))
            mask = Mask.from_array(data, f"{mask.source}, downsampled to {w}x{h}")
        return mask
    raise ValueError(f"unknown mask id {spec.id!r} (use one of {', '.join(MASK_IDS)})")


__all__ = [
    "Mask",
    "MASK_IDS",
    "load_mask",
    "prepare_upload",
    "mask_hash",
    "resample_mask",
    "shape_for_resolution",
    "mercator_to_equirect",
    "default_cache_dir",
    "read_shp_polygons",
    "rasterize_shapes",
]
