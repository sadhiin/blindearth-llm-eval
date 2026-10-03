"""Ground-truth land masks.

A mask is a ``bool`` array of shape (H, W), equirectangular, row 0 = 90 N, column 0 = 180 W,
True = land. Pixel ``(i, j)`` covers lat ``[90 - (i+1)*180/H, 90 - i*180/H]`` and lon
``[-180 + j*360/W, -180 + (j+1)*360/W]``.

Built-in sources (``MaskSpec.id``):

- ``natural-earth-land``: Natural Earth 1:10m land polygons, downloaded on first use from a fixed
  URL into the cache dir (``~/.cache/blindearth`` or ``$BLINDEARTH_CACHE``) and rasterized at
  ``MaskSpec.resolution_km`` (default 1 km, about 40076 x 20038 pixels, ~800 MB as bool, ~100 MB
  packed on disk). Rasterization runs in horizontal strips so peak memory stays near the size of
  the output array.
- ``gshhg``: GSHHG 2.3.7 full-resolution shorelines. Land = L1 (continents/islands) + L5
  (Antarctica ice front), minus L2 (lakes), plus L3 (islands in lakes), minus L4 (ponds).
- ``modis-mod44w``: MODIS MOD44W v061 water mask (``Water_mask`` layer: 0 = land, 1 = water,
  250 = fill; fill and nodata count as water). Two ways to get it:

  * Manual (always works): build a global EPSG:4326 GeoTIFF from the tiles, e.g.
    ``gdalwarp -t_srs EPSG:4326 -te -180 -90 180 90 -tr 0.0025 0.0025 -r mode <subdatasets>
    mod44w_global.tif`` (``gdalinfo <tile>.hdf`` lists the exact ``HDF4_EOS:EOS_GRID:...``
    subdataset names), then put it at ``<cache_dir>/modis-mod44w/mod44w_global.tif`` or set
    ``MaskSpec.path``.
  * Automatic (opt-in): set ``BLINDEARTH_MODIS_DOWNLOAD=1``. The tiles of one year
    (``BLINDEARTH_MODIS_YEAR``, default 2021) are found with a NASA CMR granule search, downloaded
    from LP DAAC Earthdata Cloud with an Earthdata Login (``EARTHDATA_TOKEN`` bearer token, else
    the ``urs.earthdata.nasa.gov`` entry of ``~/.netrc`` / ``$NETRC``), and reprojected straight
    to the requested grid with rasterio (mode resampling). That needs a GDAL build with the HDF4
    driver; it is checked before anything is downloaded. See :func:`build_mod44w_mask`.

- ``upload``: a user image (PNG, JPEG, or GeoTIFF) at ``MaskSpec.path``, prepared by
  :func:`prepare_upload` and downsampled to ``resolution_km`` only when it is finer than that.

Download integrity
------------------
Every download goes to a temp file in the target directory and is renamed into place only after
it passes the checks, so a truncated file or an HTML error page is never cached. The checks:
not ``text/html``, ``Content-Length`` matches, a minimum size, the upstream size / MD5 when one is
recorded in the source (see the ``_Source`` constants), and a format check (zip: CRC test of all
members plus required layers present; HDF4: magic bytes).

Neither Natural Earth nor GSHHG publishes a sha256 for its archive, so checksums use
trust-on-first-use: the sha256 of the first good download is written to
``<cache_dir>/checksums.json`` and each later read of that file is checked against it. A sha256
pinned in the source (``_Source.sha256``) takes precedence over the lockfile and can't be
overridden. On a TOFU mismatch, delete the file (and re-download), or set
``BLINDEARTH_ACCEPT_NEW_CHECKSUMS=1`` once to record the new hash (this also drops stale
rasterized caches of that source). The rasterized ``masks/*.npz`` cache is not re-checked against
the archive. Either way the mask's own content hash (``Mask.hash``, unchanged by any of this)
goes into the spec hash, so a changed download can never be mixed silently with old runs.

Sources checked on 2026-10-03 (docs, plus HTTP HEAD requests only, no data downloaded):

- Natural Earth page https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/
  (version 5.1.1, no checksum published). A HEAD on the naciscdn.org URL gave
  ``content-length: 3269070`` and S3 ``etag: "be3001f37196d2894e17aacd13ff2cc2"``. For a
  single-part S3 upload the ETag is the MD5 of the object; that it is single-part is an
  assumption (there is no ``-N`` suffix).
- GSHHG home https://www.soest.hawaii.edu/pwessel/gshhg/ (2.3.7, no checksum published). The old
  NOAA URL (``ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/...``) now returns a 404 HTML page,
  so the SOEST URL is used. HEAD: ``Content-Length: 149157845`` (Apache ETag, not an MD5).
- MOD44W v061: CMR collection ``C2565805847-LPCLOUD`` (short_name MOD44W, version 061,
  provider LPCLOUD), https://cmr.earthdata.nasa.gov/search/collections.json?short_name=MOD44W;
  layers and values from https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod44w-061; one
  granule per sinusoidal tile per year (318 for 2021), named like
  ``MOD44W.A2021001.h01v09.061.2024008090424``, with "GET DATA" URLs on
  ``https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD44W.061/...hdf``. Granule
  metadata gives no checksum. CMR paging uses the ``CMR-Search-After`` header with page_size up to
  2000 (https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html).
- Earthdata Login: the netrc/basic-auth + cookie flow follows
  https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python. User tokens last 60
  days (https://urs.earthdata.nasa.gov/documentation/for_users/user_token). Sending them as
  ``Authorization: Bearer`` to LP DAAC is shown in LP DAAC forum answers
  (https://forum.earthdata.nasa.gov/viewtopic.php?p=15142).

Not verified here, because no data was downloaded: the HDF subdataset name (it is matched by a
``:Water_mask`` suffix, case-insensitive), that the token flow is accepted by
``data.lpdaac.earthdatacloud.nasa.gov`` with redirects to S3 (the token is sent only to
Earthdata hosts, never to the S3 redirect target), that GDAL reads the tile CRS from the HDF-EOS
metadata (the MODIS sinusoidal CRS is used as a fallback), and how long a full mosaic takes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import netrc
import os
import re
import struct
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Callable, Iterator

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
    # sha256 pinned from a trusted download. It wins over the TOFU lockfile and can't be
    # overridden. None = no upstream sha256 exists, so TOFU via <cache_dir>/checksums.json.
    sha256: str | None
    filename: str
    layers: tuple[_Layer, ...]
    md5: str | None = None  # upstream-advertised MD5 (e.g. S3 ETag), checked on fresh downloads
    size: int | None = None  # upstream-advertised byte size, checked on fresh downloads
    min_size: int = 1  # anything smaller is a truncated file or an error page


NATURAL_EARTH = _Source(
    id="natural-earth-land",
    label="Natural Earth 1:10m land v5.1.1",
    url="https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
    sha256=None,  # upstream publishes no sha256: TOFU (see module docstring)
    filename="ne_10m_land.zip",
    layers=(_Layer("ne_10m_land.shp", 1),),
    md5="be3001f37196d2894e17aacd13ff2cc2",  # S3 ETag from a HEAD on 2026-10-03
    size=3_269_070,  # Content-Length from the same HEAD
    min_size=1_000_000,
)

GSHHG = _Source(
    id="gshhg",
    label="GSHHG 2.3.7 full resolution (L1+L5-L2+L3-L4)",
    # The NOAA ngdc.noaa.gov/.../gshhg/latest/ URL returns 404 (checked 2026-10-03).
    url="https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip",
    sha256=None,  # upstream publishes no checksum: TOFU (see module docstring)
    filename="gshhg-shp-2.3.7.zip",
    size=149_157_845,  # Content-Length from a HEAD on 2026-10-03
    min_size=50_000_000,
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


LOCKFILE_NAME = "checksums.json"
ACCEPT_ENV = "BLINDEARTH_ACCEPT_NEW_CHECKSUMS"


class DownloadIntegrityError(ValueError):
    """A download (or a file in the cache) is truncated, an error page, or not the expected format."""


class ChecksumMismatchError(ValueError):
    """A cached file's sha256 differs from the pinned value or the one recorded on first use."""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _accept_new() -> bool:
    return _env_flag(ACCEPT_ENV)


def _lock_path(cache_dir: Path) -> Path:
    return cache_dir / LOCKFILE_NAME


def _lock_read(cache_dir: Path) -> dict[str, dict[str, Any]]:
    p = _lock_path(cache_dir)
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(
            f"checksum lockfile {p} is unreadable ({e}); fix or delete it (deleting it means the "
            "next load trusts whatever is in the cache)"
        ) from e
    files = doc.get("files") if isinstance(doc, dict) else None
    return files if isinstance(files, dict) else {}


def _lock_write_entry(cache_dir: Path, key: str, entry: dict[str, Any]) -> None:
    """Add or replace one entry, rewriting the lockfile atomically (temp file + rename)."""
    files = _lock_read(cache_dir)
    files[key] = entry
    p = _lock_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "files": files}, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _verify_or_record(
    cache_dir: Path,
    path: Path,
    *,
    key: str,
    url: str,
    pinned: str | None,
    validate: Callable[[Path], None] | None = None,
    on_replace: Callable[[], None] | None = None,
) -> str:
    """Check ``path`` against its pinned or recorded sha256, recording it on first use.

    Precedence: a pinned sha256 always wins and a mismatch is fatal. Without a pin, the lockfile
    entry is used. With no entry yet (first download, or a file placed by hand), the file is
    validated and its hash recorded. A lockfile mismatch is fatal unless ``ACCEPT_ENV`` is set, in
    which case the new hash replaces the old one and ``on_replace`` runs (to drop derived caches).
    """
    digest = _sha256_file(path)
    entry_new = {
        "sha256": digest,
        "size": path.stat().st_size,
        "url": url,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if pinned is not None:
        if digest != pinned.lower():
            raise ChecksumMismatchError(
                f"checksum mismatch for {path}: pinned sha256 {pinned}, file has {digest}. The "
                f"pin is part of blindearth and can't be overridden. Delete {path} to re-download; "
                "if upstream really changed the file, the pin in blindearth/evalspec/masks.py "
                "needs updating."
            )
        entry = _lock_read(cache_dir).get(key)
        if not entry or entry.get("sha256") != digest:
            _lock_write_entry(cache_dir, key, {**entry_new, "pinned": True})
        return digest

    entry = _lock_read(cache_dir).get(key)
    if entry is None:
        if validate is not None:
            validate(path)
        log.warning(
            "recording sha256 %s for %s in %s (trust on first use)",
            digest, path.name, _lock_path(cache_dir),
        )
        _lock_write_entry(cache_dir, key, entry_new)
        return digest
    if entry.get("sha256") == digest:
        return digest
    if _accept_new():
        if validate is not None:
            validate(path)
        log.warning(
            "%s: sha256 changed from %s to %s; accepted because %s is set",
            path, entry.get("sha256"), digest, ACCEPT_ENV,
        )
        _lock_write_entry(cache_dir, key, entry_new)
        if on_replace is not None:
            on_replace()
        return digest
    raise ChecksumMismatchError(
        f"checksum mismatch for {path}: {_lock_path(cache_dir)} recorded sha256 "
        f"{entry.get('sha256')} on {entry.get('recorded_at', '?')}, but the file now has {digest}. "
        f"Either the cached file was corrupted or replaced (delete {path} to download it again), "
        f"or upstream published a new version (set {ACCEPT_ENV}=1 once to accept and record the "
        "new hash). Masks built from different data get different mask hashes, so old and new "
        "runs won't be mixed."
    )


def _urlopen(url: str, *, opener: urllib.request.OpenerDirector | None = None,
             headers: dict[str, str] | None = None, timeout: float = 120):
    """The one network entry point (tests monkeypatch it)."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    if opener is None:
        return urllib.request.urlopen(req, timeout=timeout)
    return opener.open(req, timeout=timeout)


def _download(
    url: str,
    dest: Path,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    min_size: int = 1,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    validate: Callable[[Path], None] | None = None,
) -> None:
    """Download ``url`` to ``dest`` atomically: temp file in the same dir, checks, then rename.

    Upstream size/MD5 mismatches are fatal unless ``ACCEPT_ENV`` is set (upstream may publish
    a new version); the format check (``validate``) and the minimum size can't be skipped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        md5 = hashlib.md5(usedforsecurity=False)
        n = 0
        with os.fdopen(fd, "wb") as out, _urlopen(url, opener=opener) as r:
            headers = getattr(r, "headers", None) or {}
            ctype = (headers.get("Content-Type") or "").lower()
            if ctype.startswith("text/html"):
                final = r.geturl() if hasattr(r, "geturl") else url
                raise DownloadIntegrityError(
                    f"{url} returned an HTML page (final URL {final}) instead of data; it is "
                    "probably an error or login page"
                )
            clen = headers.get("Content-Length")
            while chunk := r.read(1 << 20):
                out.write(chunk)
                md5.update(chunk)
                n += len(chunk)
        if clen is not None and str(clen).isdigit() and int(clen) != n:
            raise DownloadIntegrityError(
                f"{url}: truncated download ({n} of {clen} bytes)"
            )
        if n < min_size:
            raise DownloadIntegrityError(
                f"{url}: download is only {n} bytes (expected at least {min_size}); probably an "
                "error page or truncated"
            )
        problems = []
        if expected_size is not None and n != expected_size:
            problems.append(f"size {n} != expected {expected_size}")
        if expected_md5 is not None and md5.hexdigest() != expected_md5.lower():
            problems.append(f"md5 {md5.hexdigest()} != expected {expected_md5}")
        if problems:
            msg = f"{url}: {'; '.join(problems)} (values recorded from upstream)"
            if not _accept_new():
                raise DownloadIntegrityError(
                    f"{msg}. If upstream released a new version, set {ACCEPT_ENV}=1 to accept it."
                )
            log.warning("%s; accepted because %s is set", msg, ACCEPT_ENV)
        if validate is not None:
            validate(tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _find_member(names: list[str], member: str) -> str | None:
    by_path = {Path(n).as_posix(): n for n in names}
    by_base = {Path(n).name: n for n in names}
    return by_path.get(member) or by_base.get(Path(member).name)


def _validate_zip(path: Path, source: _Source) -> None:
    """The archive must be a zip whose members all pass their CRC check and that has every
    required layer (``.shp``)."""
    if path.stat().st_size < source.min_size:
        raise DownloadIntegrityError(
            f"{path} is {path.stat().st_size} bytes; expected at least {source.min_size}"
        )
    if not zipfile.is_zipfile(path):
        raise DownloadIntegrityError(f"{path} is not a zip archive")
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        raise DownloadIntegrityError(f"{path} is a corrupt zip archive: {e}") from e
    if bad is not None:
        raise DownloadIntegrityError(f"{path}: member {bad} fails its CRC check")
    missing = [l.member for l in source.layers if l.required and _find_member(names, l.member) is None]
    if missing:
        raise DownloadIntegrityError(f"{path} is missing required members: {', '.join(missing)}")


def _drop_rasterized(source_id: str, cache_dir: Path) -> None:
    for p in (cache_dir / "masks").glob(f"{source_id}_*.npz"):
        log.warning("removing %s (built from the previous archive)", p)
        p.unlink(missing_ok=True)


def _fetch(source: _Source, cache_dir: Path) -> Path:
    """Return the cached archive, downloading it once, verified against its pinned or recorded
    sha256. A file placed there by hand is validated and recorded on first use."""
    dest = cache_dir / "downloads" / source.filename
    validate = lambda p: _validate_zip(p, source)  # noqa: E731
    if not dest.exists():
        log.info("downloading %s from %s", source.label, source.url)
        _download(
            source.url, dest, min_size=source.min_size, expected_size=source.size,
            expected_md5=source.md5, validate=validate,
        )
    _verify_or_record(
        cache_dir, dest, key=f"downloads/{source.filename}", url=source.url, pinned=source.sha256,
        validate=validate, on_replace=lambda: _drop_rasterized(source.id, cache_dir),
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


# ---- MODIS MOD44W v061 automatic download (CMR search + LP DAAC tiles + rasterio mosaic)

MODIS_DOWNLOAD_ENV = "BLINDEARTH_MODIS_DOWNLOAD"
MODIS_YEAR_ENV = "BLINDEARTH_MODIS_YEAR"
MODIS_DEFAULT_YEAR = 2021  # 318 v061 granules listed in CMR (checked 2026-10-03)
EARTHDATA_TOKEN_ENV = "EARTHDATA_TOKEN"
EARTHDATA_URS_HOST = "urs.earthdata.nasa.gov"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
MOD44W_SHORT_NAME = "MOD44W"
MOD44W_VERSION = "061"
MOD44W_PROVIDER = "LPCLOUD"
_MOD44W_FILL = 250
_HDF4_MAGIC = b"\x0e\x03\x13\x01"
# MODIS sinusoidal grid: sphere radius and tile edge (36 x 18 tiles of 10 deg at the equator).
MODIS_SPHERE_R = 6371007.181
MODIS_TILE_M = 1111950.5197665233
MODIS_SINU_CRS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
_TILE_RE = re.compile(r"\.h(\d{2})v(\d{2})\.")


class EarthdataAuthError(RuntimeError):
    """No usable Earthdata Login credentials, or the server rejected them."""


def _is_earthdata_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host == EARTHDATA_URS_HOST or host.endswith(".earthdata.nasa.gov") or host.endswith(
        ".earthdatacloud.nasa.gov"
    )


class _EarthdataBearerHandler(urllib.request.BaseHandler):
    """Adds ``Authorization: Bearer <token>`` to HTTPS requests to Earthdata hosts only.

    It runs for every request, including redirects, and uses an unredirected header, so the token
    is never sent to the presigned S3 URL the data server redirects to (S3 would reject a second
    auth method, and the token must not leak to other hosts).
    """

    handler_order = 400

    def __init__(self, token: str):
        self.token = token

    def https_request(self, req: urllib.request.Request) -> urllib.request.Request:
        host = urllib.parse.urlsplit(req.full_url).hostname or ""
        req.headers.pop("Authorization", None)
        req.unredirected_hdrs.pop("Authorization", None)
        if _is_earthdata_host(host):
            req.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return req

    def http_request(self, req: urllib.request.Request) -> urllib.request.Request:
        req.headers.pop("Authorization", None)  # never send a token over plain HTTP
        req.unredirected_hdrs.pop("Authorization", None)
        return req


def _netrc_credentials() -> tuple[str, str] | None:
    path = Path(os.environ.get("NETRC") or "~/.netrc").expanduser()
    if not path.exists():
        return None
    try:
        auth = netrc.netrc(str(path)).authenticators(EARTHDATA_URS_HOST)
    except (netrc.NetrcParseError, OSError) as e:
        raise EarthdataAuthError(f"could not parse {path}: {e}") from e
    if not auth or not auth[0] or not auth[2]:
        return None
    return auth[0], auth[2]


def _earthdata_opener() -> tuple[urllib.request.OpenerDirector, str]:
    """An opener authenticated for Earthdata Login: bearer token first, then ~/.netrc."""
    cookies = urllib.request.HTTPCookieProcessor(CookieJar())
    token = os.environ.get(EARTHDATA_TOKEN_ENV, "").strip()
    if token:
        return urllib.request.build_opener(_EarthdataBearerHandler(token), cookies), "token"
    creds = _netrc_credentials()
    if creds is not None:
        # NASA's documented urllib flow: answer the URS 401 with basic auth, keep session cookies.
        pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pm.add_password(None, f"https://{EARTHDATA_URS_HOST}", creds[0], creds[1])
        return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(pm), cookies), "netrc"
    raise EarthdataAuthError(
        "MODIS MOD44W download needs NASA Earthdata Login credentials. Create an account at "
        f"https://{EARTHDATA_URS_HOST}, then either set {EARTHDATA_TOKEN_ENV} to a user token "
        "(profile > Generate Token; tokens expire after 60 days) or add\n"
        f"    machine {EARTHDATA_URS_HOST} login <user> password <password>\n"
        "to ~/.netrc (chmod 600) or the file named by $NETRC."
    )


@dataclass(frozen=True)
class _Granule:
    name: str  # GranuleUR, e.g. MOD44W.A2021001.h01v09.061.2024008090424
    h: int
    v: int
    url: str


def _cmr_mod44w_granules(year: int) -> list[_Granule]:
    """All MOD44W v061 tiles for ``year`` (one per h/v; the latest production wins)."""
    params = {
        "short_name": MOD44W_SHORT_NAME,
        "version": MOD44W_VERSION,
        "provider": MOD44W_PROVIDER,
        "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
        "page_size": "2000",
    }
    url = f"{CMR_GRANULES_URL}?{urllib.parse.urlencode(params)}"
    prefix = f"{MOD44W_SHORT_NAME}.A{year}001."
    best: dict[tuple[int, int], _Granule] = {}
    search_after: str | None = None
    for _page in range(100):  # 318 granules fit in one page; the cap only guards a loop
        headers = {"CMR-Search-After": search_after} if search_after else None
        with _urlopen(url, headers=headers, timeout=60) as r:
            doc = json.loads(r.read().decode("utf-8"))
            search_after = (getattr(r, "headers", None) or {}).get("CMR-Search-After")
        items = doc.get("items") or []
        for item in items:
            umm = item.get("umm") or {}
            name = umm.get("GranuleUR") or ""
            m = _TILE_RE.search(name)
            if not name.startswith(prefix) or m is None:
                continue
            data_url = next(
                (
                    u.get("URL")
                    for u in umm.get("RelatedUrls") or []
                    if u.get("Type") == "GET DATA"
                    and str(u.get("URL", "")).startswith("https://")
                    and str(u.get("URL", "")).lower().endswith(".hdf")
                ),
                None,
            )
            if data_url is None:
                continue
            g = _Granule(name, int(m.group(1)), int(m.group(2)), data_url)
            prev = best.get((g.h, g.v))
            if prev is None or g.name > prev.name:  # names end in the production timestamp
                best[(g.h, g.v)] = g
        if not items or not search_after:
            break
    if not best:
        raise ValueError(
            f"CMR returned no {MOD44W_SHORT_NAME} v{MOD44W_VERSION} granules for {year} ({url})"
        )
    return sorted(best.values(), key=lambda g: (g.v, g.h))


def _validate_hdf4(path: Path) -> None:
    with open(path, "rb") as f:
        head = f.read(4)
    if head != _HDF4_MAGIC:
        raise DownloadIntegrityError(f"{path} is not an HDF4 file (got {head!r})")


def modis_tile_bounds(h: int, v: int) -> tuple[float, float, float, float] | None:
    """(west, south, east, north) in degrees enclosing sinusoidal tile h/v, or None when the
    tile lies entirely outside the globe. Tile v spans exactly lat [80 - 10v, 90 - 10v]."""
    north = 90.0 - 10.0 * v
    south = north - 10.0
    if north >= 90.0 or south <= -90.0:
        return (-180.0, south, 180.0, north)  # touches a pole: any longitude
    x0 = (h - 18) * MODIS_TILE_M
    x1 = x0 + MODIS_TILE_M
    lons = [
        math.degrees(x / (MODIS_SPHERE_R * math.cos(math.radians(lat))))
        for x in (x0, x1)
        for lat in (south, north)
    ]
    west, east = max(-180.0, min(lons)), min(180.0, max(lons))
    if west >= 180.0 or east <= -180.0 or west >= east:
        return None
    return (west, south, east, north)


def _require_hdf4_driver() -> None:
    import rasterio

    with rasterio.Env() as env:
        drivers = env.drivers()
    if "HDF4" not in drivers:
        raise RuntimeError(
            "the GDAL build used by rasterio has no HDF4 driver, so MOD44W .hdf tiles can't be "
            "read (the PyPI rasterio wheels usually lack it). Install rasterio/GDAL from "
            "conda-forge, or build the GeoTIFF manually (see blindearth.evalspec.masks docstring)."
        )


def _read_mod44w_tile(path: Path) -> tuple[np.ndarray, Any, Any]:
    """(Water_mask band, affine transform, CRS) of one MOD44W HDF tile."""
    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(path) as ds:
        subs = list(ds.subdatasets)
    name = next((s for s in subs if s.lower().endswith(":water_mask")), None)
    if name is None:
        raise ValueError(f"{path}: no Water_mask subdataset (found {subs})")
    with rasterio.open(name) as src:
        return src.read(1), src.transform, src.crs or CRS.from_string(MODIS_SINU_CRS)


def _burn_modis_tile(out: np.ndarray, h: int, v: int, band: np.ndarray, transform, crs) -> None:
    """Reproject one tile (0 = land, 1 = water, 250 = fill) into the global bool grid ``out``.
    Only the tile's own footprint is touched, so neighbouring tiles never overwrite each other."""
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    bounds = modis_tile_bounds(h, v)
    if bounds is None:
        return
    west, south, east, north = bounds
    H, W = out.shape
    dlon, dlat = 360.0 / W, 180.0 / H
    r0 = max(0, int(math.floor((90.0 - north) / dlat)) - 1)
    r1 = min(H, int(math.ceil((90.0 - south) / dlat)) + 1)
    c0 = max(0, int(math.floor((west + 180.0) / dlon)) - 1)
    c1 = min(W, int(math.ceil((east + 180.0) / dlon)) + 1)
    if r1 <= r0 or c1 <= c0:
        return
    dst = np.full((r1 - r0, c1 - c0), 255, dtype=np.uint8)
    reproject(
        source=np.ascontiguousarray(band, dtype=np.uint8),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        src_nodata=_MOD44W_FILL,
        dst_transform=from_origin(-180.0 + c0 * dlon, 90.0 - r0 * dlat, dlon, dlat),
        dst_crs="EPSG:4326",
        dst_nodata=255,
        resampling=Resampling.mode,
    )
    out[r0:r1, c0:c1] |= dst == 0


def build_mod44w_mask(
    out_shape: tuple[int, int], cache_dir: Path, year: int = MODIS_DEFAULT_YEAR
) -> np.ndarray:
    """Download (once) the MOD44W v061 tiles of ``year`` and mosaic them to a bool land grid.

    Tiles are cached under ``<cache_dir>/downloads/modis-mod44w/<year>/`` and verified like the
    other downloads (atomic write, HDF4 magic check, TOFU sha256 in ``checksums.json``). Tiles
    absent from the product (open ocean) stay water. Needs rasterio with GDAL's HDF4 driver and
    Earthdata Login credentials (see :func:`_earthdata_opener`).
    """
    _require_hdf4_driver()
    opener, how = _earthdata_opener()
    granules = _cmr_mod44w_granules(year)
    log.info("MOD44W %s: %d tiles (auth: %s)", year, len(granules), how)
    tile_dir = cache_dir / "downloads" / "modis-mod44w" / str(year)
    out = np.zeros(out_shape, dtype=bool)
    for i, g in enumerate(granules, 1):
        fname = g.url.rsplit("/", 1)[-1]
        dest = tile_dir / fname
        if not dest.exists():
            log.info("MOD44W tile %d/%d: %s", i, len(granules), fname)
            try:
                _download(g.url, dest, opener=opener, min_size=1024, validate=_validate_hdf4)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise EarthdataAuthError(
                        f"Earthdata rejected the {how} credentials for {g.url} (HTTP {e.code}). "
                        "Check that the token hasn't expired, and that the account can download "
                        "LP DAAC data."
                    ) from e
                raise
            except DownloadIntegrityError as e:
                if "HTML page" in str(e):
                    raise EarthdataAuthError(
                        f"{e}. This is usually the Earthdata login page: the {how} credentials "
                        "were not accepted."
                    ) from e
                raise
        _verify_or_record(
            cache_dir, dest, key=f"downloads/modis-mod44w/{year}/{fname}", url=g.url,
            pinned=None, validate=_validate_hdf4,
        )
        band, transform, crs = _read_mod44w_tile(dest)
        _burn_modis_tile(out, g.h, g.v, band, transform, crs)
    return out


def _modis_year() -> int:
    raw = os.environ.get(MODIS_YEAR_ENV, "").strip()
    if not raw:
        return MODIS_DEFAULT_YEAR
    try:
        year = int(raw)
    except ValueError:
        raise ValueError(f"{MODIS_YEAR_ENV} must be a year like 2021, got {raw!r}") from None
    if not 2000 <= year <= 2100:
        raise ValueError(f"{MODIS_YEAR_ENV} must be a year like 2021, got {raw!r}")
    return year


def _modis_mask(spec: MaskSpec, cache_dir: Path) -> Mask:
    h, w = shape_for_resolution(spec.resolution_km)
    manual = Path(spec.path).expanduser() if spec.path else cache_dir / "modis-mod44w" / "mod44w_global.tif"
    if spec.path or manual.exists():
        if not manual.exists():
            raise FileNotFoundError(f"MODIS MOD44W mask not found at {manual} (mask.path)")
        data = _read_global_raster(manual, (h, w), land_value=0)
        label = f"MODIS MOD44W ({manual.name}) at {data.shape[1]}x{data.shape[0]}"
    else:
        year = _modis_year()
        cached = (
            cache_dir / "masks"
            / f"modis-mod44w-v{MOD44W_VERSION}-{year}_{_res_tag(spec.resolution_km)}_{w}x{h}.npz"
        )
        data = _load_cached(cached)
        if data is None:
            if not _env_flag(MODIS_DOWNLOAD_ENV):
                raise FileNotFoundError(
                    f"MODIS MOD44W mask not found at {manual}. Either set {MODIS_DOWNLOAD_ENV}=1 "
                    f"(with {EARTHDATA_TOKEN_ENV} or a ~/.netrc Earthdata entry) to download and "
                    "mosaic the tiles automatically, or build a global EPSG:4326 GeoTIFF by hand "
                    "(see blindearth.evalspec.masks docstring) and put it there or set mask.path."
                )
            data = build_mod44w_mask((h, w), cache_dir, year)
            _save_cached(cached, data)
        label = f"MODIS MOD44W v{MOD44W_VERSION} {year} (LP DAAC tiles) at {w}x{h}"
    if spec.invert:
        data = ~data
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
    "build_mod44w_mask",
    "modis_tile_bounds",
    "ChecksumMismatchError",
    "DownloadIntegrityError",
    "EarthdataAuthError",
]
