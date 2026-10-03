"""Mask downloads: integrity checks, TOFU checksums, MODIS/Earthdata plumbing. No real network."""

import hashlib
import io
import json
import math
import struct
import urllib.request
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from blindearth.evalspec import masks as M
from blindearth.types import MaskSpec


# --------------------------------------------------------------------------- helpers


def _shp_bytes(records):
    body = b""
    for i, rings in enumerate(records, 1):
        pts = [pt for r in rings for pt in r]
        parts, k = [], 0
        for r in rings:
            parts.append(k)
            k += len(r)
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        content = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), len(rings), len(pts))
        content += struct.pack(f"<{len(parts)}i", *parts)
        content += struct.pack(f"<{2 * len(pts)}d", *[c for q in pts for c in q])
        body += struct.pack(">2i", i, len(content) // 2) + content
    total = 100 + len(body)
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, total // 2)
    header += struct.pack("<2i", 1000, 5) + struct.pack("<8d", *([0.0] * 8))
    return header + body


def _ring(w, s, e, n):
    return [(w, s), (w, n), (e, n), (e, s), (w, s)]


def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


LAND_SHP = _shp_bytes([[_ring(0, 0, 90, 45)]])
GOOD = _zip_bytes({"t/land.shp": LAND_SHP, "t/readme.txt": b"x" * 200})
GOOD2 = _zip_bytes({"t/land.shp": LAND_SHP, "t/readme.txt": b"y" * 300})


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _src(**kw):
    base = dict(
        id="test-src",
        label="Test source",
        url="https://example.invalid/t.zip",
        sha256=None,
        filename="t.zip",
        layers=(M._Layer("t/land.shp", 1), M._Layer("t/lakes.shp", 0, required=False)),
        min_size=100,
    )
    base.update(kw)
    return M._Source(**base)


class FakeResp:
    def __init__(self, body=b"", headers=None, url="https://example.invalid/final", fail_after=None):
        self._buf = io.BytesIO(body)
        self.headers = dict(headers or {})
        self._url = url
        self._fail_after = fail_after
        self._reads = 0

    def read(self, n=-1):
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            raise ConnectionResetError("connection reset")
        return self._buf.read(n)

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def net(monkeypatch):
    calls, queue = [], []

    def fake_urlopen(url, *, opener=None, headers=None, timeout=120):
        calls.append(SimpleNamespace(url=url, headers=headers, opener=opener))
        if not queue:
            raise AssertionError(f"unexpected network call to {url}")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(M, "_urlopen", fake_urlopen)
    for var in (M.ACCEPT_ENV, M.MODIS_DOWNLOAD_ENV, M.MODIS_YEAR_ENV, M.EARTHDATA_TOKEN_ENV):
        monkeypatch.delenv(var, raising=False)
    return SimpleNamespace(calls=calls, queue=queue)


def _lock(tmp_path):
    return json.loads((tmp_path / M.LOCKFILE_NAME).read_text())["files"]


def _leftovers(d):
    return [p.name for p in d.iterdir() if p.name.endswith(".part")] if d.exists() else []


# --------------------------------------------------------------------------- TOFU


def test_first_download_records_tofu_then_reuses(tmp_path, net):
    net.queue.append(FakeResp(GOOD, {"Content-Type": "application/zip", "Content-Length": str(len(GOOD))}))
    path = M._fetch(_src(), tmp_path)
    assert path == tmp_path / "downloads" / "t.zip"
    assert path.read_bytes() == GOOD
    entry = _lock(tmp_path)["downloads/t.zip"]
    assert entry["sha256"] == _sha(GOOD) and entry["size"] == len(GOOD)
    assert entry["url"] == "https://example.invalid/t.zip"
    assert _leftovers(path.parent) == []

    assert M._fetch(_src(), tmp_path) == path  # cached: verified, no network
    assert len(net.calls) == 1


def test_tofu_mismatch_raises_then_accept_env_rerecords(tmp_path, net, monkeypatch):
    net.queue.append(FakeResp(GOOD))
    path = M._fetch(_src(), tmp_path)
    stale = tmp_path / "masks" / "test-src_1km_10x5.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    path.write_bytes(GOOD2)  # replaced behind our back
    with pytest.raises(M.ChecksumMismatchError, match=M.ACCEPT_ENV) as ei:
        M._fetch(_src(), tmp_path)
    assert str(path) in str(ei.value)
    assert _lock(tmp_path)["downloads/t.zip"]["sha256"] == _sha(GOOD)  # unchanged

    monkeypatch.setenv(M.ACCEPT_ENV, "1")
    M._fetch(_src(), tmp_path)
    assert _lock(tmp_path)["downloads/t.zip"]["sha256"] == _sha(GOOD2)
    assert not stale.exists()  # rasterized cache of the old archive dropped
    assert len(net.calls) == 1


def test_redownload_after_delete_is_checked_against_lockfile(tmp_path, net):
    net.queue.append(FakeResp(GOOD))
    path = M._fetch(_src(), tmp_path)
    path.unlink()
    net.queue.append(FakeResp(GOOD2))  # upstream changed
    with pytest.raises(M.ChecksumMismatchError):
        M._fetch(_src(), tmp_path)


def test_manual_file_is_validated_and_recorded(tmp_path, net):
    d = tmp_path / "downloads"
    d.mkdir()
    (d / "t.zip").write_bytes(GOOD)
    M._fetch(_src(), tmp_path)
    assert _lock(tmp_path)["downloads/t.zip"]["sha256"] == _sha(GOOD)
    assert net.calls == []

    other = tmp_path / "other"
    (other / "downloads").mkdir(parents=True)
    (other / "downloads" / "t.zip").write_bytes(b"<html>not a zip</html>" * 20)
    with pytest.raises(M.DownloadIntegrityError):
        M._fetch(_src(), other)
    assert not (other / M.LOCKFILE_NAME).exists()


def test_pinned_sha_overrides_lockfile_and_cannot_be_accepted(tmp_path, net, monkeypatch):
    src = _src(sha256=_sha(GOOD))
    net.queue.append(FakeResp(GOOD))
    path = M._fetch(src, tmp_path)
    assert _lock(tmp_path)["downloads/t.zip"]["pinned"] is True

    # A lockfile that disagrees is ignored: the pin wins.
    lock = tmp_path / M.LOCKFILE_NAME
    lock.write_text(json.dumps({"version": 1, "files": {"downloads/t.zip": {"sha256": "0" * 64}}}))
    M._fetch(src, tmp_path)
    assert _lock(tmp_path)["downloads/t.zip"]["sha256"] == _sha(GOOD)

    path.write_bytes(GOOD2)
    monkeypatch.setenv(M.ACCEPT_ENV, "1")
    with pytest.raises(M.ChecksumMismatchError, match="pinned"):
        M._fetch(src, tmp_path)


def test_unreadable_lockfile_is_an_error(tmp_path, net):
    (tmp_path / M.LOCKFILE_NAME).write_text("{not json")
    net.queue.append(FakeResp(GOOD))
    with pytest.raises(ValueError, match="lockfile"):
        M._fetch(_src(), tmp_path)


# --------------------------------------------------------------------------- download integrity


@pytest.mark.parametrize(
    "resp, match",
    [
        (FakeResp(b"<html>" + b"x" * 500, {"Content-Type": "text/html; charset=UTF-8"}), "HTML"),
        (FakeResp(GOOD, {"Content-Length": str(len(GOOD) + 10)}), "truncated"),
        (FakeResp(GOOD[: len(GOOD) // 2]), "zip"),
        (FakeResp(b"PK"), "only 2 bytes"),
        (FakeResp(_zip_bytes({"t/other.shp": LAND_SHP, "pad": b"z" * 200})), "missing required"),
    ],
)
def test_bad_downloads_are_not_cached(tmp_path, net, resp, match):
    net.queue.append(resp)
    with pytest.raises(M.DownloadIntegrityError, match=match):
        M._fetch(_src(), tmp_path)
    d = tmp_path / "downloads"
    assert not (d / "t.zip").exists()
    assert _leftovers(d) == []
    assert not (tmp_path / M.LOCKFILE_NAME).exists()


def test_corrupt_member_fails_crc(tmp_path, net):
    raw = bytearray(_zip_bytes({"t/land.shp": LAND_SHP, "t/readme.txt": b"x" * 200}))
    i = raw.find(b"x" * 200)  # stored uncompressed: flip a data byte
    raw[i] = ord("q")
    net.queue.append(FakeResp(bytes(raw)))
    with pytest.raises(M.DownloadIntegrityError, match="CRC"):
        M._fetch(_src(), tmp_path)
    assert not (tmp_path / "downloads" / "t.zip").exists()


def test_connection_error_leaves_no_partial_file(tmp_path, net):
    net.queue.append(FakeResp(GOOD, fail_after=0))
    with pytest.raises(ConnectionResetError):
        M._fetch(_src(), tmp_path)
    d = tmp_path / "downloads"
    assert not (d / "t.zip").exists() and _leftovers(d) == []


def test_upstream_size_and_md5(tmp_path, net, monkeypatch):
    good = _src(md5=hashlib.md5(GOOD).hexdigest(), size=len(GOOD))
    net.queue.append(FakeResp(GOOD))
    M._fetch(good, tmp_path / "a")

    for bad in (_src(md5="0" * 32), _src(size=len(GOOD) + 1)):
        net.queue.append(FakeResp(GOOD))
        with pytest.raises(M.DownloadIntegrityError, match=M.ACCEPT_ENV):
            M._fetch(bad, tmp_path / "b")
        assert not (tmp_path / "b" / "downloads" / "t.zip").exists()

    monkeypatch.setenv(M.ACCEPT_ENV, "1")
    net.queue.append(FakeResp(GOOD))
    M._fetch(_src(md5="0" * 32), tmp_path / "c")
    assert (tmp_path / "c" / "downloads" / "t.zip").exists()


def test_builtin_sources_have_sane_pins():
    for src in M.BUILTIN_SOURCES.values():
        assert src.url.startswith("https://")
        assert src.size is None or src.size >= src.min_size
        assert src.sha256 is None or len(src.sha256) == 64
    assert "ngdc.noaa.gov" not in M.GSHHG.url  # that URL now 404s


def test_load_mask_builtin_end_to_end(tmp_path, net, monkeypatch):
    pytest.importorskip("rasterio")
    src = _src()
    monkeypatch.setitem(M.BUILTIN_SOURCES, src.id, src)
    net.queue.append(FakeResp(GOOD))
    mask = M.load_mask(MaskSpec(id=src.id, resolution_km=10_000), cache_dir=tmp_path)
    assert mask.shape == (180, 360)
    assert mask.hash == M.mask_hash(mask.data)  # hash semantics unchanged
    assert mask.data[50, 200] and not mask.data[100, 200] and not mask.data[50, 100]
    again = M.load_mask(MaskSpec(id=src.id, resolution_km=10_000), cache_dir=tmp_path)
    assert again.hash == mask.hash and len(net.calls) == 1


# --------------------------------------------------------------------------- Earthdata auth


def test_bearer_token_only_sent_to_earthdata_https_hosts():
    h = M._EarthdataBearerHandler("tok")
    data = urllib.request.Request("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/x.hdf")
    h.https_request(data)
    assert data.unredirected_hdrs["Authorization"] == "Bearer tok"

    # Following the redirect to S3 must drop the token.
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        data, None, 307, "Temporary Redirect", {},
        "https://lp-prod-protected.s3.us-west-2.amazonaws.com/x.hdf?X-Amz-Signature=abc",
    )
    h.https_request(redirected)
    assert redirected.get_header("Authorization") is None

    smuggled = urllib.request.Request("https://example.com/x", headers={"Authorization": "Bearer tok"})
    h.https_request(smuggled)
    assert smuggled.get_header("Authorization") is None

    plain = urllib.request.Request("http://urs.earthdata.nasa.gov/x")
    h.http_request(plain)
    assert plain.get_header("Authorization") is None

    assert M._is_earthdata_host("urs.earthdata.nasa.gov")
    assert M._is_earthdata_host("data.lpdaac.earthdatacloud.nasa.gov")
    assert not M._is_earthdata_host("earthdata.nasa.gov.evil.example")
    assert not M._is_earthdata_host("evilearthdatacloud.nasa.gov")


def test_earthdata_opener_token_then_netrc_then_error(tmp_path, monkeypatch):
    monkeypatch.delenv(M.EARTHDATA_TOKEN_ENV, raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "missing-netrc"))
    with pytest.raises(M.EarthdataAuthError, match="EARTHDATA_TOKEN"):
        M._earthdata_opener()

    rc = tmp_path / "netrc"
    rc.write_text("machine urs.earthdata.nasa.gov login alice password s3cret\n")
    rc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(rc))
    assert M._netrc_credentials() == ("alice", "s3cret")
    _opener, how = M._earthdata_opener()
    assert how == "netrc"

    monkeypatch.setenv(M.EARTHDATA_TOKEN_ENV, "tok")
    opener, how = M._earthdata_opener()
    assert how == "token"
    assert any(isinstance(hd, M._EarthdataBearerHandler) for hd in opener.handlers)


def test_netrc_without_earthdata_entry_is_ignored(tmp_path, monkeypatch):
    rc = tmp_path / "netrc"
    rc.write_text("machine example.com login bob password pw\n")
    rc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(rc))
    assert M._netrc_credentials() is None


# --------------------------------------------------------------------------- CMR + tiles


def _granule(name, url_ext=".hdf"):
    base = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD44W.061"
    return {
        "umm": {
            "GranuleUR": name,
            "RelatedUrls": [
                {"URL": f"s3://lp-prod-protected/MOD44W.061/{name}/{name}{url_ext}", "Type": "GET DATA VIA DIRECT ACCESS"},
                {"URL": f"{base}/{name}/{name}.cmr.xml", "Type": "VIEW RELATED INFORMATION"},
                {"URL": f"{base}/{name}/{name}{url_ext}", "Type": "GET DATA"},
            ],
        }
    }


def _cmr_page(items, after=None):
    headers = {"Content-Type": "application/vnd.nasa.cmr.umm_results+json"}
    if after:
        headers["CMR-Search-After"] = after
    return FakeResp(json.dumps({"hits": len(items), "items": items}).encode(), headers)


def test_cmr_search_pages_dedupes_and_filters(net):
    net.queue.append(_cmr_page([
        _granule("MOD44W.A2021001.h01v09.061.2022000000000"),
        _granule("MOD44W.A2021001.h01v09.061.2024008090424"),  # newer production wins
        _granule("MOD44W.A2020001.h02v09.061.2024008090424"),  # other year
        _granule("MOD44W.A2021001.h03v09.061.2024008090424", url_ext=".nc"),  # not HDF
    ], after="tok-1"))
    net.queue.append(_cmr_page([_granule("MOD44W.A2021001.h11v10.061.2024008095511")], after="tok-2"))
    net.queue.append(_cmr_page([]))
    gs = M._cmr_mod44w_granules(2021)
    assert [(g.h, g.v) for g in gs] == [(1, 9), (11, 10)]
    assert gs[0].name.endswith("2024008090424")
    assert gs[0].url.startswith("https://data.lpdaac.earthdatacloud.nasa.gov/") and gs[0].url.endswith(".hdf")
    first = net.calls[0]
    assert first.url.startswith(M.CMR_GRANULES_URL + "?")
    for part in ("short_name=MOD44W", "version=061", "provider=LPCLOUD", "page_size=2000"):
        assert part in first.url
    assert first.headers is None
    assert net.calls[1].headers == {"CMR-Search-After": "tok-1"}
    assert net.calls[2].headers == {"CMR-Search-After": "tok-2"}


def test_cmr_search_empty_is_an_error(net):
    net.queue.append(_cmr_page([]))
    with pytest.raises(ValueError, match="no MOD44W"):
        M._cmr_mod44w_granules(2021)


def test_modis_tile_bounds():
    w, s, e, n = M.modis_tile_bounds(17, 8)
    assert (s, n, e) == (0.0, 10.0, pytest.approx(0.0, abs=1e-9))
    assert w == pytest.approx(-10.0 / math.cos(math.radians(10.0)), rel=1e-6)
    assert M.modis_tile_bounds(18, 0) == (-180.0, 80.0, 180.0, 90.0)
    w, s, e, n = M.modis_tile_bounds(0, 8)
    assert w == -180.0 and e == pytest.approx(-170.0, rel=1e-6)
    assert M.modis_tile_bounds(0, 1) is None  # wholly outside the sinusoidal globe


def test_validate_hdf4(tmp_path):
    good = tmp_path / "a.hdf"
    good.write_bytes(M._HDF4_MAGIC + b"\0" * 10)
    M._validate_hdf4(good)
    bad = tmp_path / "b.hdf"
    bad.write_bytes(b"<html>")
    with pytest.raises(M.DownloadIntegrityError):
        M._validate_hdf4(bad)


def test_modis_without_optin_is_file_not_found(tmp_path, net):
    with pytest.raises(FileNotFoundError, match=M.MODIS_DOWNLOAD_ENV):
        M.load_mask(MaskSpec(id="modis-mod44w"), cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        M.load_mask(MaskSpec(id="modis-mod44w", path=str(tmp_path / "nope.tif")), cache_dir=tmp_path)
    assert net.calls == []


def test_modis_optin_without_credentials(tmp_path, net, monkeypatch):
    monkeypatch.setenv(M.MODIS_DOWNLOAD_ENV, "1")
    monkeypatch.setenv("NETRC", str(tmp_path / "missing-netrc"))
    monkeypatch.setattr(M, "_require_hdf4_driver", lambda: None)
    with pytest.raises(M.EarthdataAuthError, match="urs.earthdata.nasa.gov"):
        M.load_mask(MaskSpec(id="modis-mod44w"), cache_dir=tmp_path)
    assert net.calls == []  # fails before any request


def test_modis_bad_year_env(tmp_path, net, monkeypatch):
    monkeypatch.setenv(M.MODIS_DOWNLOAD_ENV, "1")
    monkeypatch.setenv(M.MODIS_YEAR_ENV, "last year")
    with pytest.raises(ValueError, match=M.MODIS_YEAR_ENV):
        M.load_mask(MaskSpec(id="modis-mod44w"), cache_dir=tmp_path)


def test_modis_login_page_is_auth_error(tmp_path, net, monkeypatch):
    monkeypatch.setenv(M.EARTHDATA_TOKEN_ENV, "expired")
    monkeypatch.setattr(M, "_require_hdf4_driver", lambda: None)
    net.queue.append(_cmr_page([_granule("MOD44W.A2021001.h18v04.061.2024008090424")]))
    net.queue.append(FakeResp(b"<html>login</html>", {"Content-Type": "text/html"},
                              url="https://urs.earthdata.nasa.gov/oauth/authorize"))
    with pytest.raises(M.EarthdataAuthError, match="token"):
        M.build_mod44w_mask((180, 360), tmp_path, 2021)
    tiles = tmp_path / "downloads" / "modis-mod44w" / "2021"
    assert not any(tiles.glob("*.hdf"))


def test_modis_auto_download_and_mosaic(tmp_path, net, monkeypatch):
    pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    monkeypatch.setenv(M.MODIS_DOWNLOAD_ENV, "1")
    monkeypatch.setenv(M.EARTHDATA_TOKEN_ENV, "tok")
    monkeypatch.setattr(M, "_require_hdf4_driver", lambda: None)

    T = M.MODIS_TILE_M
    band = np.zeros((40, 40), np.uint8)  # all land
    band[:, 36:] = 1  # eastern strip water
    band[0, 0] = 250  # fill -> water
    read = []

    def fake_read(path):
        read.append(path.name)
        # tile h18v04: x in [0, T], y in [4T, 5T] (lat 40..50 N)
        return band, from_origin(0.0, 5 * T, T / 40, T / 40), CRS.from_string(M.MODIS_SINU_CRS)

    monkeypatch.setattr(M, "_read_mod44w_tile", fake_read)
    name = "MOD44W.A2021001.h18v04.061.2024008090424"
    net.queue.append(_cmr_page([_granule(name)]))
    net.queue.append(FakeResp(M._HDF4_MAGIC + b"\0" * 4096, {"Content-Type": "application/x-hdf"}))

    mask = M.load_mask(MaskSpec(id="modis-mod44w", resolution_km=10_000), cache_dir=tmp_path)
    assert mask.shape == (180, 360)
    assert mask.hash == M.mask_hash(mask.data)
    assert "MOD44W v061 2021" in mask.source
    d = mask.data
    assert d[45, 180:189].all()  # lat 44.5 N, lon 0..9 E: land
    assert not d[:39].any() and not d[51:].any()  # outside the tile's latitude band
    assert not d[:, :179].any() and not d[:, 200:].any()  # outside its longitude span
    assert net.calls[0].opener is None  # CMR search is anonymous
    assert net.calls[1].opener is not None  # tile download is authenticated
    assert f"downloads/modis-mod44w/2021/{name}.hdf" in _lock(tmp_path)
    assert read == [f"{name}.hdf"]

    again = M.load_mask(MaskSpec(id="modis-mod44w", resolution_km=10_000), cache_dir=tmp_path)
    assert again.hash == mask.hash and len(net.calls) == 2  # mosaic cached

    inv = M.load_mask(MaskSpec(id="modis-mod44w", resolution_km=10_000, invert=True), cache_dir=tmp_path)
    assert np.array_equal(inv.data, ~mask.data)
