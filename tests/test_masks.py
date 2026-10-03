import hashlib
import struct

import numpy as np
import pytest

from blindearth.evalspec.masks import (
    Mask,
    load_mask,
    mask_hash,
    mercator_to_equirect,
    prepare_upload,
    rasterize_shapes,
    read_shp_polygons,
    resample_mask,
    shape_for_resolution,
)
from blindearth.types import MaskSpec

Image = pytest.importorskip("PIL.Image")


def _save_png(path, arr, mode=None):
    Image.fromarray(arr, mode=mode).save(path)
    return path


# --------------------------------------------------------------------------- hash


def test_hash_definition_and_sensitivity():
    rng = np.random.default_rng(0)
    a = rng.random((2050, 16)) > 0.5  # crosses the hashing strip boundary
    expected = hashlib.sha256()
    expected.update(b"blindearth-mask:2050x16:")
    expected.update(np.packbits(a, axis=None).tobytes())
    assert mask_hash(a) == expected.hexdigest()

    b = a.copy()
    b[1000, 3] = ~b[1000, 3]
    assert mask_hash(b) != mask_hash(a)
    assert mask_hash(np.zeros((2, 8), bool)) != mask_hash(np.zeros((4, 4), bool))
    m = Mask.from_array(a.astype(np.uint8), "x")
    assert m.data.dtype == bool and m.hash == mask_hash(a)


def test_land_fraction():
    data = np.zeros((4, 8), bool)
    data[:2] = True  # northern hemisphere land
    m = Mask.from_array(data, "t")
    assert m.land_fraction() == pytest.approx(0.5)
    assert m.land_fraction(area_weighted=False) == pytest.approx(0.5)


def test_shape_for_resolution():
    h, w = shape_for_resolution(1.0)
    assert w == 2 * h and abs(w - 40075) <= 2
    assert shape_for_resolution(10_000) == (180, 360)  # floor at 1 deg
    with pytest.raises(ValueError):
        shape_for_resolution(0)


# --------------------------------------------------------------------------- resampling


def test_resample_down_majority_ties_to_water():
    data = np.zeros((4, 8), bool)
    data[0:2, 0:2] = True  # full block -> land
    data[0, 2] = data[1, 2] = data[0, 3] = True  # 3/4 -> land
    data[2, 0] = data[2, 1] = True  # 2/4 -> tie -> water
    out = resample_mask(data, (2, 4))
    assert out.tolist() == [[True, True, False, False], [False, False, False, False]]


def test_resample_up_nearest_and_mixed():
    data = np.array([[True, False], [False, True]])
    up = resample_mask(data, (4, 4))
    assert up.tolist() == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ]
    mixed = resample_mask(np.ones((4, 8), bool), (8, 4))
    assert mixed.shape == (8, 4) and mixed.all()
    assert resample_mask(data, (2, 2)) is data


def test_resample_non_integer_factor():
    data = np.zeros((10, 20), bool)
    data[:5] = True
    out = resample_mask(data, (3, 6))
    assert out.shape == (3, 6)
    assert out[0].all() and not out[2].any()


# --------------------------------------------------------------------------- uploads


def _block_image():
    img = np.full((20, 40), 30, np.uint8)
    img[5:10, 10:20] = 200
    expected = np.zeros((20, 40), bool)
    expected[5:10, 10:20] = True
    return img, expected


def test_upload_gray_png_otsu(tmp_path):
    img, expected = _block_image()
    p = _save_png(tmp_path / "m.png", img)
    mask, preview, thr = prepare_upload(p, projection="equirectangular", invert=False, threshold=None)
    assert np.array_equal(mask.data, expected)
    assert 30 / 255 <= thr < 200 / 255
    assert preview.dtype == np.uint8 and preview.shape == (20, 40, 3)
    assert (preview[expected] == 255).all() and (preview[~expected] == 0).all()
    assert mask.source.startswith("upload:m.png")


def test_upload_invert_and_fixed_threshold(tmp_path):
    img, expected = _block_image()
    p = _save_png(tmp_path / "m.png", img)
    mask, _, thr = prepare_upload(p, projection="equirectangular", invert=True, threshold=0.5)
    assert thr == 0.5
    assert np.array_equal(mask.data, ~expected)
    mask2, _, _ = prepare_upload(p, projection="equirectangular", invert=False, threshold=0.9)
    assert not mask2.data.any()


def test_upload_rgb_png(tmp_path):
    img, expected = _block_image()
    rgb = np.stack([img] * 3, axis=-1)
    p = _save_png(tmp_path / "rgb.png", rgb)
    mask, _, _ = prepare_upload(p, projection="equirectangular", invert=False, threshold=None)
    assert np.array_equal(mask.data, expected)


def test_upload_transparent_pixels_become_water(tmp_path):
    rgba = np.full((20, 40, 4), 255, np.uint8)
    rgba[:, :20, 3] = 0  # left half transparent white
    rgba[:5, 20:, :3] = 0  # some opaque black too, so Otsu has two levels
    p = _save_png(tmp_path / "a.png", rgba)
    mask, _, _ = prepare_upload(p, projection="equirectangular", invert=False, threshold=None)
    assert not mask.data[:, :20].any()
    assert mask.data[5:, 20:].all()
    assert not mask.data[:5, 20:].any()


def test_upload_aspect_ratio_validated(tmp_path):
    p = _save_png(tmp_path / "bad.png", np.zeros((20, 30), np.uint8))
    with pytest.raises(ValueError, match="2:1"):
        prepare_upload(p, projection="equirectangular", invert=False, threshold=0.5)


def test_upload_uniform_image_needs_threshold(tmp_path):
    p = _save_png(tmp_path / "flat.png", np.zeros((20, 40), np.uint8))
    with pytest.raises(ValueError, match="uniform"):
        prepare_upload(p, projection="equirectangular", invert=False, threshold=None)


def test_upload_web_mercator_reprojected(tmp_path):
    img = np.zeros((64, 64), np.uint8)
    img[:32] = 255  # northern half land
    p = _save_png(tmp_path / "merc.png", img)
    mask, _, _ = prepare_upload(p, projection="web_mercator", invert=False, threshold=None)
    assert mask.shape == (32, 64)
    assert mask.data[:16].all() and not mask.data[16:].any()
    with pytest.raises(ValueError):
        prepare_upload(tmp_path / "merc.png", projection="mollweide", invert=False, threshold=None)


def test_mercator_rows_are_stretched_toward_poles():
    img = np.zeros((100, 100), np.uint8)
    img[:10] = 1  # top 10% of Mercator y: well above 60 N
    out = mercator_to_equirect(img)
    lat_c = 90 - (np.arange(out.shape[0]) + 0.5) * 180 / out.shape[0]
    marked = lat_c[out[:, 0] == 1]
    assert marked.min() > 60


def test_mercator_requires_square():
    with pytest.raises(ValueError):
        mercator_to_equirect(np.zeros((32, 64), np.uint8))


def test_load_mask_upload_and_downsample(tmp_path):
    img = np.zeros((400, 800), np.uint8)
    img[:200] = 255
    p = _save_png(tmp_path / "big.png", img)
    spec = MaskSpec(id="upload", path=str(p), resolution_km=100.0)
    mask = load_mask(spec, cache_dir=tmp_path)
    assert mask.shape == shape_for_resolution(100.0)
    assert mask.data[: mask.shape[0] // 2].all() and not mask.data[mask.shape[0] // 2 :].any()

    small = load_mask(MaskSpec(id="upload", path=str(p), resolution_km=1.0), cache_dir=tmp_path)
    assert small.shape == (400, 800)  # coarser than 1 km: kept as is


def test_load_mask_errors(tmp_path):
    with pytest.raises(ValueError):
        load_mask(MaskSpec(id="upload"), cache_dir=tmp_path)
    with pytest.raises(ValueError):
        load_mask(MaskSpec(id="nope"), cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_mask(MaskSpec(id="modis-mod44w"), cache_dir=tmp_path)


# --------------------------------------------------------------------------- vector rasterization


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


def test_read_shp_polygons():
    buf = _shp_bytes([[_ring(0, 0, 90, 45)], [_ring(-10, -10, 10, 10), _ring(-5, -5, 5, 5)]])
    recs = list(read_shp_polygons(buf))
    assert len(recs) == 2
    assert recs[0][0] == (0.0, 0.0, 90.0, 45.0)
    assert len(recs[1][1]) == 2 and recs[1][1][1].shape == (5, 2)
    with pytest.raises(ValueError):
        list(read_shp_polygons(b"x" * 120))


def test_rasterize_shapes_with_hole_and_burn_order():
    pytest.importorskip("rasterio")
    outer = np.array(_ring(0, 0, 90, 90), float)
    hole = np.array(_ring(22.5, 22.5, 67.5, 67.5), float)
    lake = np.array(_ring(-90, -90, 0, 0), float)
    shapes = [
        ((0, 0, 90, 90), [outer, hole], 1),
        ((-90, -90, 0, 0), [lake], 1),
        ((-90, -90, 0, 0), [lake], 0),  # later burn 0 carves it out again
    ]
    out = rasterize_shapes(shapes, (8, 16))  # 22.5 deg pixels
    expected = np.zeros((8, 16), bool)
    expected[0:4, 8:12] = True
    expected[1:3, 9:11] = False
    assert np.array_equal(out, expected)


def test_upload_geotiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    img, expected = _block_image()
    p = tmp_path / "m.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=20, width=40, count=1, dtype="uint8", crs="EPSG:4326",
        transform=from_bounds(-180, -90, 180, 90, 40, 20),
    ) as dst:
        dst.write(img, 1)
    mask, _, _ = prepare_upload(p, projection="equirectangular", invert=False, threshold=None)
    assert np.array_equal(mask.data, expected)
    with pytest.raises(ValueError):
        prepare_upload(p, projection="web_mercator", invert=False, threshold=None)
