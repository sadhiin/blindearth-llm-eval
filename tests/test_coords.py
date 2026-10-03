import pytest

from blindearth.evalspec.coords import format_coord, format_decimal


@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (-12, -45, "12° S, 45° W"),
        (12, 45, "12° N, 45° E"),
        (89, -179, "89° N, 179° W"),
        (0, 0, "0°, 0°"),
        (-0.0, -0.0, "0°, 0°"),
        (12.5, -45.25, "12.5° N, 45.25° W"),
        (90, 180, "90° N, 180°"),
        (-90, -180, "90° S, 180°"),
        (1.0, 0.00001, "1° N, 0°"),
    ],
)
def test_hemisphere(lat, lon, expected):
    assert format_coord(lat, lon, "hemisphere") == expected


@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (-12, -45, "-12.0, -45.0"),
        (12.5, 0, "12.5, 0.0"),
        (-0.0, 179, "0.0, 179.0"),
        (1.23456, -2.5, "1.2346, -2.5"),
    ],
)
def test_signed_decimal(lat, lon, expected):
    assert format_coord(lat, lon, "signed_decimal") == expected


@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (-12.5, 45.25, "12°30'00\" S, 45°15'00\" E"),
        (0, -179, "0°00'00\", 179°00'00\" W"),
        (10.999999, 0.5, "11°00'00\" N, 0°30'00\" E"),
        (-90, 180, "90°00'00\" S, 180°00'00\""),
    ],
)
def test_dms(lat, lon, expected):
    assert format_coord(lat, lon, "dms") == expected


def test_out_of_range_lat_rejected():
    with pytest.raises(ValueError):
        format_coord(91, 0, "hemisphere")


def test_lon_wraps():
    assert format_coord(0, 190, "signed_decimal") == "0.0, -170.0"


def test_unknown_format():
    with pytest.raises(ValueError):
        format_coord(0, 0, "utm")  # type: ignore[arg-type]


def test_format_decimal():
    assert format_decimal(-12) == "-12.0"
    assert format_decimal(0.1) == "0.1"
