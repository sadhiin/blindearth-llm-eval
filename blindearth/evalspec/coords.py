"""Coordinate formatting for prompts.

Formats (all part of the eval spec and so of the run hash):

- ``hemisphere`` (Henry's prompt): ``12° S, 45° W``. Degree sign, one space, hemisphere letter.
  Whole degrees print without decimals (``12° S``); fractional degrees print with up to 4
  decimals, trailing zeros stripped (``12.5° S``, ``45.25° W``). Zero prints without a
  hemisphere letter (``0°``), as does the antimeridian (``180°``), because neither belongs to a
  hemisphere. The poles print as ``90° N`` / ``90° S``.
- ``signed_decimal``: ``-12.0, -45.0``. At least one decimal, up to 4, trailing zeros stripped
  beyond the first (``12.5``, ``-45.25``, ``0.0``).
- ``dms``: ``12°30'00" S, 45°15'00" E``. Rounded to the nearest whole second with carry into
  minutes/degrees; same zero / 180 rule as ``hemisphere``.

Latitudes must lie in [-90, 90]. Longitudes in [-180, 180] are printed as given (so -180 prints as
``180°`` / ``-180.0``); anything outside is wrapped into [-180, 180).
"""

from __future__ import annotations

from blindearth.types import CoordFormat

_MAX_DECIMALS = 4
COORD_FORMATS: tuple[str, ...] = ("hemisphere", "signed_decimal", "dms")


def _check(lat: float, lon: float) -> tuple[float, float]:
    lat = float(lat)
    lon = float(lon)
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude {lat} outside [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _plain(x: float) -> str:
    """Absolute-free number: integer without decimals, else up to 4 decimals."""
    v = round(x, _MAX_DECIMALS)
    if v == 0:
        return "0"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.{_MAX_DECIMALS}f}".rstrip("0").rstrip(".")


def format_decimal(x: float) -> str:
    """Signed decimal with at least one decimal place: ``-12.0``, ``12.5``, ``0.0``."""
    v = round(float(x), _MAX_DECIMALS)
    if v == 0:
        v = 0.0
    s = f"{v:.{_MAX_DECIMALS}f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def _letter(v: float, pos: str, neg: str, is_lon: bool) -> str:
    a = round(abs(v), _MAX_DECIMALS)
    if a == 0 or (is_lon and a == 180):
        return ""
    return " " + (pos if v > 0 else neg)


def _hemi(v: float, pos: str, neg: str, is_lon: bool) -> str:
    return f"{_plain(abs(v))}°{_letter(v, pos, neg, is_lon)}"


def _dms(v: float, pos: str, neg: str, is_lon: bool) -> str:
    total = int(round(abs(v) * 3600))
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    letter = "" if total == 0 or (is_lon and total == 180 * 3600) else " " + (pos if v > 0 else neg)
    return f"{d}°{m:02d}'{s:02d}\"{letter}"


def format_coord(lat: float, lon: float, fmt: CoordFormat) -> str:
    """Format one point as ``"<lat>, <lon>"`` in the given style."""
    lat, lon = _check(lat, lon)
    if fmt == "hemisphere":
        return f"{_hemi(lat, 'N', 'S', False)}, {_hemi(lon, 'E', 'W', True)}"
    if fmt == "signed_decimal":
        return f"{format_decimal(lat)}, {format_decimal(lon)}"
    if fmt == "dms":
        return f"{_dms(lat, 'N', 'S', False)}, {_dms(lon, 'E', 'W', True)}"
    raise ValueError(f"unknown coord_format {fmt!r} (use one of {', '.join(COORD_FORMATS)})")
