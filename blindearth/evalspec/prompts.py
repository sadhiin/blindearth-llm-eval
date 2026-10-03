"""Prompt rendering.

Templates use three placeholders:

- ``{coord}``: the point formatted with the eval's ``coord_format`` (``12° S, 45° W``).
- ``{lat}``, ``{lon}``: signed decimals (``-12.0``), independent of ``coord_format``. They accept
  a Python format spec, e.g. ``{lat:.2f}``.

Only these three names are substituted; any other braces in the template are left untouched, so
templates can contain literal ``{`` / ``}`` without escaping.
"""

from __future__ import annotations

import re

from blindearth.evalspec.coords import format_coord, format_decimal
from blindearth.types import CoordFormat, PromptSpec

DEFAULT_PROMPT_ID = "default-land-water"

# Built-in prompt ids usable as ``eval.prompt: <id>`` in an eval file.
BUILTIN_PROMPTS: dict[str, PromptSpec] = {
    DEFAULT_PROMPT_ID: PromptSpec(),
}

_PLACEHOLDER = re.compile(r"\{(coord|lat|lon)(?::([^{}]*))?\}")


def template_placeholders(template: str) -> set[str]:
    return {m.group(1) for m in _PLACEHOLDER.finditer(template)}


def validate_template(template: str) -> None:
    """Raise ValueError unless the template names the location ({coord}, or both {lat} and {lon})."""
    names = template_placeholders(template)
    if "coord" in names:
        return
    if {"lat", "lon"} <= names:
        return
    raise ValueError(
        "prompt template must contain {coord}, or both {lat} and {lon}; "
        f"found {sorted(names) or 'no placeholders'}"
    )


def render_prompt(prompt: PromptSpec, lat: float, lon: float, fmt: CoordFormat) -> str:
    """The user message for one point. The system prompt is not included (it is sent separately)."""

    def sub(m: re.Match[str]) -> str:
        name, spec = m.group(1), m.group(2)
        if name == "coord":
            if spec:
                raise ValueError("{coord} does not take a format spec")
            return format_coord(lat, lon, fmt)
        value = float(lat if name == "lat" else lon)
        if spec:
            return format(value, spec)
        return format_decimal(value)

    return _PLACEHOLDER.sub(sub, prompt.template)
