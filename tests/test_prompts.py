import pytest

from blindearth.evalspec.prompts import BUILTIN_PROMPTS, render_prompt, validate_template
from blindearth.types import PromptSpec


def test_default_prompt_matches_henry():
    out = render_prompt(PromptSpec(), -12, -45, "hemisphere")
    assert out == (
        "If this location is over land, say 'Land'. If this location is over water, say 'Water'. "
        "Do not say anything else. 12° S, 45° W"
    )


def test_builtin_registry():
    assert BUILTIN_PROMPTS["default-land-water"] == PromptSpec()


def test_coord_follows_format():
    p = PromptSpec(id="c", template="Where: {coord}")
    assert render_prompt(p, -12, -45, "signed_decimal") == "Where: -12.0, -45.0"


def test_lat_lon_placeholders_and_format_spec():
    p = PromptSpec(id="c", template="lat={lat} lon={lon} ({lat:.2f}, {lon:+.1f})")
    assert render_prompt(p, -12, 45.5, "hemisphere") == "lat=-12.0 lon=45.5 (-12.00, +45.5)"


def test_other_braces_untouched():
    p = PromptSpec(id="c", template='Reply as JSON {"answer": ...} for {coord}')
    assert render_prompt(p, 1, 1, "hemisphere") == 'Reply as JSON {"answer": ...} for 1° N, 1° E'


def test_validate_template():
    validate_template("{coord}")
    validate_template("{lat} {lon}")
    with pytest.raises(ValueError):
        validate_template("{lat} only")
    with pytest.raises(ValueError):
        validate_template("no placeholders")
