import math

import pytest

from blindearth.extract import (
    classify_token,
    extract,
    p_land_from_logprobs,
    parse_answer,
    strip_thinking,
)
from blindearth.types import ClassifyResult, ExtractionMode, Usage


def _res(texts=None, logprobs=None, error=None):
    return ClassifyResult(
        texts=list(texts or []), usage=Usage(), latency_s=0.1,
        first_token_logprobs=logprobs, error=error,
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Land", "land"),
        ("Water", "water"),
        ("land", "land"),
        ("WATER.", "water"),
        ("  'Land'  ", "land"),
        ("**Water**", "water"),
        ("- Land", "land"),
        ("Land, because it is Brazil.", "land"),
        ("<think>hmm, ocean?</think>\nWater", "water"),
        ("<thinking>a</thinking> <thinking>b</thinking> Land", "land"),
        ("<think>still thinking about whether it is Land", None),
        ("<|channel|>analysis<|message|>coast...<|end|><|start|>assistant<|channel|>final<|message|>Land<|return|>", "land"),
        ("Answer: Land", None),
        ("Land/Water", None),
        ("Landmass", None),
        ("I can't determine that.", None),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_strip_thinking():
    assert strip_thinking("<think>x</think>Land") == "Land"
    assert strip_thinking("Water") == "Water"
    assert strip_thinking("<think>x") is None


@pytest.mark.parametrize(
    "tok,expected",
    [
        ("Land", "land"), (" Land", "land"), ("land", "land"), (" LAND", "land"),
        ("La", "land"), (" La", "land"), ("Lan", "land"), ("ĠLand", "land"), ("▁Water", "water"),
        ("Wa", "water"), ("Water", "water"), ("Land.", "land"), ("'Land", "land"),
        ("L", None), ("W", None), ("Lands", None), ("Lo", None), ("I", None), ("", None), (" ", None),
    ],
)
def test_classify_token(tok, expected):
    assert classify_token(tok) == expected


def test_p_land_softmax_and_validity_mass():
    lp = {"Land": math.log(0.6), "Water": math.log(0.3), "I": math.log(0.1)}
    p, mass = p_land_from_logprobs(lp)
    assert p == pytest.approx(2 / 3)
    assert mass == pytest.approx(0.9)


def test_p_land_sums_variants_and_split_token():
    lp = {"La": math.log(0.2), " Land": math.log(0.2), "land": math.log(0.1), " Water": math.log(0.25)}
    p, mass = p_land_from_logprobs(lp)
    assert p == pytest.approx(0.5 / 0.75)
    assert mass == pytest.approx(0.75)


def test_p_land_no_match():
    assert p_land_from_logprobs({"Sorry": -0.1, "I": -2.0}) == (None, 0.0)
    assert p_land_from_logprobs({}) == (None, 0.0)


def test_extract_logprobs():
    r = _res(["Land"], {"Land": math.log(0.8), "Water": math.log(0.2)})
    e = extract(r, ExtractionMode.LOGPROBS)
    assert e.p_land == pytest.approx(0.8)
    assert e.validity_mass == pytest.approx(1.0)
    assert (e.n_valid, e.n_samples, e.invalid) == (1, 1, False)
    assert e.answer_text == "Land"


def test_extract_logprobs_low_mass_still_valid_but_none_when_absent():
    r = _res(["Hmm"], {"Hmm": math.log(0.95), "Land": math.log(0.05)})
    e = extract(r, ExtractionMode.LOGPROBS)
    assert e.p_land == pytest.approx(1.0) and e.validity_mass == pytest.approx(0.05)
    assert not e.invalid

    e2 = extract(_res(["Land"], None), ExtractionMode.LOGPROBS)
    assert e2.p_land is None and e2.invalid and e2.validity_mass is None


def test_extract_greedy():
    assert extract(_res(["Land"]), ExtractionMode.GREEDY).p_land == 1.0
    assert extract(_res(["water."]), ExtractionMode.GREEDY).p_land == 0.0
    e = extract(_res(["I refuse"]), ExtractionMode.GREEDY)
    assert e.p_land is None and e.invalid and e.n_valid == 0 and e.n_samples == 1
    assert extract(_res([]), ExtractionMode.GREEDY).invalid


def test_extract_sample_excludes_invalid():
    e = extract(_res(["Land", "Water", "Land", "Sorry"]), ExtractionMode.SAMPLE)
    assert e.p_land == pytest.approx(2 / 3)
    assert (e.n_valid, e.n_samples, e.invalid) == (3, 4, False)
    assert e.validity_mass is None

    bad = extract(_res(["", "no idea"]), ExtractionMode.SAMPLE)
    assert bad.p_land is None and bad.invalid and bad.n_valid == 0 and bad.n_samples == 2


def test_extract_error_and_auto():
    e = extract(_res(["Land"], error="HTTP 500"), ExtractionMode.SAMPLE)
    assert e.p_land is None and e.invalid

    auto_lp = extract(_res(["Water"], {"Water": 0.0}), ExtractionMode.AUTO)
    assert auto_lp.p_land == 0.0 and auto_lp.validity_mass == pytest.approx(1.0)
    auto_s = extract(_res(["Water", "Land"]), ExtractionMode.AUTO)
    assert auto_s.p_land == 0.5 and auto_s.validity_mass is None


def test_answer_text_truncated():
    e = extract(_res(["x" * 500]), ExtractionMode.GREEDY)
    assert len(e.answer_text) == 200
