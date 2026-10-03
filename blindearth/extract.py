"""Turn one model response into P(Land) plus a validity flag (spec: "Probability extraction").

Parsing (all modes)
    Any thinking block is removed first: text after the last ``</think>`` / ``</thinking>`` /
    ``</reasoning>`` closing tag (or after the last ``<|message|>`` of a gpt-oss style
    ``<|channel|>final`` block). An opening tag with no closing tag means the answer was cut off
    inside the thinking, which is invalid. The first whitespace-separated word whose
    punctuation-stripped form is non-empty is taken (so ``**Land**``, ``'Water'.`` and ``- Land``
    parse), punctuation is stripped from both ends, and only ``land`` / ``water`` (any case) are
    accepted. Anything else (refusals, ``Answer: Land``, ``Land/Water``, empty or truncated output)
    is invalid. Only the first word counts, so ``Land, because ...`` is Land.

Logprobs mode
    Each first-token alternative is normalized (leading whitespace and the BPE / SentencePiece
    space markers ``Ġ`` and ``▁`` removed, plus a leading quote or ``*``; lowercased). It counts as
    Land when it is a prefix of ``land`` of at least 2 characters (``La``, ``Lan``, ``Land``, as
    Henry did for the ``La`` + ``nd`` split) or ``land`` followed only by punctuation/whitespace;
    the same for Water. Single-letter tokens are ignored because ``L``/``W`` start too many other
    words. Probabilities of all matching tokens are summed per class (so ``"Land"`` and
    ``" Land"`` add up). ``validity_mass = P(Land) + P(Water)`` from the raw distribution, and
    ``p_land = P(Land) / validity_mass`` (the softmax over the two). No matching token gives
    ``p_land = None`` and ``validity_mass = 0``. If the adapter returned no logprobs at all the
    point is invalid (it is not silently re-scored from the text, which would mix modes).

Sample mode
    Each sample is parsed. ``n_valid`` = samples that said Land or Water, ``n_samples`` = samples
    returned. ``p_land = land / n_valid`` when ``n_valid > 0``; with no valid sample the point is
    invalid and ``p_land`` is ``None``. Invalid samples are therefore excluded rather than counted
    against either class; the invalid rate stays visible through ``n_valid < n_samples``.

Greedy mode
    The first text only; ``p_land`` is 1.0, 0.0 or ``None`` (invalid).

``ExtractionMode.AUTO`` should be resolved by the planner; if it reaches here it is treated as
logprobs when the result carries logprobs, else as sample mode.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from blindearth.types import ClassifyResult, ExtractionMode, Extracted

ANSWER_TEXT_MAX = 200
MIN_PREFIX_LEN = 2
LOW_VALIDITY_MASS = 0.5  # below this, reports flag the point (the model wanted to say something else)

_TARGETS = ("land", "water")
_PUNCT = "".join(
    sorted(set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "“”‘’«»„‚‹›…–—·•¡¿「」『』。、，．：；！？"))
)
_CLOSE_RE = re.compile(r"</\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
_OPEN_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>", re.IGNORECASE)
_HARMONY_TAG_RE = re.compile(r"<\|[a-z_]+\|>", re.IGNORECASE)


def strip_thinking(text: str) -> str | None:
    """Final answer text with any thinking block removed; None if cut off inside thinking."""
    if not text:
        return text
    m = None
    for m in _HARMONY_FINAL_RE.finditer(text):
        pass
    if m is not None:
        text = _HARMONY_TAG_RE.sub(" ", text[m.end():])
    closes = list(_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end():]
    if _OPEN_RE.search(text):
        return None
    return text


def parse_answer(text: str) -> Literal["land", "water"] | None:
    if text is None:
        return None
    final = strip_thinking(text)
    if not final:
        return None
    for word in final.split():
        w = word.strip(_PUNCT)
        if not w:
            continue
        w = w.lower()
        if w in _TARGETS:
            return w  # type: ignore[return-value]
        return None
    return None


def _norm_token(tok: str) -> str:
    t = tok.lstrip(" \t\r\n Ġ▁Ċ\"'`*“‘")
    return t.lower()


def classify_token(tok: str) -> Literal["land", "water"] | None:
    """Which answer a first-token alternative starts, by case-insensitive prefix match."""
    t = _norm_token(tok)
    if not t:
        return None
    for target in _TARGETS:
        if len(t) >= MIN_PREFIX_LEN and target.startswith(t):
            return target  # type: ignore[return-value]
        if t.startswith(target) and not t[len(target):].strip(_PUNCT + " \t\r\n"):
            return target  # type: ignore[return-value]
    return None


def p_land_from_logprobs(first_token_logprobs: dict[str, float]) -> tuple[float | None, float]:
    """(p_land, validity_mass) from first-token logprobs (natural log)."""
    p = {"land": 0.0, "water": 0.0}
    for tok, lp in (first_token_logprobs or {}).items():
        cls = classify_token(tok)
        if cls is None or lp is None:
            continue
        lp = float(lp)
        if math.isnan(lp):
            continue
        p[cls] += math.exp(lp) if lp > -745 else 0.0
    mass = p["land"] + p["water"]
    if mass <= 0.0:
        return None, 0.0
    return p["land"] / mass, min(mass, 1.0)


def _answer_text(result: ClassifyResult) -> str:
    t = result.texts[0] if result.texts else ""
    return (t or "")[:ANSWER_TEXT_MAX]


def extract(result: ClassifyResult, mode: ExtractionMode) -> Extracted:
    mode = ExtractionMode(mode)
    texts = list(result.texts or [])
    answer_text = _answer_text(result)

    if result.error is not None:
        return Extracted(None, 0, len(texts), None, answer_text, True)

    if mode is ExtractionMode.AUTO:
        mode = ExtractionMode.LOGPROBS if result.first_token_logprobs else ExtractionMode.SAMPLE

    if mode is ExtractionMode.LOGPROBS:
        if not result.first_token_logprobs:
            return Extracted(None, 0, 1, None, answer_text, True)
        p, mass = p_land_from_logprobs(result.first_token_logprobs)
        return Extracted(p, 1 if p is not None else 0, 1, mass, answer_text, p is None)

    if mode is ExtractionMode.GREEDY:
        ans = parse_answer(texts[0]) if texts else None
        p = None if ans is None else (1.0 if ans == "land" else 0.0)
        return Extracted(p, 0 if p is None else 1, 1, None, answer_text, p is None)

    if mode is ExtractionMode.SAMPLE:
        answers = [parse_answer(t) for t in texts]
        n_land = sum(a == "land" for a in answers)
        n_valid = sum(a is not None for a in answers)
        p = n_land / n_valid if n_valid > 0 else None
        return Extracted(p, n_valid, len(texts), None, answer_text, n_valid == 0)

    raise ValueError(f"unknown extraction mode {mode!r}")
