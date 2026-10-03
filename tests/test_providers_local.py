"""TransformersAdapter pieces that need no torch: token matching, logprob merging, config mapping.

The forward pass itself needs torch + a model download, so it is exercised only through a fake
model/tokenizer injected into the adapter (still no torch import: torch is stubbed).
"""

from __future__ import annotations

import math
import sys
import types

import pytest

from blindearth.providers.base import UnsupportedConfigError
from blindearth.providers.local import (
    TransformersAdapter,
    is_land_water_piece,
    land_water_token_ids,
    logprob_dict,
    normalize_piece,
)
from blindearth.types import CallParams, ExtractionMode, ModelSpec, ProviderSpec, RunConfig

PROV = ProviderSpec(id="hf", kind="transformers")


def _adapter(**extra):
    return TransformersAdapter(PROV, ModelSpec(id="q", provider="hf",
                                               name="Qwen/Qwen2.5-0.5B-Instruct", quant="bf16",
                                               extra=extra))


def test_normalize_piece_space_markers():
    assert normalize_piece("ĠLand") == " Land"
    assert normalize_piece("▁Water") == " Water"
    assert normalize_piece("La") == "La"


@pytest.mark.parametrize("surface,ok", [
    ("Land", True), (" Land", True), ("land", True), ("LAND", True), ("La", True), (" wat", True),
    ("W", True), ("Water.", True), (" Landscape", True),
    ("  Land", False), ("", False), (" ", False), ("Lamp", False), ("Ocean", False),
    ("ater", False),
])
def test_is_land_water_piece(surface, ok):
    assert is_land_water_piece(surface) is ok


def test_land_water_token_ids_and_logprob_dict():
    pieces = ["<s>", "Land", "ĠLand", "La", "nd", "ĠWater", "Ocean", None, "▁Land"]
    ids = land_water_token_ids(pieces)
    assert ids == [1, 2, 3, 5, 8]
    # ids 2 and 8 share the surface " Land": their probabilities are summed
    d = logprob_dict([1, 2, 8, 5], [math.log(0.5), math.log(0.1), math.log(0.1), math.log(0.2)],
                     pieces)
    assert d["Land"] == pytest.approx(math.log(0.5))
    assert d[" Land"] == pytest.approx(math.log(0.2))
    assert d[" Water"] == pytest.approx(math.log(0.2))


def test_map_config():
    a = _adapter()
    n = a.map_config(RunConfig(effort="off"), ExtractionMode.LOGPROBS)
    assert n["chat_template_kwargs"] == {"enable_thinking": False}
    assert n["quant"] == "bf16"
    assert n["top_k"] == 20
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(top_logprobs=5), ExtractionMode.SAMPLE)
    b = _adapter(effort_map={"high": {"enable_thinking": True}})
    with pytest.raises(UnsupportedConfigError):  # thinking first -> first-token logprobs useless
        b.map_config(RunConfig(effort="high"), ExtractionMode.LOGPROBS)
    assert b.map_config(RunConfig(effort="high"), ExtractionMode.SAMPLE)[
        "chat_template_kwargs"] == {"enable_thinking": True}
    caps = a.default_capabilities()
    assert caps.logprobs and caps.top_logprobs_max is None and caps.max_concurrency == 1


def test_no_torch_import_at_construction(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    a = _adapter()
    a.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    assert "torch" not in sys.modules  # constructing/mapping must not import torch


# --------------------------------------------------------------------------- fake forward pass


class _T:
    """Tiny tensor stand-in with just what _run_sync uses."""

    device = "cpu"

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        d = self.data
        s = []
        while isinstance(d, list):
            s.append(len(d))
            d = d[0] if d else None
        return tuple(s)

    def to(self, device):
        return self

    def float(self):
        return self

    def tolist(self):
        return self.data

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2 and isinstance(idx[1], slice):  # seqs = out[:, n_in:]
                return _T([row[idx[1]] for row in self.data])
            r = self.data
            for i in idx:
                r = r[i]
            return _T(r)
        if isinstance(idx, _T):
            return _T([self.data[i] for i in idx.data])
        return _T(self.data[idx])


def _fake_torch():
    t = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    t.no_grad = lambda: _NoGrad()
    t.manual_seed = lambda s: None

    def log_softmax(x, dim=-1):
        m = max(x.data)
        z = math.log(sum(math.exp(v - m) for v in x.data)) + m
        return _T([v - z for v in x.data])

    def topk(x, k):
        order = sorted(range(len(x.data)), key=lambda i: -x.data[i])[:k]
        return types.SimpleNamespace(indices=_T(order), values=_T([x.data[i] for i in order]))

    t.log_softmax = log_softmax
    t.topk = topk
    t.tensor = lambda data, device=None: _T(list(data))
    return t


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 9
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        self.kw = kw
        return "PROMPT"

    def __call__(self, text, return_tensors, add_special_tokens):
        return {"input_ids": _T([[1, 2, 3]]), "attention_mask": _T([[1, 1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        words = {4: "Land", 5: "Water", 9: ""}
        return "".join(words.get(i, "?") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return text.split()


class _FakeModel:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, input_ids, attention_mask=None):
        return types.SimpleNamespace(logits=_T([[[0.0] * len(self.logits), self.logits]]))

    def generate(self, input_ids, attention_mask, pad_token_id, **kw):
        self.gen_kw = kw
        return _T([[1, 2, 3, 4, 9]])


async def test_fake_forward_pass_exact_logprobs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    a = _adapter()
    # vocab: 0 <pad>, 1-3 prompt, 4 "Land", 5 "Water", 6 " Land", 7 "Ocean", 8 "La", 9 eos
    pieces = ["<pad>", "a", "b", "c", "Land", "Water", "ĠLand", "Ocean", "La", "</s>"]
    logits = [-9.0, -9.0, -9.0, -9.0, 3.0, 1.0, 0.5, 2.0, -1.0, -9.0]
    a._tok, a._model, a._device = _FakeTok(), _FakeModel(logits), "cpu"
    a._pieces, a._lw_ids = pieces, land_water_token_ids(pieces)
    a._load = lambda: None
    r = await a.classify("p", None, CallParams(temperature=0.0, max_output_tokens=4,
                                               logprobs=True, top_logprobs=2, effort="off"))
    z = math.log(sum(math.exp(v) for v in logits))
    lp = r.first_token_logprobs
    assert lp["Land"] == pytest.approx(3.0 - z)
    assert lp["Ocean"] == pytest.approx(2.0 - z)  # in the top-2
    assert lp["Water"] == pytest.approx(1.0 - z)  # land/water candidates always included
    assert lp[" Land"] == pytest.approx(0.5 - z)
    assert lp["La"] == pytest.approx(-1.0 - z)
    assert r.texts == ["Land"]
    assert r.finish_reasons == ["stop"]
    assert a._tok.kw == {"enable_thinking": False}
    assert a._model.gen_kw["do_sample"] is False
    assert r.usage.input_tokens == 3 and r.usage.output_tokens == 2
