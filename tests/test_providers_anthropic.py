from __future__ import annotations

from types import SimpleNamespace as NS

import anthropic as anthropic_sdk
import httpx
import pytest

from blindearth.providers.anthropic import AnthropicAdapter, anthropic_family, map_anthropic_error
from blindearth.providers.base import ProviderError, RateLimitError, UnsupportedConfigError
from blindearth.types import CallParams, ExtractionMode, ModelSpec, ProviderSpec, RunConfig

PROV = ProviderSpec(id="anthropic", kind="anthropic", api_key_env="ANTHROPIC_API_KEY")


def _msg(text="Land", thinking=None, out=3, think_tokens=None, stop="end_turn",
         model="claude-opus-5-5"):
    content = []
    if thinking is not None:
        content.append(NS(type="thinking", thinking=thinking, signature="sig"))
    content.append(NS(type="text", text=text))
    details = NS(thinking_tokens=think_tokens) if think_tokens is not None else None
    usage = NS(input_tokens=52, output_tokens=out, cache_read_input_tokens=0,
               cache_creation_input_tokens=0, output_tokens_details=details)
    return NS(content=content, usage=usage, stop_reason=stop, model=model)


class AsyncIter:
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


class FakeBatches:
    def __init__(self):
        self.created = None
        self.status = "in_progress"
        self.rows = []

    async def create(self, requests):
        self.created = requests
        return NS(id="msgbatch_1")

    async def retrieve(self, batch_id):
        return NS(id=batch_id, processing_status=self.status)

    async def results(self, batch_id):
        return AsyncIter(self.rows)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.batches = FakeBatches()

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeClient:
    def __init__(self, responses=()):
        self.messages = FakeMessages(responses)


def _adapter(name="claude-opus-5-5", responses=(), **mkw):
    model = ModelSpec(id=name, provider="anthropic", name=name, **mkw)
    return AnthropicAdapter(PROV, model, api_key="k", client=FakeClient(responses))


# --------------------------------------------------------------------------- mapping


def test_family_detection():
    assert anthropic_family("claude-opus-5-5") == "always"
    assert anthropic_family("claude-fable-5-1") == "always"
    assert anthropic_family("claude-opus-5") == "opus5"
    assert anthropic_family("claude-sonnet-5-5") == "sonnet55"
    assert anthropic_family("claude-sonnet-5") == "sonnet5"
    assert anthropic_family("claude-opus-4-8") == "adaptive"
    assert anthropic_family("claude-sonnet-4-6") == "adaptive46"
    assert anthropic_family("claude-haiku-4-5") == "budget"
    assert anthropic_family("claude-sonnet-4-5-20250929") == "budget"


def test_opus55_mapping():
    a = _adapter()
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(), ExtractionMode.GREEDY)  # temperature fixed
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(temperature=0.7), ExtractionMode.SAMPLE)
    n = a.map_config(RunConfig(effort="low", n_samples=4), ExtractionMode.SAMPLE)
    assert n["thinking"] == {"type": "adaptive"}
    assert n["output_config"] == {"effort": "low"}
    assert n["max_tokens"] >= 4096
    assert n["n_requests"] == 4
    n = a.map_config(RunConfig(effort="xhigh"), ExtractionMode.SAMPLE)  # native passthrough
    assert n["output_config"] == {"effort": "xhigh"}


def test_budget_family_mapping():
    a = _adapter("claude-haiku-4-5")
    n = a.map_config(RunConfig(effort="high"), ExtractionMode.SAMPLE)
    assert n["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert n["max_tokens"] >= 16000 + 1024
    off = a.map_config(RunConfig(effort="off", temperature=0.3), ExtractionMode.SAMPLE)
    assert "thinking" not in off and off["temperature"] == 0.3
    g = a.map_config(RunConfig(), ExtractionMode.GREEDY)
    assert g["temperature"] == 0.0
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(effort="xhigh"), ExtractionMode.SAMPLE)


def test_off_mappings_per_family():
    assert _adapter("claude-sonnet-5-5").map_config(
        RunConfig(effort="off"), ExtractionMode.SAMPLE)["thinking"] == {"type": "between_tools"}
    assert _adapter("claude-opus-5").map_config(
        RunConfig(effort="off"), ExtractionMode.SAMPLE)["thinking"] == {"type": "disabled"}
    assert "thinking" not in _adapter("claude-sonnet-4-6").map_config(
        RunConfig(effort="off"), ExtractionMode.SAMPLE)


def test_default_capabilities():
    caps = _adapter().default_capabilities()
    assert not caps.logprobs and caps.supports_batch
    assert "off" not in caps.supported_efforts and "low" in caps.supported_efforts
    assert "off" in _adapter("claude-haiku-4-5").default_capabilities().supported_efforts


# --------------------------------------------------------------------------- classify


async def test_classify_loops_n_and_strips_thinking():
    resp = [_msg("Water", thinking="", out=40, think_tokens=38) for _ in range(3)]
    a = _adapter(responses=resp)
    r = await a.classify("12° S, 45° W", "sys", CallParams(temperature=1.0, max_output_tokens=16,
                                                            effort="low", n=3))
    calls = a._client.messages.calls
    assert len(calls) == 3
    assert calls[0]["system"] == "sys"
    assert calls[0]["thinking"] == {"type": "adaptive"}
    assert calls[0]["extra_body"] == {"output_config": {"effort": "low"}}
    assert calls[0]["max_tokens"] >= 4096
    assert "temperature" not in calls[0]  # fixed by the provider
    assert r.texts == ["Water"] * 3
    assert r.thinking_texts == [None] * 3
    assert r.usage.thinking_tokens == 114 and r.usage.output_tokens == 120
    assert r.usage.input_tokens == 156
    assert r.resolved_model == "claude-opus-5-5"
    assert r.finish_reasons == ["end_turn"] * 3
    assert r.first_token_logprobs is None


async def test_store_thinking_requests_summary_and_keeps_text():
    a = _adapter(responses=[_msg("Land", thinking="Paris is inland", out=20, think_tokens=18)])
    r = await a.classify("p", None, CallParams(temperature=None, max_output_tokens=16,
                                               effort="medium", store_thinking=True))
    assert a._client.messages.calls[0]["thinking"]["display"] == "summarized"
    assert r.thinking_texts == ["Paris is inland"]


async def test_thinking_tokens_estimated_when_not_reported():
    a = _adapter("claude-haiku-4-5",
                 responses=[_msg("Land", thinking="long reasoning", out=200, think_tokens=None)])
    r = await a.classify("p", None, CallParams(temperature=None, max_output_tokens=16,
                                               effort="low"))
    assert 190 <= r.usage.thinking_tokens < 200


async def test_sampling_model_sends_temperature():
    a = _adapter("claude-haiku-4-5", responses=[_msg("Land")])
    await a.classify("p", None, CallParams(temperature=1.0, max_output_tokens=16))
    assert a._client.messages.calls[0]["temperature"] == 1.0
    assert "thinking" not in a._client.messages.calls[0]


async def test_classify_refuses_unhonourable_temperature():
    a = _adapter(responses=[_msg()])
    with pytest.raises(ProviderError) as ei:
        await a.classify("p", None, CallParams(temperature=0.0, max_output_tokens=16))
    assert not ei.value.retryable


# --------------------------------------------------------------------------- errors


def _resp(status, headers=None):
    return httpx.Response(status, headers=headers or {},
                          request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def test_error_mapping():
    e = map_anthropic_error(anthropic_sdk.RateLimitError(
        "rate", response=_resp(429, {"retry-after": "7"}), body=None))
    assert isinstance(e, RateLimitError) and e.retry_after_s == 7.0
    e = map_anthropic_error(anthropic_sdk.InternalServerError(
        "overloaded", response=_resp(529), body=None))
    assert isinstance(e, ProviderError) and e.retryable and e.status == 529
    e = map_anthropic_error(anthropic_sdk.AuthenticationError(
        "bad key", response=_resp(401), body=None))
    assert isinstance(e, ProviderError) and not e.retryable
    e = map_anthropic_error(anthropic_sdk.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com")))
    assert isinstance(e, ProviderError) and e.retryable


async def test_classify_maps_sdk_errors():
    a = _adapter(responses=[anthropic_sdk.RateLimitError(
        "rate", response=_resp(429, {"retry-after": "3"}), body=None)])
    with pytest.raises(RateLimitError) as ei:
        await a.classify("p", None, CallParams(temperature=None, max_output_tokens=16))
    assert ei.value.retry_after_s == 3.0


# --------------------------------------------------------------------------- batches


async def test_batch_submit_and_poll():
    a = _adapter("claude-haiku-4-5")
    bid = await a.submit_batch([
        ("p1", "prompt one", None, CallParams(temperature=1.0, max_output_tokens=16, n=2)),
        ("p2", "prompt two", "sys", CallParams(temperature=1.0, max_output_tokens=16, n=1)),
    ])
    assert bid == "msgbatch_1"
    batches = a._client.messages.batches
    ids = [r["custom_id"] for r in batches.created]
    assert ids == ["p1__s0", "p1__s1", "p2__s0"]
    assert batches.created[2]["params"]["system"] == "sys"
    assert await a.poll_batch(bid) is None

    batches.status = "ended"
    batches.rows = [
        NS(custom_id="p1__s1", result=NS(type="succeeded", message=_msg("Water"))),
        NS(custom_id="p1__s0", result=NS(type="succeeded", message=_msg("Land"))),
        NS(custom_id="p2__s0", result=NS(type="errored",
                                         error=NS(type="error",
                                                  error=NS(type="overloaded_error")))),
    ]
    out = await a.poll_batch(bid)
    assert out["p1"].texts == ["Land", "Water"]
    assert out["p1"].error is None
    assert out["p1"].usage.input_tokens == 104
    assert out["p2"].texts == [] and "overloaded_error" in out["p2"].error
