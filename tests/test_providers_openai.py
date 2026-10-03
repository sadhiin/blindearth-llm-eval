from __future__ import annotations

import json
from types import SimpleNamespace as NS

import httpx
import openai as openai_sdk
import pytest

from blindearth.providers.base import ProviderError, RateLimitError, UnsupportedConfigError
from blindearth.providers.openai_adapter import OpenAIAdapter, map_openai_error, openai_family
from blindearth.types import CallParams, ExtractionMode, ModelSpec, ProviderSpec, RunConfig

PROV = ProviderSpec(id="openai", kind="openai", api_key_env="OPENAI_API_KEY")


def _completion(texts=("Land",), model="gpt-4.1-mini-2025-04-14", reasoning=0, logprobs=None,
                finish="stop"):
    choices = [NS(index=i, message=NS(content=t), finish_reason=finish,
                  logprobs=logprobs if i == 0 else None) for i, t in enumerate(texts)]
    usage = NS(prompt_tokens=50, completion_tokens=2 + reasoning,
               completion_tokens_details=NS(reasoning_tokens=reasoning))
    return NS(choices=choices, usage=usage, model=model)


def _lp(*alts):
    first = NS(token=alts[0][0], logprob=alts[0][1],
               top_logprobs=[NS(token=t, logprob=l) for t, l in alts])
    return NS(content=[first, NS(token=".", logprob=-0.1, top_logprobs=[])])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeFiles:
    def __init__(self):
        self.uploaded = None
        self.contents = {}

    async def create(self, file, purpose):
        self.uploaded = (file, purpose)
        return NS(id="file-in")

    async def content(self, file_id):
        return NS(text=self.contents[file_id])


class FakeBatches:
    def __init__(self):
        self.created = None
        self.state = NS(status="in_progress", output_file_id=None, error_file_id=None, errors=None)

    async def create(self, **kw):
        self.created = kw
        return NS(id="batch_1")

    async def retrieve(self, batch_id):
        return self.state


class FakeClient:
    def __init__(self, responses=()):
        self.chat = NS(completions=FakeCompletions(responses))
        self.files = FakeFiles()
        self.batches = FakeBatches()


def _adapter(name="gpt-4.1-mini", responses=()):
    return OpenAIAdapter(PROV, ModelSpec(id=name, provider="openai", name=name), api_key="k",
                         client=FakeClient(responses))


def test_family_detection():
    assert openai_family("gpt-4o-mini") == "chat"
    assert openai_family("gpt-4.1") == "chat"
    assert openai_family("gpt-5-chat-latest") == "chat"
    assert openai_family("o3") == "o"
    assert openai_family("o4-mini") == "o"
    assert openai_family("gpt-5") == "gpt5"
    assert openai_family("gpt-5-mini") == "gpt5"
    assert openai_family("gpt-5.1") == "gpt51"
    assert openai_family("gpt-5.2-pro") == "gpt52"


def test_chat_model_mapping():
    a = _adapter()
    n = a.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    assert n["logprobs"] is True and n["top_logprobs"] == 20
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(top_logprobs=50), ExtractionMode.LOGPROBS)
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)
    off = a.map_config(RunConfig(effort="off", temperature=1.0, n_samples=4),
                       ExtractionMode.SAMPLE)
    assert "reasoning_effort" not in off and off["temperature"] == 1.0 and off["n"] == 4
    assert a.map_config(RunConfig(), ExtractionMode.GREEDY)["temperature"] == 0.0


def test_reasoning_model_refusals_and_effort_table():
    o3 = _adapter("o3")
    with pytest.raises(UnsupportedConfigError):
        o3.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    with pytest.raises(UnsupportedConfigError):
        o3.map_config(RunConfig(temperature=0.5), ExtractionMode.SAMPLE)
    with pytest.raises(UnsupportedConfigError):
        o3.map_config(RunConfig(), ExtractionMode.GREEDY)
    with pytest.raises(UnsupportedConfigError):
        o3.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)
    with pytest.raises(UnsupportedConfigError):
        o3.map_config(RunConfig(effort="max"), ExtractionMode.SAMPLE)
    n = o3.map_config(RunConfig(effort="high"), ExtractionMode.SAMPLE)
    assert n["reasoning_effort"] == "high" and n["max_completion_tokens"] >= 16000

    assert _adapter("gpt-5").map_config(
        RunConfig(effort="off"), ExtractionMode.SAMPLE)["reasoning_effort"] == "minimal"
    g51 = _adapter("gpt-5.1")
    lp = g51.map_config(RunConfig(effort="off"), ExtractionMode.LOGPROBS)
    assert lp["reasoning_effort"] == "none" and lp["logprobs"] is True
    with pytest.raises(UnsupportedConfigError):
        g51.map_config(RunConfig(effort="low"), ExtractionMode.LOGPROBS)
    with pytest.raises(UnsupportedConfigError):
        g51.map_config(RunConfig(effort="max"), ExtractionMode.SAMPLE)
    assert _adapter("gpt-5.2").map_config(
        RunConfig(effort="max"), ExtractionMode.SAMPLE)["reasoning_effort"] == "xhigh"


async def test_classify_logprobs_and_usage():
    comp = _completion(logprobs=_lp(("Land", -0.05), ("Water", -3.2), (" Land", -6.0)))
    a = _adapter(responses=[comp])
    r = await a.classify("12° S, 45° W", "be terse",
                         CallParams(temperature=0.0, max_output_tokens=8, logprobs=True,
                                    top_logprobs=5, seed=7))
    kw = a._client.chat.completions.calls[0]
    assert kw["logprobs"] is True and kw["top_logprobs"] == 5 and kw["seed"] == 7
    assert kw["extra_body"] == {"max_completion_tokens": 8}
    assert kw["messages"][0] == {"role": "system", "content": "be terse"}
    assert r.first_token_logprobs == {"Land": -0.05, "Water": -3.2, " Land": -6.0}
    assert r.texts == ["Land"]
    assert r.resolved_model == "gpt-4.1-mini-2025-04-14"
    assert r.usage.input_tokens == 50 and r.usage.output_tokens == 2


async def test_classify_reasoning_model_n_and_reasoning_tokens():
    comp = _completion(texts=("Water", "Water", "Land", "Water"), model="o3-2025-04-16",
                       reasoning=300)
    a = _adapter("o3", responses=[comp])
    r = await a.classify("p", "sys", CallParams(temperature=1.0, max_output_tokens=16,
                                                effort="medium", n=4))
    kw = a._client.chat.completions.calls[0]
    assert kw["n"] == 4
    assert "temperature" not in kw
    assert kw["extra_body"]["reasoning_effort"] == "medium"
    assert kw["extra_body"]["max_completion_tokens"] >= 8192
    assert kw["messages"][0]["role"] == "developer"
    assert r.texts == ["Water", "Water", "Land", "Water"]
    assert r.usage.thinking_tokens == 300


async def test_classify_reasoning_model_rejects_logprobs_at_call_time():
    a = _adapter("o3", responses=[_completion()])
    with pytest.raises(ProviderError) as ei:
        await a.classify("p", None, CallParams(temperature=None, max_output_tokens=16,
                                               logprobs=True, top_logprobs=5))
    assert not ei.value.retryable


def _resp(status, headers=None):
    return httpx.Response(status, headers=headers or {},
                          request=httpx.Request("POST", "https://api.openai.com/v1/chat"))


def test_error_mapping():
    e = map_openai_error(openai_sdk.RateLimitError(
        "rate", response=_resp(429, {"retry-after": "2"}), body={"code": "rate_limit_exceeded"}))
    assert isinstance(e, RateLimitError) and e.retry_after_s == 2.0
    e = map_openai_error(openai_sdk.RateLimitError(
        "quota", response=_resp(429), body={"code": "insufficient_quota"}))
    assert not isinstance(e, RateLimitError) and not e.retryable
    e = map_openai_error(openai_sdk.InternalServerError("x", response=_resp(503), body=None))
    assert e.retryable and e.status == 503
    e = map_openai_error(openai_sdk.BadRequestError("x", response=_resp(400), body=None))
    assert not e.retryable
    e = map_openai_error(openai_sdk.APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com")))
    assert e.retryable


async def test_batch_roundtrip():
    a = _adapter()
    bid = await a.submit_batch([
        ("p1", "prompt", None, CallParams(temperature=0.0, max_output_tokens=8, logprobs=True,
                                          top_logprobs=5)),
        ("p2", "prompt2", None, CallParams(temperature=0.0, max_output_tokens=8)),
    ])
    assert bid == "batch_1"
    (fname, data), purpose = a._client.files.uploaded
    assert purpose == "batch"
    lines = [json.loads(x) for x in data.decode().splitlines()]
    assert lines[0]["custom_id"] == "p1" and lines[0]["url"] == "/v1/chat/completions"
    assert lines[0]["body"]["logprobs"] is True
    assert a._client.batches.created["endpoint"] == "/v1/chat/completions"
    assert await a.poll_batch(bid) is None

    ok_body = {"model": "gpt-4.1-mini-2025-04-14",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"content": "Water"},
                            "logprobs": {"content": [{"token": "Water", "logprob": -0.01,
                                                      "top_logprobs": [
                                                          {"token": "Water", "logprob": -0.01},
                                                          {"token": "Land", "logprob": -4.6}]}]}}],
               "usage": {"prompt_tokens": 50, "completion_tokens": 1}}
    a._client.files.contents = {
        "file-out": json.dumps({"custom_id": "p1", "response": {"status_code": 200,
                                                                  "body": ok_body}}),
        "file-err": json.dumps({"custom_id": "p2", "response": {"status_code": 500, "body": {}},
                                "error": {"message": "server error"}}),
    }
    a._client.batches.state = NS(status="completed", output_file_id="file-out",
                                 error_file_id="file-err", errors=None)
    out = await a.poll_batch(bid)
    assert out["p1"].texts == ["Water"]
    assert out["p1"].first_token_logprobs == {"Water": -0.01, "Land": -4.6}
    assert out["p2"].error == "server error"
