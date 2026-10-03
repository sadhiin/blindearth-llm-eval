from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from blindearth.providers.base import ProviderError, RateLimitError, UnsupportedConfigError
from blindearth.providers.google import GoogleAdapter, gemini_family, map_google_error
from blindearth.types import Capabilities, CallParams, ExtractionMode, ModelSpec, ProviderSpec, RunConfig

PROV = ProviderSpec(id="google", kind="google", api_key_env="GEMINI_API_KEY")


def _resp(answer="Land", thought=None, thoughts_n=0, logprobs=None, finish="STOP",
          version="gemini-2.5-flash-001"):
    parts = []
    if thought is not None:
        parts.append(NS(text=thought, thought=True))
    parts.append(NS(text=answer, thought=None))
    cand = NS(content=NS(parts=parts), finish_reason=NS(name=finish), logprobs_result=logprobs)
    um = NS(prompt_token_count=48, candidates_token_count=1, thoughts_token_count=thoughts_n)
    return NS(candidates=[cand], usage_metadata=um, model_version=version, prompt_feedback=None)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeClient:
    def __init__(self, responses=()):
        self.aio = NS(models=FakeModels(responses))


def _adapter(name="gemini-2.5-flash", responses=()):
    return GoogleAdapter(PROV, ModelSpec(id=name, provider="google", name=name), api_key="k",
                         client=FakeClient(responses))


def test_family_detection():
    assert gemini_family("gemini-2.5-pro") == "g25pro"
    assert gemini_family("gemini-2.5-flash-lite") == "g25flash"
    assert gemini_family("gemini-3-pro-preview") == "g3"
    assert gemini_family("gemini-2.0-flash") == "g2"


def test_effort_mapping_table():
    flash = _adapter()
    assert flash.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_budget": 0}
    assert flash.map_config(RunConfig(effort="medium"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_budget": 8192}
    pro = _adapter("gemini-2.5-pro")
    with pytest.raises(UnsupportedConfigError):
        pro.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)
    assert pro.map_config(RunConfig(effort="max"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_budget": 32768}
    g3 = _adapter("gemini-3-pro-preview")
    assert g3.map_config(RunConfig(effort="high"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_level": "HIGH"}
    for bad in ("off", "max"):
        with pytest.raises(UnsupportedConfigError):
            g3.map_config(RunConfig(effort=bad), ExtractionMode.SAMPLE)
    g2 = _adapter("gemini-2.0-flash")
    assert "thinking_config" not in g2.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)
    with pytest.raises(UnsupportedConfigError):
        g2.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)


def test_logprobs_gated_by_capabilities():
    flash = _adapter()
    with pytest.raises(UnsupportedConfigError):
        flash.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    flash.capabilities = Capabilities(logprobs=True, top_logprobs_max=20,
                                      probed_at="2026-10-01T00:00:00+00:00")
    n = flash.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    assert n["response_logprobs"] is True and n["logprobs"] == 20


async def test_classify_strips_thoughts_and_counts_them():
    a = _adapter(responses=[_resp("Water", thought="ocean...", thoughts_n=120)] * 2)
    r = await a.classify("p", "sys", CallParams(temperature=1.0, max_output_tokens=16,
                                                effort="low", n=2, store_thinking=True))
    cfg = a._client.aio.models.calls[0]["config"]
    assert cfg["system_instruction"] == "sys"
    assert cfg["thinking_config"] == {"thinking_budget": 1024, "include_thoughts": True}
    assert cfg["max_output_tokens"] >= 4096
    assert len(a._client.aio.models.calls) == 2
    assert r.texts == ["Water", "Water"]
    assert r.thinking_texts == ["ocean...", "ocean..."]
    assert r.usage.thinking_tokens == 240
    assert r.usage.output_tokens == 242  # candidates + thoughts
    assert r.finish_reasons == ["stop", "stop"]
    assert r.resolved_model == "gemini-2.5-flash-001"


async def test_classify_logprobs_first_token():
    lr = NS(top_candidates=[NS(candidates=[NS(token="Land", log_probability=-0.2),
                                           NS(token="Water", log_probability=-1.8)])],
            chosen_candidates=[NS(token="Land", log_probability=-0.2)])
    a = _adapter("gemini-2.0-flash", responses=[_resp("Land", logprobs=lr)])
    r = await a.classify("p", None, CallParams(temperature=0.0, max_output_tokens=4,
                                               logprobs=True, top_logprobs=5))
    cfg = a._client.aio.models.calls[0]["config"]
    assert cfg["response_logprobs"] is True and cfg["logprobs"] == 5
    assert r.first_token_logprobs == {"Land": -0.2, "Water": -1.8}


async def test_blocked_prompt_is_empty_answer():
    blocked = NS(candidates=[], usage_metadata=None, model_version="m",
                 prompt_feedback=NS(block_reason=NS(name="SAFETY")))
    a = _adapter(responses=[blocked])
    r = await a.classify("p", None, CallParams(temperature=None, max_output_tokens=4))
    assert r.texts == [""] and r.finish_reasons == ["blocked:safety"]


class FakeAPIError(Exception):
    def __init__(self, code, msg="err", details=None):
        super().__init__(msg)
        self.code = code
        self.details = details


def test_error_mapping():
    e = map_google_error(FakeAPIError(429, details={"retryDelay": "17s"}))
    assert isinstance(e, RateLimitError) and e.retry_after_s == 17.0
    e = map_google_error(FakeAPIError(503))
    assert isinstance(e, ProviderError) and e.retryable
    e = map_google_error(FakeAPIError(400))
    assert isinstance(e, ProviderError) and not e.retryable
    e = map_google_error(TimeoutError("slow"))
    assert isinstance(e, ProviderError) and e.retryable
