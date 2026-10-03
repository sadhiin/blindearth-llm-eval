"""OpenAI-compatible, OpenRouter, Ollama and llama.cpp adapters over httpx (mocked with respx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from blindearth.providers.base import ProviderError, RateLimitError, UnsupportedConfigError
from blindearth.providers.local import LlamaCppAdapter, OllamaAdapter
from blindearth.providers.openai_compat import (
    OpenAICompatAdapter,
    first_token_logprobs_openai,
    split_think,
)
from blindearth.providers.openrouter import OpenRouterAdapter
from blindearth.types import CallParams, Capabilities, ExtractionMode, ModelSpec, ProviderSpec, RunConfig

VLLM = ProviderSpec(id="my-vllm", kind="openai_compatible", base_url="http://localhost:8000/v1")
URL = "http://localhost:8000/v1/chat/completions"


def _model(name="Qwen/Qwen2.5-7B-Instruct", provider="my-vllm", **extra):
    return ModelSpec(id="m", provider=provider, name=name, extra=extra)


def _body(contents=("Land",), *, logprobs=None, model="Qwen/Qwen2.5-7B-Instruct",
          reasoning_tokens=None, reasoning=None, **top):
    choices = []
    for i, c in enumerate(contents):
        msg = {"role": "assistant", "content": c}
        if reasoning:
            msg["reasoning_content"] = reasoning
        choices.append({"index": i, "message": msg, "finish_reason": "stop",
                        "logprobs": logprobs if i == 0 else None})
    usage = {"prompt_tokens": 40, "completion_tokens": 3 * len(contents)}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {"id": "x", "model": model, "choices": choices, "usage": usage, **top}


def P(**kw):
    base = dict(temperature=1.0, max_output_tokens=8)
    base.update(kw)
    return CallParams(**base)


# --------------------------------------------------------------------------- helpers


def test_split_think_variants():
    assert split_think("<think>hmm</think>\nLand") == ("Land", "hmm")
    assert split_think("reasoning here</think>Water") == ("Water", "reasoning here")
    assert split_think("<think>never closed") == ("", "never closed")
    assert split_think("Land.") == ("Land.", None)
    assert split_think(None) == ("", None)


def test_first_token_logprobs_skips_think_block_and_whitespace():
    lp = {"content": [
        {"token": "<think>", "logprob": 0.0, "top_logprobs": []},
        {"token": "x", "logprob": 0.0, "top_logprobs": []},
        {"token": "</think>", "logprob": 0.0, "top_logprobs": []},
        {"token": "\n\n", "logprob": 0.0, "top_logprobs": []},
        {"token": "Water", "logprob": -0.1,
         "top_logprobs": [{"token": "Water", "logprob": -0.1}, {"token": "Land", "logprob": -2.4}]},
    ]}
    assert first_token_logprobs_openai(lp) == {"Water": -0.1, "Land": -2.4}
    assert first_token_logprobs_openai(None) is None


# --------------------------------------------------------------------------- compat adapter


@respx.mock
async def test_compat_logprobs_request_and_parse():
    lp = {"content": [{"token": "Land", "logprob": -0.3,
                       "top_logprobs": [{"token": "Land", "logprob": -0.3},
                                        {"token": "Water", "logprob": -1.4},
                                        {"token": " Land", "logprob": -5.0}]}]}
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_body(logprobs=lp)))
    a = OpenAICompatAdapter(VLLM, _model(), api_key="secret")
    r = await a.classify("12° S, 45° W", "sys", P(temperature=0.0, logprobs=True, top_logprobs=20,
                                                   seed=3))
    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == "Bearer secret"
    assert sent["logprobs"] is True and sent["top_logprobs"] == 20 and sent["seed"] == 3
    assert sent["max_tokens"] == 8 and sent["temperature"] == 0.0
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert r.first_token_logprobs == {"Land": -0.3, "Water": -1.4, " Land": -5.0}
    assert r.resolved_model == "Qwen/Qwen2.5-7B-Instruct"
    assert r.native_params["top_logprobs"] == 20 and "messages" not in r.native_params
    await a.aclose()


@respx.mock
async def test_compat_n_in_one_request_and_think_stripping():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_body(
        ("<think>coast</think>Land", "Water", "Land", "Land"), reasoning_tokens=50)))
    a = OpenAICompatAdapter(VLLM, _model())
    r = await a.classify("p", None, P(n=4, store_thinking=True))
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content)["n"] == 4
    assert r.texts == ["Land", "Water", "Land", "Land"]
    assert r.thinking_texts[0] == "coast" and r.thinking_texts[1] is None
    assert r.usage.thinking_tokens == 50
    assert "authorization" not in route.calls[0].request.headers


@respx.mock
async def test_compat_reasoning_content_kept_only_with_store_thinking():
    respx.post(URL).mock(return_value=httpx.Response(200, json=_body(reasoning="deep thought")))
    a = OpenAICompatAdapter(VLLM, _model())
    r = await a.classify("p", None, P())
    assert r.thinking_texts == [None] and r.texts == ["Land"]
    r = await a.classify("p", None, P(store_thinking=True))
    assert r.thinking_texts == ["deep thought"]


@respx.mock
async def test_compat_http_errors():
    respx.post(URL).mock(side_effect=[
        httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow"}),
        httpx.Response(503, text="down"),
        httpx.Response(401, text="no"),
        httpx.ConnectTimeout("t"),
        httpx.ConnectError("refused"),
    ])
    a = OpenAICompatAdapter(VLLM, _model())
    with pytest.raises(RateLimitError) as ei:
        await a.classify("p", None, P())
    assert ei.value.retry_after_s == 3.0
    with pytest.raises(ProviderError) as ei:
        await a.classify("p", None, P())
    assert ei.value.retryable and ei.value.status == 503
    with pytest.raises(ProviderError) as ei:
        await a.classify("p", None, P())
    assert not ei.value.retryable and ei.value.status == 401
    for _ in range(2):
        with pytest.raises(ProviderError) as ei:
            await a.classify("p", None, P())
        assert ei.value.retryable


def test_compat_effort_mapping_defaults_and_override():
    a = OpenAICompatAdapter(VLLM, _model())
    assert a.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)["reasoning_effort"] == "low"
    for lvl in ("off", "max"):
        with pytest.raises(UnsupportedConfigError):
            a.map_config(RunConfig(effort=lvl), ExtractionMode.SAMPLE)
    b = OpenAICompatAdapter(VLLM, _model(effort_map={
        "off": {"chat_template_kwargs": {"enable_thinking": False}}, "low": None}))
    n = b.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)
    assert n["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(UnsupportedConfigError):
        b.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)


def test_compat_probed_capabilities_gate_config():
    a = OpenAICompatAdapter(VLLM, _model())
    a.capabilities = Capabilities(logprobs=False, effort_param=False,
                                  probed_at="2026-10-01T00:00:00+00:00")
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(), ExtractionMode.LOGPROBS)
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)


def test_compat_thinking_raises_max_tokens():
    a = OpenAICompatAdapter(VLLM, _model())
    n = a.map_config(RunConfig(effort="high", max_output_tokens=16), ExtractionMode.SAMPLE)
    assert n["max_tokens"] >= 16000


# --------------------------------------------------------------------------- local servers


@respx.mock
async def test_ollama_loops_samples_and_default_url():
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body(("Land",), model="llama3.2:3b")))
    a = OllamaAdapter(ProviderSpec(id="ollama", kind="ollama"), _model("llama3.2:3b", "ollama"))
    r = await a.classify("p", None, P(n=3, effort="off"))
    assert route.call_count == 3
    sent = json.loads(route.calls[0].request.content)
    assert "n" not in sent and sent["reasoning_effort"] == "none"
    assert r.texts == ["Land"] * 3 and r.native_params["n_requests"] == 3
    assert r.usage.input_tokens == 120


@respx.mock
async def test_llamacpp_off_uses_chat_template_kwargs():
    route = respx.post("http://localhost:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body(("Water",))))
    a = LlamaCppAdapter(ProviderSpec(id="lc", kind="llamacpp"), _model("qwen3", "lc"))
    await a.classify("p", None, P(effort="off"))
    sent = json.loads(route.calls[0].request.content)
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(effort="high"), ExtractionMode.SAMPLE)


# --------------------------------------------------------------------------- openrouter


@respx.mock
async def test_openrouter_reasoning_and_require_parameters():
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body(
            ("Land",), model="anthropic/claude-sonnet-5-5", provider="Anthropic",
            logprobs=None)))
    a = OpenRouterAdapter(ProviderSpec(id="or", kind="openrouter"),
                          _model("anthropic/claude-sonnet-5-5", "or"), api_key="k")
    r = await a.classify("p", None, P(effort="max"))
    sent = json.loads(route.calls[0].request.content)
    assert sent["reasoning"] == {"max_tokens": 32000}
    assert route.calls[0].request.headers["x-title"] == "blindearth"
    assert r.resolved_model == "anthropic/claude-sonnet-5-5@Anthropic"

    await a.classify("p", None, P(effort="off", logprobs=True, top_logprobs=5))
    sent = json.loads(route.calls[1].request.content)
    assert sent["reasoning"] == {"enabled": False}
    assert sent["provider"] == {"require_parameters": True}


def test_openrouter_effort_table():
    a = OpenRouterAdapter(ProviderSpec(id="or", kind="openrouter"), _model("x/y", "or"))
    assert a.map_config(RunConfig(effort="medium"), ExtractionMode.SAMPLE)["reasoning"] == {
        "effort": "medium"}
    with pytest.raises(UnsupportedConfigError):
        a.map_config(RunConfig(), ExtractionMode.LOGPROBS)  # unprobed: logprobs unknown


def test_missing_base_url_is_an_error():
    with pytest.raises(ValueError):
        OpenAICompatAdapter(ProviderSpec(id="x", kind="openai_compatible"), _model())
