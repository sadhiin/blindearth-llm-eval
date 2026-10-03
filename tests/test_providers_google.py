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


class FakeBatches:
    def __init__(self):
        self.created = []
        self.jobs = {}

    async def create(self, *, model, src, config=None):
        name = f"batches/{len(self.created)}"
        self.created.append({"model": model, "src": src, "config": config})
        self.jobs[name] = NS(name=name, state=NS(name="JOB_STATE_PENDING"), dest=None, error=None)
        return self.jobs[name]

    async def get(self, *, name):
        return self.jobs[name]


class FakeClient:
    def __init__(self, responses=()):
        self.aio = NS(models=FakeModels(responses), batches=FakeBatches())


def _adapter(name="gemini-2.5-flash", responses=(), extra=None, prov=PROV):
    return GoogleAdapter(prov, ModelSpec(id=name, provider="google", name=name,
                                         extra=extra or {}),
                         api_key="k", client=FakeClient(responses))


def test_family_detection():
    assert gemini_family("gemini-2.5-pro") == "g25pro"
    assert gemini_family("gemini-2.5-flash-lite") == "g25flashlite"
    assert gemini_family("gemini-2.5-flash") == "g25flash"
    assert gemini_family("gemini-3-pro-preview") == "g3"
    assert gemini_family("gemini-flash-latest") == "g3"
    assert gemini_family("gemini-2.0-flash") == "g2"


def test_gemini3_levels_per_model():
    g3pro = _adapter("gemini-3-pro-preview")
    with pytest.raises(UnsupportedConfigError):  # 3 Pro: low/high only
        g3pro.map_config(RunConfig(effort="medium"), ExtractionMode.SAMPLE)
    with pytest.raises(UnsupportedConfigError):
        g3pro.map_config(RunConfig(effort="minimal"), ExtractionMode.SAMPLE)
    assert g3pro.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_level": "LOW"}
    flash = _adapter("gemini-3-flash-preview")
    assert flash.map_config(RunConfig(effort="minimal"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_level": "MINIMAL"}
    f38 = _adapter("gemini-3.8-flash")
    with pytest.raises(UnsupportedConfigError):
        f38.map_config(RunConfig(effort="minimal"), ExtractionMode.SAMPLE)
    assert "minimal" not in f38.default_capabilities().supported_efforts
    # explicit override is trusted
    ov = _adapter("gemini-3.8-flash", extra={"thinking_levels": ["minimal", "low"]})
    assert ov.map_config(RunConfig(effort="minimal"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_level": "MINIMAL"}


def test_flash_lite_off_by_default_and_budget_checks():
    lite = _adapter("gemini-2.5-flash-lite")
    assert "max_output_tokens" not in lite.map_config(RunConfig(), ExtractionMode.SAMPLE)
    assert lite.map_config(RunConfig(effort="off"), ExtractionMode.SAMPLE)[
        "thinking_config"] == {"thinking_budget": 0}
    assert not lite._thinks(None, None)
    assert _adapter("gemini-2.5-flash")._thinks(None, None)
    assert _adapter("gemini-3-pro-preview")._thinks(None, None)
    forced = _adapter("gemini-2.5-flash-lite", extra={"thinks_by_default": True})
    assert "max_output_tokens" in forced.map_config(RunConfig(), ExtractionMode.SAMPLE)
    bad = _adapter("gemini-2.5-pro", extra={"thinking_levels": None})
    bad.profile = dict(bad.profile, map={**bad.profile["map"], "tiny": {"thinking_budget": 64}})
    with pytest.raises(UnsupportedConfigError):  # below 2.5 Pro minimum of 128
        bad.map_config(RunConfig(effort="tiny"), ExtractionMode.SAMPLE)
    both = _adapter("gemini-3-flash-preview", extra={
        "effort_map": {"low": {"thinking_level": "LOW", "thinking_budget": 512}}})
    with pytest.raises(UnsupportedConfigError):
        both.map_config(RunConfig(effort="low"), ExtractionMode.SAMPLE)


async def test_batch_submit_uses_same_config_and_metadata_keys():
    a = _adapter()
    assert a.supports_batch and a.default_capabilities().supports_batch
    params = CallParams(temperature=1.0, max_output_tokens=16, effort="low", n=2)
    bid = await a.submit_batch([("p1", "prompt1", "sys", params), ("p2", "prompt2", "sys", params)])
    created = a._client.aio.batches.created
    assert bid == "batches/0" and len(created) == 1
    src = created[0]["src"]
    assert [r["metadata"]["key"] for r in src] == ["p1__s0", "p1__s1", "p2__s0", "p2__s1"]
    assert src[0]["contents"] == "prompt1"
    assert src[0]["config"] == a._config("sys", params)
    assert created[0]["model"] == "gemini-2.5-flash"


async def test_batch_poll_running_then_results():
    a = _adapter()
    params = CallParams(temperature=1.0, max_output_tokens=16, n=2)
    bid = await a.submit_batch([("p1", "x", None, params), ("p2", "y", None, params)])
    assert await a.poll_batch(bid) is None
    job = a._client.aio.batches.jobs[bid]
    job.state = NS(name="JOB_STATE_SUCCEEDED")
    job.dest = NS(inlined_responses=[
        NS(response=_resp("Land"), metadata={"key": "p1__s0"}, error=None),
        # metadata missing: falls back to submit order (index 1 -> p1__s1)
        NS(response=_resp("Water"), metadata=None, error=None),
        NS(response=None, metadata={"key": "p2__s0"}, error=NS(code=500, message="boom")),
        NS(response=_resp("Land"), metadata={"key": "p2__s1"}, error=None),
    ])
    out = await a.poll_batch(bid)
    assert out["p1"].texts == ["Land", "Water"] and out["p1"].error is None
    assert out["p1"].usage.input_tokens == 96
    assert out["p2"].texts == ["Land"]
    assert "1/2 batch samples failed" in out["p2"].error and "boom" in out["p2"].error


async def test_batch_job_failure_marks_known_items():
    a = _adapter()
    bid = await a.submit_batch([("p7", "x", None, CallParams(temperature=None,
                                                             max_output_tokens=4))])
    job = a._client.aio.batches.jobs[bid]
    job.state = NS(name="JOB_STATE_EXPIRED")
    out = await a.poll_batch(bid)
    assert out["p7"].error and "JOB_STATE_EXPIRED" in out["p7"].error


async def test_batch_splits_large_submissions(monkeypatch):
    import blindearth.providers.google as g

    monkeypatch.setattr(g, "INLINE_BATCH_MAX_BYTES", 400)
    a = _adapter()
    p = CallParams(temperature=None, max_output_tokens=4)
    bid = await a.submit_batch([(f"p{i}", "x" * 150, None, p) for i in range(4)])
    names = bid.split(",")
    assert len(names) > 1
    for n in names[:-1]:
        a._client.aio.batches.jobs[n].state = NS(name="JOB_STATE_SUCCEEDED")
    assert await a.poll_batch(bid) is None  # last job still pending


def test_vertex_has_no_batch():
    vprov = ProviderSpec(id="vx", kind="google", extra={"vertexai": True})
    a = _adapter(prov=vprov)
    assert not a.supports_batch and not a.default_capabilities().supports_batch


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
