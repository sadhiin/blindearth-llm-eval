from __future__ import annotations

import asyncio

import pytest

from blindearth.providers.base import (
    Adapter,
    ProviderError,
    RateLimitError,
    UnsupportedConfigError,
)
from blindearth.providers.probe import probe
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    ModelSpec,
    ProviderSpec,
    RunConfig,
    Usage,
)


class FakeAdapter(Adapter):
    kind = "openai_compatible"

    def __init__(self, *, logprobs=True, top_cap=20, mapped_efforts=("off", "low", "medium",
                                                                     "high", "max"),
                 accepted_efforts=("low", "medium", "high"), burst_limit=None,
                 baseline_error=None, supports_batch=False):
        super().__init__(ProviderSpec(id="p", kind="openai_compatible", base_url="http://x"),
                         ModelSpec(id="m", provider="p", name="m"))
        self.lp = logprobs
        self.top_cap = top_cap
        self.mapped = set(mapped_efforts)
        self.accepted = set(accepted_efforts)
        self.burst_limit = burst_limit
        self.baseline_error = baseline_error
        self.supports_batch = supports_batch
        self.calls: list[CallParams] = []
        self.in_flight = 0
        self.seen_caps = []

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict:
        self.seen_caps.append(self.capabilities)
        if config.effort is not None and config.effort not in self.mapped:
            raise UnsupportedConfigError(config.effort)
        return {}

    def default_capabilities(self) -> Capabilities:
        return Capabilities(max_concurrency=4, temperature_fixed_with_thinking=True)

    async def classify(self, prompt, system_prompt, params):
        self.calls.append(params)
        if self.baseline_error and len(self.calls) == 1:
            raise self.baseline_error
        if params.effort is not None and params.effort not in self.accepted:
            raise ProviderError("400 unknown effort", status=400)
        lp = None
        if params.logprobs:
            if not self.lp:
                lp = None
            elif params.top_logprobs > self.top_cap:
                raise ProviderError("400 top_logprobs too large", status=400)
            else:
                lp = {f"t{i}": -float(i) for i in range(params.top_logprobs)}
        self.in_flight += 1
        try:
            await asyncio.sleep(0)
            if self.burst_limit is not None and self.in_flight > self.burst_limit:
                raise RateLimitError("429", retry_after_s=1)
        finally:
            self.in_flight -= 1
        return ClassifyResult(texts=["Land"], usage=Usage(10, 1), latency_s=0.01,
                              first_token_logprobs=lp, resolved_model="m-v1")


async def test_probe_full_budget_and_results():
    a = FakeAdapter(top_cap=10)
    a.capabilities = Capabilities(probed_at="old", logprobs=False)
    caps = await probe(a, n_calls=20)
    assert len(a.calls) == 20
    assert caps.logprobs and caps.top_logprobs_max == 10
    assert caps.supported_efforts == ["low", "medium", "high"]
    assert caps.effort_param
    assert caps.temperature_fixed_with_thinking
    assert caps.probed_at and caps.probed_at != "old"
    assert caps.supports_batch is False
    # map_config was consulted without the stale probe result
    assert all(c is None for c in a.seen_caps)
    assert a.capabilities.probed_at == "old"  # restored afterwards
    # 1 baseline + 1 logprobs + 2 top-N + 5 efforts = 9; the other 11 are the burst
    assert caps.max_concurrency == 11


async def test_probe_no_logprobs_skips_top_n_and_refused_efforts_cost_nothing():
    a = FakeAdapter(logprobs=False, mapped_efforts=("low",), accepted_efforts=("low",),
                    supports_batch=True)
    caps = await probe(a, n_calls=20)
    assert not caps.logprobs and caps.top_logprobs_max is None
    assert caps.supported_efforts == ["low"]
    assert caps.supports_batch
    assert len(a.calls) == 20
    efforts_called = [p.effort for p in a.calls if p.effort]
    assert efforts_called == ["low"]
    assert any("refused by mapping" in n for n in caps.notes)


async def test_probe_burst_rate_limited_lowers_concurrency():
    a = FakeAdapter(burst_limit=3)
    caps = await probe(a, n_calls=20)
    assert caps.max_concurrency is not None and caps.max_concurrency <= 3


async def test_probe_baseline_failure_raises():
    a = FakeAdapter(baseline_error=ProviderError("401 bad key", status=401))
    with pytest.raises(ProviderError):
        await probe(a, n_calls=20)


async def test_probe_small_budget_is_respected():
    a = FakeAdapter()
    caps = await probe(a, n_calls=3)
    assert len(a.calls) == 3
    assert caps.logprobs
