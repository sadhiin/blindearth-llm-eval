"""Capability probe: measure what an endpoint actually supports (provider docs are not trusted).

Budget of `n_calls` (default 20) real calls, spent in this order:

1. baseline call (must succeed, else the probe raises the ProviderError);
2. logprobs with top_logprobs=5;
3. top-N cap: top_logprobs 20, then 10 (only if logprobs worked);
4. one call per effort level off/low/medium/high/max that the adapter's own mapping accepts
   (levels its `map_config` refuses are recorded as unsupported without a call);
5. the remaining calls as one concurrent burst to guess safe concurrency
   (no 429 -> burst size, or the adapter's static guess if larger; any 429 -> half the
   number of successes, at least 1).

Batch support is taken from the adapter (a batch round trip can take hours, so it is not
exercised). `temperature_fixed_with_thinking` comes from the adapter's static profile.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from blindearth.providers.base import (
    Adapter,
    ProviderError,
    RateLimitError,
    UnsupportedConfigError,
)
from blindearth.types import CallParams, Capabilities, ExtractionMode, RunConfig

PROBE_PROMPTS = [
    "If this location is over land, say 'Land'. If this location is over water, say 'Water'. "
    f"Do not say anything else. {c}"
    for c in ("49° N, 3° E", "31° S, 121° W", "1° N, 25° E", "41° N, 29° W")
]
EFFORT_LEVELS = ("off", "low", "medium", "high", "max")
TOP_N_CANDIDATES = (20, 10)
PROBE_MAX_TOKENS = 16


class _Budget:
    def __init__(self, n: int):
        self.left = n
        self.used = 0
        self.rate_limited = 0

    def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        self.used += 1
        return True


async def _call(adapter: Adapter, budget: _Budget, params: CallParams, i: int = 0):
    """-> (ok, result_or_exception). Returns (False, None) when the budget is exhausted."""
    if not budget.take():
        return False, None
    try:
        res = await adapter.classify(PROBE_PROMPTS[i % len(PROBE_PROMPTS)], None, params)
    except RateLimitError as e:
        budget.rate_limited += 1
        return False, e
    except ProviderError as e:
        return False, e
    if res.error:
        return False, ProviderError(res.error)
    return True, res


def _params(**kw) -> CallParams:
    base = dict(temperature=None, max_output_tokens=PROBE_MAX_TOKENS, n=1)
    base.update(kw)
    return CallParams(**base)


async def probe(adapter: Adapter, n_calls: int = 20) -> Capabilities:
    static = adapter.default_capabilities()
    saved_caps = adapter.capabilities
    adapter.capabilities = None  # let map_config judge by the static profile, not an old probe
    budget = _Budget(max(1, n_calls))
    notes: list[str] = []
    try:
        # 1. baseline
        ok, res = await _call(adapter, budget, _params())
        if not ok:
            if isinstance(res, ProviderError):
                raise res
            raise ProviderError("probe: baseline call failed", retryable=False)
        notes.append(f"baseline ok, resolved_model={res.resolved_model}, "
                     f"latency={res.latency_s:.2f}s")

        # 2. logprobs
        logprobs = False
        top_max: int | None = None
        ok, res = await _call(adapter, budget, _params(logprobs=True, top_logprobs=5), 1)
        if ok and res.first_token_logprobs:
            logprobs = True
            top_max = 5
            notes.append(f"logprobs ok ({len(res.first_token_logprobs)} alternatives at k=5)")
        elif res is not None:
            notes.append(f"logprobs unsupported: {res if not ok else 'empty logprobs'}")

        # 3. top-N cap
        if logprobs:
            if static.top_logprobs_max is None and adapter.kind == "transformers":
                top_max = None  # full vocabulary
                notes.append("top-N unbounded (full vocabulary)")
            else:
                for k in TOP_N_CANDIDATES:
                    ok, res = await _call(adapter, budget,
                                          _params(logprobs=True, top_logprobs=k), 2)
                    if res is None:
                        break
                    if ok and res.first_token_logprobs:
                        got = len(res.first_token_logprobs)
                        top_max = k if got >= k // 2 else max(top_max or 0, got)
                        notes.append(f"top_logprobs={k} accepted ({got} returned)")
                        break
                    notes.append(f"top_logprobs={k} rejected")

        # 4. effort levels
        accepted: list[str] = []
        for lvl in EFFORT_LEVELS:
            try:
                adapter.map_config(RunConfig(effort=lvl), ExtractionMode.SAMPLE)
            except UnsupportedConfigError as e:
                notes.append(f"effort={lvl}: refused by mapping ({e})")
                continue
            ok, res = await _call(adapter, budget, _params(effort=lvl), 3)
            if res is None:
                notes.append(f"effort={lvl}: not probed (budget exhausted)")
                continue
            if ok:
                accepted.append(lvl)
            else:
                notes.append(f"effort={lvl}: rejected by endpoint ({res})")
        native_effort = any(lvl != "off" for lvl in accepted)

        # 5. concurrency burst
        burst = budget.left
        max_conc = static.max_concurrency
        if burst > 0:
            before = budget.rate_limited
            results = await asyncio.gather(
                *[_call(adapter, budget, _params(), i) for i in range(burst)])
            n_ok = sum(1 for ok, _ in results if ok)
            n_429 = budget.rate_limited - before
            if n_429 == 0 and n_ok == burst:
                max_conc = max(burst, static.max_concurrency or burst)
                notes.append(f"burst of {burst} concurrent calls ok")
            else:
                max_conc = max(1, n_ok // 2)
                notes.append(f"burst of {burst}: {n_ok} ok, {n_429} rate-limited")

        return Capabilities(
            logprobs=logprobs,
            top_logprobs_max=top_max if logprobs else None,
            effort_param=native_effort,
            supported_efforts=accepted,
            temperature_fixed_with_thinking=static.temperature_fixed_with_thinking,
            max_concurrency=max_conc,
            supports_batch=bool(adapter.supports_batch),
            probed_at=datetime.now(timezone.utc).isoformat(),
            notes=notes + [f"{budget.used} probe calls"],
        )
    finally:
        adapter.capabilities = saved_caps


__all__ = ["probe", "PROBE_PROMPTS", "EFFORT_LEVELS"]
