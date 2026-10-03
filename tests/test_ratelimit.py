from __future__ import annotations

import asyncio
import random
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import pytest

from blindearth.providers.base import ProviderError, RateLimitError
from blindearth.ratelimit import (
    CircuitBreaker,
    ProviderLimiter,
    TokenBucket,
    backoff_delay,
    call_with_retries,
    parse_retry_after,
)


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# --------------------------------------------------------------------------- retry-after


def test_parse_retry_after_seconds_and_garbage():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("soon") is None


def test_parse_retry_after_http_date():
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    v = parse_retry_after(format_datetime(when, usegmt=True))
    assert v is not None and 25 <= v <= 31


# --------------------------------------------------------------------------- token bucket


def test_token_bucket_refills_over_time():
    clock = FakeClock()
    b = TokenBucket(60, clock=clock)  # 1 token / s
    assert b.try_take(60) == 0.0
    wait = b.try_take(1)
    assert wait == pytest.approx(1.0)
    clock.t = 2.0
    assert b.try_take(1) == 0.0
    assert b.available == pytest.approx(1.0)


def test_token_bucket_oversize_request_capped_at_capacity():
    clock = FakeClock()
    b = TokenBucket(10, clock=clock)
    assert b.try_take(1000) == 0.0  # capped, does not deadlock


# --------------------------------------------------------------------------- limiter


def test_limiter_halves_on_429_and_recovers_additively():
    lim = ProviderLimiter(concurrency=16, rpm=None, tpm=None)
    lim.on_rate_limited(None)
    assert lim.limit == 8
    lim.on_rate_limited(None)
    assert lim.limit == 4
    for _ in range(3):
        lim.on_success()
    assert lim.limit == 4  # needs `limit` successes for +1
    lim.on_success()
    assert lim.limit == 5
    for _ in range(100):
        lim.on_success()
    assert lim.limit == 16  # never above the configured concurrency


def test_limiter_never_below_one():
    lim = ProviderLimiter(concurrency=1, rpm=None, tpm=None)
    lim.on_rate_limited(None)
    assert lim.limit == 1


async def test_limiter_caps_concurrency():
    lim = ProviderLimiter(concurrency=2, rpm=None, tpm=None)
    active = 0
    peak = 0

    async def work():
        nonlocal active, peak
        async with lim.slot(10):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*[work() for _ in range(6)])
    assert peak == 2
    assert lim.in_flight == 0


async def test_limiter_rpm_bucket_consumed():
    lim = ProviderLimiter(concurrency=4, rpm=600, tpm=10_000)
    async with lim.slot(100):
        pass
    assert lim._rpm_bucket.available < 600
    assert lim._tpm_bucket.available <= 9_900 + 1


# --------------------------------------------------------------------------- circuit breaker


def test_circuit_breaker_opens_on_consecutive_failures():
    clock = FakeClock()
    cb = CircuitBreaker(threshold=3, window_s=60, clock=clock)
    cb.record(False)
    cb.record(False)
    assert not cb.open
    cb.record(True)  # success resets the streak
    cb.record(False)
    cb.record(False)
    assert not cb.open
    cb.record(False)
    assert cb.open
    cb.record(True)  # latched
    assert cb.open
    cb.reset()
    assert not cb.open


def test_circuit_breaker_window_expires_old_failures():
    clock = FakeClock()
    cb = CircuitBreaker(threshold=3, window_s=10, clock=clock)
    cb.record(False)
    cb.record(False)
    clock.t = 30
    cb.record(False)
    assert not cb.open


# --------------------------------------------------------------------------- retries


def test_backoff_delay_bounds():
    rng = random.Random(0)
    for attempt in range(8):
        d = backoff_delay(attempt, base=1.0, cap=60.0, rng=rng)
        ceiling = min(60.0, 2 ** attempt)
        assert ceiling / 2 <= d <= ceiling


async def test_call_with_retries_retries_then_succeeds():
    calls = 0
    sleeps: list[float] = []

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("boom", status=503, retryable=True)
        return "ok"

    async def fake_sleep(s):
        sleeps.append(s)

    lim = ProviderLimiter(4, None, None)
    assert await call_with_retries(fn, limiter=lim, sleep=fake_sleep) == "ok"
    assert calls == 3
    assert len(sleeps) == 2


async def test_call_with_retries_non_retryable_raises_immediately():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ProviderError("bad key", status=401, retryable=False)

    async def fake_sleep(s):
        raise AssertionError("should not sleep")

    with pytest.raises(ProviderError):
        await call_with_retries(fn, sleep=fake_sleep)
    assert calls == 1


async def test_call_with_retries_honours_retry_after_and_halves_limiter():
    calls = 0
    sleeps: list[float] = []

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("slow down", retry_after_s=12.0)
        return 1

    async def fake_sleep(s):
        sleeps.append(s)

    lim = ProviderLimiter(8, None, None)
    await call_with_retries(fn, limiter=lim, sleep=fake_sleep, rng=random.Random(1))
    assert sleeps and sleeps[0] >= 12.0
    assert lim.limit == 4
    assert lim.n_rate_limited == 1


async def test_call_with_retries_gives_up_after_max_attempts():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ProviderError("timeout", retryable=True)

    async def fake_sleep(s):
        return None

    with pytest.raises(ProviderError):
        await call_with_retries(fn, max_attempts=5, sleep=fake_sleep)
    assert calls == 5
