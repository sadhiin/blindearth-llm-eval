"""Per-provider rate limiting, retries and a circuit breaker.

- `ProviderLimiter`: token buckets for requests/min and tokens/min plus an adaptive concurrency
  cap. A 429 halves the cap (multiplicative decrease) and pauses new slots for `retry-after`;
  every `cap` consecutive successes add one slot back (slow additive recovery) up to the
  configured concurrency.
- `call_with_retries`: exponential backoff with full jitter, at most `max_attempts` attempts,
  only for retryable `ProviderError`s; honours `RateLimitError.retry_after_s`.
  It does NOT acquire a limiter slot: `fn` should do `async with limiter.slot(est): ...`
  itself, so every attempt (including retries) is rate limited.
- `CircuitBreaker`: opens after `threshold` consecutive failures inside `window_s` and stays
  open (latched) until `reset()`, so a bad key or a dead endpoint pauses the cell.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from blindearth.providers.base import ProviderError, RateLimitError

T = TypeVar("T")

__all__ = [
    "TokenBucket",
    "ProviderLimiter",
    "CircuitBreaker",
    "call_with_retries",
    "backoff_delay",
    "parse_retry_after",
]


def parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` header (seconds or HTTP date) -> seconds, or None."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return max(0.0, dt.timestamp() - time.time())


class TokenBucket:
    """Classic token bucket refilled continuously at `per_minute / 60` units per second."""

    def __init__(self, per_minute: float, *, clock: Callable[[], float] = time.monotonic):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.capacity = float(per_minute)
        self.rate = float(per_minute) / 60.0
        self._tokens = float(per_minute)
        self._clock = clock
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def try_take(self, amount: float) -> float:
        """Take `amount` if available and return 0.0, else return the seconds to wait."""
        amount = min(float(amount), self.capacity)  # an oversize request still gets through
        self._refill()
        if self._tokens >= amount:
            self._tokens -= amount
            return 0.0
        return (amount - self._tokens) / self.rate

    async def take(self, amount: float) -> None:
        async with self._lock:
            while True:
                wait = self.try_take(amount)
                if wait <= 0:
                    return
                await asyncio.sleep(wait)


class ProviderLimiter:
    """Concurrency + rpm + tpm limiter for one provider (shared by its cells)."""

    def __init__(self, concurrency: int, rpm: int | None, tpm: int | None):
        self.max_concurrency = max(1, int(concurrency))
        self.limit = self.max_concurrency  # current adaptive cap
        self.rpm = rpm
        self.tpm = tpm
        self._rpm_bucket = TokenBucket(rpm) if rpm else None
        self._tpm_bucket = TokenBucket(tpm) if tpm else None
        self._in_flight = 0
        self._cond = asyncio.Condition()
        self._pause_until = 0.0
        self._successes = 0
        self.n_rate_limited = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def _wait_pause(self) -> None:
        while True:
            delay = self._pause_until - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    @asynccontextmanager
    async def _slot(self, est_tokens: int) -> AsyncIterator[None]:
        await self._wait_pause()
        async with self._cond:
            await self._cond.wait_for(lambda: self._in_flight < self.limit)
            self._in_flight += 1
        try:
            if self._rpm_bucket is not None:
                await self._rpm_bucket.take(1)
            if self._tpm_bucket is not None and est_tokens > 0:
                await self._tpm_bucket.take(est_tokens)
            yield
        finally:
            async with self._cond:
                self._in_flight -= 1
                self._cond.notify_all()

    def slot(self, est_tokens: int):  # -> AsyncContextManager[None]
        return self._slot(est_tokens)

    def on_rate_limited(self, retry_after_s: float | None) -> None:
        self.n_rate_limited += 1
        self._successes = 0
        self.limit = max(1, self.limit // 2)
        if retry_after_s and retry_after_s > 0:
            self._pause_until = max(self._pause_until, time.monotonic() + retry_after_s)

    def on_success(self) -> None:
        if self.limit >= self.max_concurrency:
            self._successes = 0
            return
        self._successes += 1
        if self._successes >= self.limit:  # +1 slot per "window" of successes
            self._successes = 0
            self.limit = min(self.max_concurrency, self.limit + 1)
            self._notify_soon()

    def _notify_soon(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        loop.create_task(_notify())


class CircuitBreaker:
    """Opens after `threshold` consecutive failures within `window_s` seconds. Latched."""

    def __init__(self, threshold: int = 20, window_s: float = 60.0, *,
                 clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.window_s = window_s
        self._clock = clock
        self._failures: deque[float] = deque()
        self._open = False

    def record(self, ok: bool) -> None:
        if self._open:
            return
        now = self._clock()
        if ok:
            self._failures.clear()
            return
        self._failures.append(now)
        while self._failures and now - self._failures[0] > self.window_s:
            self._failures.popleft()
        if len(self._failures) >= self.threshold:
            self._open = True

    @property
    def open(self) -> bool:
        return self._open

    def reset(self) -> None:
        self._open = False
        self._failures.clear()


def backoff_delay(attempt: int, *, base: float = 1.0, cap: float = 60.0,
                  rng: random.Random | None = None) -> float:
    """Full-jitter exponential backoff for retry number `attempt` (0-based)."""
    r = rng or random
    ceiling = min(cap, base * (2 ** attempt))
    return r.uniform(ceiling / 2, ceiling)


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    limiter: ProviderLimiter | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run `fn` with retries on retryable ProviderErrors (429, 5xx, timeouts).

    Non-retryable errors are raised immediately. After the last attempt the last error is raised.
    """
    attempt = 0
    while True:
        try:
            result = await fn()
        except RateLimitError as e:
            if limiter is not None:
                limiter.on_rate_limited(e.retry_after_s)
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = backoff_delay(attempt - 1, base=base_delay, cap=max_delay, rng=rng)
            if e.retry_after_s is not None:
                delay = max(delay, e.retry_after_s + (rng or random).uniform(0, base_delay))
            await sleep(delay)
            continue
        except ProviderError as e:
            if not e.retryable:
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            await sleep(backoff_delay(attempt - 1, base=base_delay, cap=max_delay, rng=rng))
            continue
        if limiter is not None:
            limiter.on_success()
        return result
