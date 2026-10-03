"""Adapter interface every provider implements.

`classify(prompt, system_prompt, params) -> ClassifyResult` is the only call the runner makes
per point. Adapters never parse Land/Water; that is the extractor's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    ModelSpec,
    ProviderSpec,
    RunConfig,
)


class ProviderError(Exception):
    """A failed call. `retryable` drives backoff; non-retryable errors trip the circuit breaker."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class RateLimitError(ProviderError):
    def __init__(self, message: str, *, retry_after_s: float | None = None, status: int | None = 429):
        super().__init__(message, status=status, retryable=True)
        self.retry_after_s = retry_after_s


class UnsupportedConfigError(ValueError):
    """Raised by map_config when a requested setting (e.g. an effort level) is unsupported.

    The planner surfaces this as a refusal; adapters must never silently drop a setting.
    """


class Adapter(ABC):
    kind: str = ""

    def __init__(self, provider: ProviderSpec, model: ModelSpec, api_key: str | None = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.capabilities: Capabilities | None = None  # set from stored probe result

    @abstractmethod
    async def classify(
        self, prompt: str, system_prompt: str | None, params: CallParams
    ) -> ClassifyResult:
        """One prompt, `params.n` samples. Raise ProviderError / RateLimitError on failure."""

    @abstractmethod
    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        """Normalized config -> provider-native params (stored with the run).

        Raise UnsupportedConfigError for anything this provider/model cannot honour.
        """

    def default_capabilities(self) -> Capabilities:
        """Static best guess, used before a probe has run. The probe result overrides it."""
        return Capabilities()

    # Batch API (optional). Adapters that support it override all three.
    supports_batch: bool = False

    async def submit_batch(
        self, items: list[tuple[str, str, str | None, CallParams]]
    ) -> str:
        """items: (custom_id, prompt, system_prompt, params). Returns provider batch id."""
        raise NotImplementedError

    async def poll_batch(self, batch_id: str) -> dict[str, ClassifyResult] | None:
        """None while still running; else custom_id -> result (error set on failed items)."""
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
