"""Anthropic (Claude) adapter on the official async `anthropic` SDK.

No logprobs on the Claude API, so sampling mode loops the request `n` times. Text only, no tools.
Thinking blocks are stripped from the answer text; thinking tokens are counted from
`usage.output_tokens_details.thinking_tokens` when the API reports it (otherwise estimated as
output tokens minus ~len(answer)/4 when thinking blocks were present); thinking text is kept only
with `params.store_thinking` (and then `display: "summarized"` is requested on adaptive models,
since the default display on current models returns empty thinking text).

Server-side refusal fallbacks are deliberately NOT enabled: a fallback would answer with a
different model and silently contaminate the run. A refusal comes back as finish reason
`refusal`, which the extractor treats as invalid.

Effort mapping (normalized -> native). Families are detected from the model name; override with
`extra.anthropic_family` in the registry.

| family (models)                                   | off                         | low/medium/high/max                         | sampling params |
|---------------------------------------------------|-----------------------------|---------------------------------------------|-----------------|
| always (fable-5*, mythos-5*, opus-5-5)            | refused (thinking always on)| thinking adaptive + output_config.effort=<same> | refused (400)   |
| opus5 (opus-5)                                    | thinking disabled           | adaptive + effort=<same>                    | refused         |
| sonnet55 (sonnet-5-5)                             | thinking between_tools      | adaptive + effort=<same>                    | refused         |
| adaptive (opus-4-7, opus-4-8)                     | thinking disabled           | adaptive + effort=<same>                    | refused         |
| sonnet5 (sonnet-5; thinks by default)             | thinking disabled           | adaptive + effort=<same>                    | refused         |
| adaptive46 (opus-4-6, sonnet-4-6)                 | thinking omitted            | adaptive + effort; `max` ok, `xhigh` not    | allowed (thinking off only) |
| budget (haiku-4-5, *-4-5, *-4-1, *-4-0, 3.x)      | thinking omitted            | enabled + budget_tokens low 1024 / medium 4096 / high 16000 / max 32000 | allowed (thinking off only) |

Native effort strings (e.g. `xhigh`) pass through on the families that list them. `None` leaves
the provider default (Opus 5.5 still thinks at its default `medium`). Thinking calls get
`max_tokens` auto-raised (see openai_compat.THINKING_MAX_TOKENS_FLOOR; budget models get
budget + 1024 at least).

Where the model fixes temperature, a temperature of 1.0 (the API default) is accepted and not
sent; any other temperature, and greedy mode, are refused in `map_config`.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic as anthropic_sdk

from blindearth.providers.base import (
    Adapter,
    ProviderError,
    RateLimitError,
    UnsupportedConfigError,
)
from blindearth.providers.openai_compat import (
    NORMALIZED_EFFORTS,
    _get,
    is_thinking_effort,
    raise_for_thinking,
)
from blindearth.ratelimit import parse_retry_after
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    RunConfig,
    Usage,
)

BUDGET_TOKENS = {"low": 1024, "medium": 4096, "high": 16000, "max": 32000}
STREAM_ABOVE_MAX_TOKENS = 16000  # SDK wants streaming for long non-streaming requests

# family -> (efforts accepted, off-thinking config | "omit" | None=refused, sampling allowed)
FAMILIES: dict[str, dict[str, Any]] = {
    "always": {"efforts": ["low", "medium", "high", "xhigh", "max"], "off": None,
               "sampling": False},
    "opus5": {"efforts": ["low", "medium", "high", "xhigh", "max"],
              "off": {"type": "disabled"}, "sampling": False},
    "sonnet55": {"efforts": ["low", "medium", "high", "xhigh", "max"],
                 "off": {"type": "between_tools"}, "sampling": False},
    "adaptive": {"efforts": ["low", "medium", "high", "xhigh", "max"],
                 "off": {"type": "disabled"}, "sampling": False},
    "sonnet5": {"efforts": ["low", "medium", "high", "xhigh", "max"],
                "off": {"type": "disabled"}, "sampling": False},
    "adaptive46": {"efforts": ["low", "medium", "high", "max"], "off": "omit",
                   "sampling": True},
    "budget": {"efforts": ["low", "medium", "high", "max"], "off": "omit", "sampling": True},
}
# Families that think when no effort is given (thinking on by default).
DEFAULT_THINKING_FAMILIES = ("always", "opus5", "sonnet55", "sonnet5")


def anthropic_family(name: str) -> str:
    n = name.lower()
    if n.startswith(("claude-fable-5", "claude-mythos-5", "claude-opus-5-5")):
        return "always"
    if n.startswith("claude-opus-5"):
        return "opus5"
    if n.startswith("claude-sonnet-5-5"):
        return "sonnet55"
    if n.startswith("claude-sonnet-5"):
        return "sonnet5"
    if n.startswith(("claude-opus-4-7", "claude-opus-4-8")):
        return "adaptive"
    if n.startswith(("claude-opus-4-6", "claude-sonnet-4-6")):
        return "adaptive46"
    return "budget"


class AnthropicAdapter(Adapter):
    kind = "anthropic"
    supports_batch = True

    def __init__(self, provider, model, api_key: str | None = None, *, client: Any = None):
        super().__init__(provider, model, api_key)
        self._client = client
        self.family = model.extra.get("anthropic_family") or anthropic_family(model.name)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown anthropic_family {self.family!r}")
        self.profile = FAMILIES[self.family]
        self.timeout_s = float(provider.extra.get("timeout_s", 600.0))

    @property
    def client(self):
        if self._client is None:
            kw: dict[str, Any] = {"max_retries": 0, "timeout": self.timeout_s}
            if self.api_key:
                kw["api_key"] = self.api_key
            if self.provider.base_url:
                kw["base_url"] = self.provider.base_url
            self._client = anthropic_sdk.AsyncAnthropic(**kw)
        return self._client

    # ---- mapping

    def _effort_native(self, effort: str | None) -> dict[str, Any]:
        """-> {'thinking': ..., 'output_config': ...} (keys only when sent)."""
        p = self.profile
        if effort is None:
            return {}
        if effort == "off":
            off = p["off"]
            if off is None:
                raise UnsupportedConfigError(
                    f"{self.model.name}: thinking cannot be disabled; use effort=low"
                )
            return {} if off == "omit" else {"thinking": dict(off)}
        if effort not in p["efforts"]:
            raise UnsupportedConfigError(
                f"{self.model.name}: effort={effort!r} not supported "
                f"(accepted: off={p['off'] is not None}, {p['efforts']})"
            )
        if self.family == "budget":
            if effort not in BUDGET_TOKENS:
                raise UnsupportedConfigError(f"{self.model.name}: effort={effort!r} unsupported")
            return {"thinking": {"type": "enabled", "budget_tokens": BUDGET_TOKENS[effort]}}
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}

    def _thinks(self, effort: str | None) -> bool:
        if self.family == "always" or self.model.forced_thinking:
            return True
        if effort == "off":
            return False
        if effort is not None:
            return True
        # provider default when effort is None: these families think unless told otherwise
        return self.family in DEFAULT_THINKING_FAMILIES

    def _sampling_allowed(self, effort: str | None) -> bool:
        return bool(self.profile["sampling"]) and not self._thinks(effort)

    def _max_tokens(self, requested: int, effort: str | None) -> int:
        thinks = self._thinks(effort)
        mt = raise_for_thinking(requested, effort if thinks and effort not in (None, "off")
                                else None, forced=thinks)
        if self.family == "budget" and effort in BUDGET_TOKENS:
            mt = max(mt, BUDGET_TOKENS[effort] + 1024)
        return mt

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        if mode == ExtractionMode.LOGPROBS or config.top_logprobs is not None:
            raise UnsupportedConfigError("Anthropic API returns no logprobs; use sample mode")
        native: dict[str, Any] = {"model": self.model.name}
        native.update(self._effort_native(config.effort))
        temp = 0.0 if mode == ExtractionMode.GREEDY else config.temperature
        if mode == ExtractionMode.GREEDY and config.temperature not in (None, 0, 0.0):
            raise UnsupportedConfigError("greedy mode requires temperature 0")
        if self._sampling_allowed(config.effort):
            if temp is not None:
                native["temperature"] = temp
        else:
            if temp is not None and float(temp) != 1.0:
                raise UnsupportedConfigError(
                    f"{self.model.name}: temperature is fixed by the provider in this "
                    f"configuration (thinking or no sampling params); got {temp}"
                )
            native["temperature"] = "provider-fixed (1.0)"
        if config.max_output_tokens is not None or self._thinks(config.effort):
            native["max_tokens"] = self._max_tokens(config.max_output_tokens or 0, config.effort)
        if mode == ExtractionMode.SAMPLE and config.n_samples:
            native["n_requests"] = config.n_samples  # API has no `n`; loops
        if config.seed is not None:
            native["seed"] = "not supported by API (point order only)"
        return native

    def default_capabilities(self) -> Capabilities:
        p = self.profile
        efforts = (["off"] if p["off"] is not None else []) + list(p["efforts"])
        return Capabilities(
            logprobs=False,
            top_logprobs_max=None,
            effort_param=True,
            supported_efforts=efforts,
            temperature_fixed_with_thinking=True,
            max_concurrency=16,
            supports_batch=True,
            notes=[f"static profile: family={self.family}"],
        )

    # ---- requests

    def _request_kwargs(self, prompt: str, system_prompt: str | None,
                        params: CallParams) -> dict[str, Any]:
        try:
            eff = self._effort_native(params.effort)
        except UnsupportedConfigError as e:
            raise ProviderError(str(e), retryable=False) from e
        kw: dict[str, Any] = {
            "model": self.model.name,
            "max_tokens": self._max_tokens(params.max_output_tokens, params.effort),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kw["system"] = system_prompt
        if "thinking" in eff:
            kw["thinking"] = eff["thinking"]
        if "output_config" in eff:
            kw["output_config"] = eff["output_config"]
        thinks = self._thinks(params.effort)
        if thinks and params.store_thinking and self.family != "budget":
            th = dict(kw.get("thinking") or {"type": "adaptive"})
            if th.get("type") == "adaptive":
                th["display"] = "summarized"
                kw["thinking"] = th
        if self._sampling_allowed(params.effort):
            if params.temperature is not None:
                kw["temperature"] = params.temperature
        elif params.temperature is not None and float(params.temperature) != 1.0:
            raise ProviderError(
                f"{self.model.name}: temperature {params.temperature} cannot be honoured "
                "(provider-fixed)", retryable=False)
        return kw

    async def _send(self, kw: dict[str, Any]):
        # output_config goes through extra_body so older SDK versions still accept it.
        call = dict(kw)
        extra_body = {}
        if "output_config" in call:
            extra_body["output_config"] = call.pop("output_config")
        if extra_body:
            call["extra_body"] = extra_body
        try:
            if call["max_tokens"] > STREAM_ABOVE_MAX_TOKENS:
                async with self.client.messages.stream(**call) as stream:
                    return await stream.get_final_message()
            return await self.client.messages.create(**call)
        except Exception as e:  # noqa: BLE001 - mapped below
            raise map_anthropic_error(e) from e

    def _parse_message(self, msg: Any, store_thinking: bool
                       ) -> tuple[str, str | None, str, Usage]:
        texts: list[str] = []
        thoughts: list[str] = []
        saw_thinking = False
        for block in _get(msg, "content", []) or []:
            btype = _get(block, "type")
            if btype == "text":
                texts.append(_get(block, "text", "") or "")
            elif btype == "thinking":
                saw_thinking = True
                t = _get(block, "thinking", "") or ""
                if t:
                    thoughts.append(t)
            elif btype == "redacted_thinking":
                saw_thinking = True
        text = "".join(texts).strip()
        u = _get(msg, "usage")
        out_tokens = int(_get(u, "output_tokens", 0) or 0)
        details = _get(u, "output_tokens_details")
        thinking_tokens = _get(details, "thinking_tokens", None)
        if thinking_tokens is None:
            thinking_tokens = max(0, out_tokens - (len(text) + 3) // 4) if saw_thinking else 0
        usage = Usage(
            input_tokens=int(_get(u, "input_tokens", 0) or 0)
            + int(_get(u, "cache_read_input_tokens", 0) or 0)
            + int(_get(u, "cache_creation_input_tokens", 0) or 0),
            output_tokens=out_tokens,
            thinking_tokens=int(thinking_tokens),
        )
        thinking_text = "\n\n".join(thoughts) if (store_thinking and thoughts) else None
        return text, thinking_text, str(_get(msg, "stop_reason")), usage

    async def classify(self, prompt: str, system_prompt: str | None,
                       params: CallParams) -> ClassifyResult:
        if params.logprobs:
            raise ProviderError("Anthropic API returns no logprobs", retryable=False)
        kw = self._request_kwargs(prompt, system_prompt, params)
        t0 = time.perf_counter()
        texts: list[str] = []
        thinking_texts: list[str | None] = []
        finish: list[str] = []
        usage = Usage()
        resolved = None
        raw = []
        for _ in range(max(1, params.n)):
            msg = await self._send(kw)
            raw.append(msg)
            text, th, fr, u = self._parse_message(msg, params.store_thinking)
            texts.append(text)
            thinking_texts.append(th)
            finish.append(fr)
            usage = usage + u
            resolved = resolved or _get(msg, "model")
        native = {k: v for k, v in kw.items() if k not in ("messages", "system")}
        native["n_requests"] = max(1, params.n)
        return ClassifyResult(
            texts=texts, usage=usage, latency_s=time.perf_counter() - t0,
            finish_reasons=finish, first_token_logprobs=None, thinking_texts=thinking_texts,
            resolved_model=resolved, native_params=native, raw=raw,
        )

    # ---- Message Batches

    async def submit_batch(self, items: list[tuple[str, str, str | None, CallParams]]) -> str:
        """custom_ids must match ^[A-Za-z0-9_-]{1,58}$ (a `__s<k>` sample suffix is appended)."""
        requests = []
        for custom_id, prompt, system_prompt, params in items:
            kw = self._request_kwargs(prompt, system_prompt, params)
            for k in range(max(1, params.n)):
                requests.append({"custom_id": f"{custom_id}__s{k}", "params": kw})
        try:
            batch = await self.client.messages.batches.create(requests=requests)
        except Exception as e:  # noqa: BLE001
            raise map_anthropic_error(e) from e
        return batch.id

    async def poll_batch(self, batch_id: str) -> dict[str, ClassifyResult] | None:
        try:
            batch = await self.client.messages.batches.retrieve(batch_id)
            if _get(batch, "processing_status") != "ended":
                return None
            results_iter = await self.client.messages.batches.results(batch_id)
            rows = [r async for r in results_iter]
        except Exception as e:  # noqa: BLE001
            raise map_anthropic_error(e) from e
        grouped: dict[str, list[tuple[int, Any]]] = {}
        for r in rows:
            cid = _get(r, "custom_id")
            base, _, k = cid.rpartition("__s")
            if not base:
                base, k = cid, "0"
            grouped.setdefault(base, []).append((int(k) if k.isdigit() else 0, _get(r, "result")))
        out: dict[str, ClassifyResult] = {}
        for base, samples in grouped.items():
            samples.sort(key=lambda x: x[0])
            res = ClassifyResult(texts=[], usage=Usage(), latency_s=0.0,
                                 native_params={"batch_id": batch_id, "n_requests": len(samples)})
            errors = []
            for _, result in samples:
                rtype = _get(result, "type")
                if rtype == "succeeded":
                    msg = _get(result, "message")
                    text, th, fr, u = self._parse_message(msg, store_thinking=True)
                    res.texts.append(text)
                    res.thinking_texts.append(th)
                    res.finish_reasons.append(fr)
                    res.usage = res.usage + u
                    res.resolved_model = res.resolved_model or _get(msg, "model")
                else:
                    err = _get(result, "error")
                    detail = _get(_get(err, "error"), "type") or _get(err, "type") or ""
                    errors.append(f"{rtype}{(': ' + str(detail)) if detail else ''}")
            if errors:
                res.error = f"{len(errors)}/{len(samples)} batch samples failed: " + "; ".join(
                    errors)
            out[base] = res
        return out

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass


def map_anthropic_error(e: Exception) -> Exception:
    if isinstance(e, ProviderError):
        return e
    if isinstance(e, anthropic_sdk.RateLimitError):
        headers = getattr(getattr(e, "response", None), "headers", None) or {}
        return RateLimitError(str(e), retry_after_s=parse_retry_after(headers.get("retry-after")))
    if isinstance(e, anthropic_sdk.APITimeoutError):
        return ProviderError(f"timeout: {e}", retryable=True)
    if isinstance(e, anthropic_sdk.APIConnectionError):
        return ProviderError(f"connection error: {e}", retryable=True)
    if isinstance(e, anthropic_sdk.APIStatusError):
        status = int(getattr(e, "status_code", 0) or 0)
        if status == 429:
            headers = getattr(getattr(e, "response", None), "headers", None) or {}
            return RateLimitError(str(e),
                                  retry_after_s=parse_retry_after(headers.get("retry-after")))
        retryable = status in (408, 409) or status >= 500  # includes 529 overloaded
        return ProviderError(str(e), status=status, retryable=retryable)
    return e


__all__ = ["AnthropicAdapter", "anthropic_family", "FAMILIES", "BUDGET_TOKENS",
           "map_anthropic_error", "NORMALIZED_EFFORTS"]
