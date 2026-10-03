"""OpenAI adapter on the official async `openai` SDK (Chat Completions + Batch API).

Logprobs: `logprobs=True, top_logprobs=k` (cap 20); `first_token_logprobs` comes from the first
content token's `top_logprobs`. Reasoning models usually reject logprobs and non-default
temperature, so `map_config` refuses those combinations (UnsupportedConfigError) instead of
dropping them. Reasoning tokens come from `usage.completion_tokens_details.reasoning_tokens`
(already included in completion_tokens, which is how OpenAI bills them).

Model families (detected from the name; override with registry `extra.openai_family`):

| family    | models                                   | sampling/logprobs              |
|-----------|------------------------------------------|--------------------------------|
| chat      | gpt-4o*, gpt-4.1*, gpt-3.5*, *-chat*      | allowed                        |
| o         | o1*, o3*, o4*                            | refused                        |
| gpt5      | gpt-5, gpt-5-mini, gpt-5-nano            | refused                        |
| gpt51     | gpt-5.1*                                 | only at effort=off ("none")    |
| gpt52     | gpt-5.2 and later gpt-5.x                | only at effort=off ("none")    |

Effort mapping (normalized -> `reasoning_effort`):

| normalized | chat            | o        | gpt5                         | gpt51   | gpt52   |
|------------|-----------------|----------|------------------------------|---------|---------|
| off        | no-op (no reasoning) | refused | "minimal" (least reasoning; not zero) | "none" | "none" |
| low        | refused         | "low"    | "low"                        | "low"   | "low"   |
| medium     | refused         | "medium" | "medium"                     | "medium"| "medium"|
| high       | refused         | "high"   | "high"                       | "high"  | "high"  |
| max        | refused         | refused  | refused                      | refused | "xhigh" |

Native strings (e.g. "minimal", "xhigh") pass through when listed for the family.
Override per model with `extra.effort_map: {level: native|null}`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import openai as openai_sdk

from blindearth.providers.base import (
    Adapter,
    ProviderError,
    RateLimitError,
    UnsupportedConfigError,
)
from blindearth.providers.openai_compat import (
    _get,
    check_logprobs_config,
    first_token_logprobs_openai,
    mode_temperature,
    openai_usage,
    raise_for_thinking,
    resolve_effort_map,
    split_think,
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

FAMILIES: dict[str, dict[str, Any]] = {
    "chat": {"reasoning": False, "map": {"off": "__noop__"}, "native": []},
    "o": {"reasoning": True, "map": {"low": "low", "medium": "medium", "high": "high"},
          "native": ["low", "medium", "high"]},
    "gpt5": {"reasoning": True,
             "map": {"off": "minimal", "low": "low", "medium": "medium", "high": "high"},
             "native": ["minimal", "low", "medium", "high"]},
    "gpt51": {"reasoning": True,
              "map": {"off": "none", "low": "low", "medium": "medium", "high": "high"},
              "native": ["none", "low", "medium", "high"], "sampling_when_none": True},
    "gpt52": {"reasoning": True,
              "map": {"off": "none", "low": "low", "medium": "medium", "high": "high",
                      "max": "xhigh"},
              "native": ["none", "low", "medium", "high", "xhigh"], "sampling_when_none": True},
}
TOP_LOGPROBS_CAP = 20


def openai_family(name: str) -> str:
    n = name.lower().split("/")[-1]
    if n.startswith("ft:"):
        n = n[3:]
    if "-chat" in n or n.startswith(("gpt-4", "gpt-3.5", "chatgpt")):
        return "chat"
    if re.match(r"^o\d", n):
        return "o"
    m = re.match(r"^gpt-5\.(\d+)", n)
    if m:
        return "gpt51" if int(m.group(1)) == 1 else "gpt52"
    if n.startswith("gpt-5"):
        return "gpt5"
    if re.match(r"^gpt-(\d+)", n) and int(re.match(r"^gpt-(\d+)", n).group(1)) >= 6:
        return "gpt52"
    return "chat"


class OpenAIAdapter(Adapter):
    kind = "openai"
    supports_batch = True

    def __init__(self, provider, model, api_key: str | None = None, *, client: Any = None):
        super().__init__(provider, model, api_key)
        self._client = client
        self.family = model.extra.get("openai_family") or openai_family(model.name)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown openai_family {self.family!r}")
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
            org = self.provider.extra.get("organization")
            if org:
                kw["organization"] = org
            self._client = openai_sdk.AsyncOpenAI(**kw)
        return self._client

    # ---- mapping

    def _native_effort(self, effort: str | None) -> str | None:
        """Normalized/native effort -> reasoning_effort value; None = send nothing."""
        if effort is None:
            return None
        emap = resolve_effort_map(self, self.profile["map"])
        if effort in emap:
            v = emap[effort]
            if v is None:
                raise UnsupportedConfigError(f"{self.model.name}: effort={effort!r} refused")
            return None if v == "__noop__" else v
        if effort in self.profile["native"]:
            return effort
        raise UnsupportedConfigError(
            f"{self.model.name} ({self.family}): effort={effort!r} unsupported; "
            f"accepted: {sorted(set(self.profile['map']) | set(self.profile['native']))}"
        )

    def _sampling_ok(self, native_effort: str | None) -> bool:
        if not self.profile["reasoning"]:
            return True
        return bool(self.profile.get("sampling_when_none")) and native_effort == "none"

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        ne = self._native_effort(config.effort)
        native: dict[str, Any] = {"model": self.model.name}
        if ne is not None:
            native["reasoning_effort"] = ne
        t = mode_temperature(config, mode)
        sampling_ok = self._sampling_ok(ne)
        if mode == ExtractionMode.LOGPROBS and not sampling_ok:
            raise UnsupportedConfigError(
                f"{self.model.name}: reasoning model does not return logprobs "
                f"(effort={config.effort!r}); use sample mode")
        k = check_logprobs_config(self, config, mode)
        if k is not None:
            native["logprobs"] = True
            native["top_logprobs"] = k
        if t is not None:
            if sampling_ok:
                native["temperature"] = t
            elif float(t) != 1.0:
                raise UnsupportedConfigError(
                    f"{self.model.name}: reasoning model only supports the default temperature "
                    f"1.0, got {t}")
            else:
                native["temperature"] = "provider default (1.0)"
        thinks = self.profile["reasoning"] and ne not in ("none",)
        if config.max_output_tokens is not None or thinks:
            native["max_completion_tokens"] = raise_for_thinking(
                config.max_output_tokens or 0,
                (config.effort if config.effort not in (None, "off") else None),
                forced=thinks)
        if mode == ExtractionMode.SAMPLE and config.n_samples:
            native["n"] = config.n_samples
        if config.seed is not None:
            native["seed"] = config.seed
        return native

    def default_capabilities(self) -> Capabilities:
        p = self.profile
        efforts = sorted(set(k for k, v in p["map"].items() if v is not None) | set(p["native"]))
        lp = not p["reasoning"] or bool(p.get("sampling_when_none"))
        return Capabilities(
            logprobs=lp,
            top_logprobs_max=TOP_LOGPROBS_CAP if lp else None,
            effort_param=p["reasoning"],
            supported_efforts=efforts if p["reasoning"] else ["off"],
            temperature_fixed_with_thinking=p["reasoning"],
            max_concurrency=32,
            supports_batch=True,
            notes=[f"static profile: family={self.family}"]
            + (["logprobs only at effort=off"] if p.get("sampling_when_none") else []),
        )

    # ---- requests

    def _body(self, prompt: str, system_prompt: str | None, params: CallParams) -> dict[str, Any]:
        try:
            ne = self._native_effort(params.effort)
        except UnsupportedConfigError as e:
            raise ProviderError(str(e), retryable=False) from e
        sampling_ok = self._sampling_ok(ne)
        messages = []
        if system_prompt:
            role = "developer" if self.profile["reasoning"] else "system"
            messages.append({"role": role, "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        thinks = self.profile["reasoning"] and ne != "none"
        body: dict[str, Any] = {
            "model": self.model.name,
            "messages": messages,
            "max_completion_tokens": raise_for_thinking(
                params.max_output_tokens,
                params.effort if params.effort not in (None, "off") else None,
                forced=thinks),
        }
        if ne is not None:
            body["reasoning_effort"] = ne
        if params.temperature is not None:
            if sampling_ok:
                body["temperature"] = params.temperature
            elif float(params.temperature) != 1.0:
                raise ProviderError(f"{self.model.name}: temperature {params.temperature} "
                                    "unsupported on reasoning model", retryable=False)
        if params.logprobs:
            if not sampling_ok:
                raise ProviderError(f"{self.model.name}: logprobs unsupported", retryable=False)
            body["logprobs"] = True
            body["top_logprobs"] = min(params.top_logprobs or 5, TOP_LOGPROBS_CAP)
        if params.n > 1:
            body["n"] = params.n
        if params.seed is not None:
            body["seed"] = params.seed
        return body

    def _result(self, comp: Any, params: CallParams, latency: float,
                body: dict[str, Any]) -> ClassifyResult:
        choices = sorted(_get(comp, "choices", []) or [], key=lambda c: _get(c, "index", 0) or 0)
        if not choices:
            raise ProviderError("response has no choices", retryable=True)
        texts, thinking, finish = [], [], []
        for c in choices:
            msg = _get(c, "message")
            answer, th = split_think(_get(msg, "content", "") or "")
            texts.append(answer)
            thinking.append(th if params.store_thinking else None)
            finish.append(str(_get(c, "finish_reason")))
        first_lp = None
        if params.logprobs:
            first_lp = first_token_logprobs_openai(_get(choices[0], "logprobs"))
        return ClassifyResult(
            texts=texts, usage=openai_usage(_get(comp, "usage")), latency_s=latency,
            finish_reasons=finish, first_token_logprobs=first_lp, thinking_texts=thinking,
            resolved_model=_get(comp, "model"),
            native_params={k: v for k, v in body.items() if k != "messages"}, raw=comp,
        )

    @staticmethod
    def _split_kwargs(body: dict[str, Any]) -> dict[str, Any]:
        # Newer params go through extra_body so older SDK versions still accept them.
        kw = dict(body)
        extra = {k: kw.pop(k) for k in ("max_completion_tokens", "reasoning_effort") if k in kw}
        if extra:
            kw["extra_body"] = extra
        return kw

    async def classify(self, prompt: str, system_prompt: str | None,
                       params: CallParams) -> ClassifyResult:
        body = self._body(prompt, system_prompt, params)
        t0 = time.perf_counter()
        try:
            comp = await self.client.chat.completions.create(**self._split_kwargs(body))
        except Exception as e:  # noqa: BLE001
            raise map_openai_error(e) from e
        return self._result(comp, params, time.perf_counter() - t0, body)

    # ---- Batch API

    async def submit_batch(self, items: list[tuple[str, str, str | None, CallParams]]) -> str:
        lines = []
        for custom_id, prompt, system_prompt, params in items:
            body = self._body(prompt, system_prompt, params)
            lines.append(json.dumps({"custom_id": custom_id, "method": "POST",
                                     "url": "/v1/chat/completions", "body": body}))
        data = ("\n".join(lines) + "\n").encode()
        # Results are parsed generically in poll_batch (logprobs read when present).
        try:
            f = await self.client.files.create(file=("blindearth_batch.jsonl", data),
                                               purpose="batch")
            b = await self.client.batches.create(
                input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h",
                metadata={"source": "blindearth", "model": self.model.name[:500]})
        except Exception as e:  # noqa: BLE001
            raise map_openai_error(e) from e
        return b.id

    async def _file_text(self, file_id: str) -> str:
        content = await self.client.files.content(file_id)
        text = getattr(content, "text", None)
        if callable(text):
            text = text()
        if text is None:
            raw = getattr(content, "content", b"")
            text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        return text

    async def poll_batch(self, batch_id: str) -> dict[str, ClassifyResult] | None:
        try:
            b = await self.client.batches.retrieve(batch_id)
        except Exception as e:  # noqa: BLE001
            raise map_openai_error(e) from e
        status = _get(b, "status")
        if status in ("validating", "in_progress", "finalizing", "cancelling"):
            return None
        out: dict[str, ClassifyResult] = {}
        lines: list[str] = []
        try:
            for fid in (_get(b, "output_file_id"), _get(b, "error_file_id")):
                if fid:
                    lines.extend((await self._file_text(fid)).splitlines())
        except Exception as e:  # noqa: BLE001
            raise map_openai_error(e) from e
        if not lines and status != "completed":
            errs = _get(b, "errors")
            raise ProviderError(f"batch {batch_id} {status}: {errs}", retryable=False)
        parse_params = CallParams(temperature=None, max_output_tokens=0, logprobs=True,
                                  store_thinking=True)
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id")
            resp = row.get("response") or {}
            body = resp.get("body") or {}
            status_code = resp.get("status_code")
            if row.get("error") or (status_code and status_code >= 400) or not body.get("choices"):
                err = row.get("error") or body.get("error") or {"message": f"status {status_code}"}
                out[cid] = ClassifyResult(texts=[], usage=Usage(), latency_s=0.0,
                                          error=str(err.get("message", err)
                                                    if isinstance(err, dict) else err),
                                          native_params={"batch_id": batch_id})
                continue
            res = self._result(body, parse_params, 0.0, {"batch_id": batch_id})
            out[cid] = res
        return out

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass


def map_openai_error(e: Exception) -> Exception:
    if isinstance(e, ProviderError):
        return e
    if isinstance(e, openai_sdk.RateLimitError):
        body = getattr(e, "body", None)
        code = body.get("code") if isinstance(body, dict) else getattr(e, "code", None)
        if code == "insufficient_quota":
            return ProviderError(f"insufficient quota: {e}", status=429, retryable=False)
        headers = getattr(getattr(e, "response", None), "headers", None) or {}
        return RateLimitError(str(e), retry_after_s=parse_retry_after(headers.get("retry-after")))
    if isinstance(e, openai_sdk.APITimeoutError):
        return ProviderError(f"timeout: {e}", retryable=True)
    if isinstance(e, openai_sdk.APIConnectionError):
        return ProviderError(f"connection error: {e}", retryable=True)
    if isinstance(e, openai_sdk.APIStatusError):
        status = int(getattr(e, "status_code", 0) or 0)
        retryable = status in (408, 409) or status >= 500
        return ProviderError(str(e), status=status, retryable=retryable)
    return e


__all__ = ["OpenAIAdapter", "openai_family", "FAMILIES", "map_openai_error"]
