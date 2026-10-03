"""OpenAI-compatible adapter (vLLM, SGLang, LM Studio, Together, Fireworks, Groq, DeepSeek, ...).

Talks to `{base_url}/chat/completions` with httpx. Text only, no tools. Also hosts helpers
shared by the other adapters (think-tag stripping, OpenAI-format logprob parsing, the
thinking `max_tokens` floor, normalized effort levels).

Effort mapping (OpenAI-compatible servers; the spec leaves this undefined, so this is ours):

| normalized | native body params (default)          |
|------------|---------------------------------------|
| off        | refused unless an `effort_map` defines it (servers disagree: Qwen3 wants
|            | `chat_template_kwargs.enable_thinking=false`, others `reasoning_effort=none`) |
| low        | `reasoning_effort: "low"`             |
| medium     | `reasoning_effort: "medium"`          |
| high       | `reasoning_effort: "high"`            |
| max        | refused unless an `effort_map` defines it |
| other str  | passed as `reasoning_effort: <str>` (provider-native value) |

Override per provider or per model with `extra.effort_map: {level: {body params}}`
(model extra wins over provider extra). A level mapped to `null` is refused.

Server facts the local subclasses rely on (docs checked 2026-10-03):
- Ollama `/v1/chat/completions` honours `reasoning_effort` / `reasoning.effort`; `"none"`
  requests no thinking; boolean-only thinking models map every other recognised effort to
  `true`, and unsupported level names silently fall back to the model default (check
  `/api/show` -> `thinking.values`). `n` and logprobs are NOT supported on `/v1`; logprobs
  exist only on native `/api/chat` (`logprobs`, `top_logprobs`), where thinking is `think`
  (bool | model-defined string such as gpt-oss "low"/"medium"/"high" | null).
  https://docs.ollama.com/api/openai-compatibility , https://docs.ollama.com/capabilities/thinking ,
  https://docs.ollama.com/api/chat
- llama.cpp `llama-server` `/v1/chat/completions`: `chat_template_kwargs: {enable_thinking:
  false}` and `reasoning_effort: "none"` both disable thinking; any other `reasoning_effort` is
  only handed to the Jinja template (ignored if the template does not use it);
  `reasoning_budget_tokens` caps thinking per request; `reasoning_format` (`none`|`deepseek`|
  `deepseek-legacy`) controls `<think>` extraction into `reasoning_content`; OpenAI `logprobs` +
  `top_logprobs` (default 20) map to native `n_probs`.
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md ,
  tools/server/server-common.cpp (oaicompat_chat_params_parse)

Thinking runs (any effort other than None/off, or `forced_thinking`) get `max_tokens` raised to at
least `THINKING_MAX_TOKENS_FLOOR[effort]` so the answer is not cut off after the reasoning.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from blindearth.providers.base import (
    Adapter,
    ProviderError,
    RateLimitError,
    UnsupportedConfigError,
)
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    RunConfig,
    Usage,
)

NORMALIZED_EFFORTS = ("off", "low", "medium", "high", "max")

# Minimum max_output_tokens for a thinking call, by effort. Native/unknown effort -> "medium".
THINKING_MAX_TOKENS_FLOOR = {"low": 4096, "medium": 8192, "high": 16000, "xhigh": 24000,
                             "max": 32000}
DEFAULT_TOP_LOGPROBS = 20

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- shared helpers


def is_thinking_effort(effort: str | None) -> bool:
    return effort is not None and effort != "off"


def thinking_floor(effort: str | None) -> int:
    if not is_thinking_effort(effort):
        return 0
    return THINKING_MAX_TOKENS_FLOOR.get(str(effort), THINKING_MAX_TOKENS_FLOOR["medium"])


def raise_for_thinking(max_tokens: int, effort: str | None, *, forced: bool = False) -> int:
    """Auto-raise max_tokens for thinking runs (spec: 'auto-raised for thinking runs')."""
    if forced and not is_thinking_effort(effort):
        return max(max_tokens, THINKING_MAX_TOKENS_FLOOR["medium"])
    return max(max_tokens, thinking_floor(effort))


def split_think(text: str | None) -> tuple[str, str | None]:
    """Split `<think>...</think>` reasoning from the answer. Returns (answer, thinking|None)."""
    if not text:
        return "", None
    thoughts = [m.strip() for m in _THINK_RE.findall(text)]
    answer = _THINK_RE.sub("", text)
    low = answer.lower()
    if "</think>" in low:  # opening tag was part of the chat template (Qwen3 style)
        i = low.rfind("</think>")
        thoughts.insert(0, answer[:i].strip())
        answer = answer[i + len("</think>"):]
    elif "<think>" in low:  # unclosed, truncated mid-reasoning: no answer
        i = low.find("<think>")
        thoughts.append(answer[i + len("<think>"):].strip())
        answer = answer[:i]
    thinking = "\n\n".join(t for t in thoughts if t) or None
    return answer.strip(), thinking


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-key access, so SDK objects and plain JSON dicts parse the same way."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def first_token_logprobs_openai(logprobs_obj: Any) -> dict[str, float] | None:
    """OpenAI-format `choice.logprobs` -> {token: logprob} for the first answer token.

    Skips a leading `<think>...</think>` block and whitespace-only tokens so the first token of
    the *final answer* is used. The sampled token itself is included even if absent from the top
    list.
    """
    content = _get(logprobs_obj, "content")
    if not content:
        return None
    start = 0
    joined = ""
    n_close = 0
    for i, tok in enumerate(content):
        joined += (_get(tok, "token", "") or "").lower()
        c = joined.count("</think>")
        if c > n_close:  # a closing tag just completed: the answer starts after this token
            n_close = c
            start = i + 1
    for tok in content[start:]:
        t = _get(tok, "token", "") or ""
        if t.strip() == "":
            continue
        out: dict[str, float] = {}
        for alt in _get(tok, "top_logprobs", None) or []:
            at = _get(alt, "token")
            lp = _get(alt, "logprob")
            if at is not None and lp is not None:
                out[at] = max(float(lp), out.get(at, float("-inf")))
        lp = _get(tok, "logprob")
        if lp is not None and t not in out:
            out[t] = float(lp)
        return out or None
    return None


def openai_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    details = _get(usage, "completion_tokens_details")
    reasoning = _get(details, "reasoning_tokens", 0) or 0
    return Usage(
        input_tokens=int(_get(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(_get(usage, "completion_tokens", 0) or 0),
        thinking_tokens=int(reasoning),
    )


def http_error_to_provider_error(status: int, message: str,
                                 headers: Any = None) -> ProviderError:
    """Map an HTTP status to RateLimitError / retryable / non-retryable ProviderError."""
    from blindearth.ratelimit import parse_retry_after

    if status == 429:
        ra = None
        if headers is not None:
            ra = parse_retry_after(headers.get("retry-after"))
            if ra is None and headers.get("retry-after-ms"):
                try:
                    ra = float(headers.get("retry-after-ms")) / 1000.0
                except ValueError:
                    ra = None
        return RateLimitError(f"HTTP 429: {message}", retry_after_s=ra)
    retryable = status in (408, 409, 425) or status >= 500
    return ProviderError(f"HTTP {status}: {message}", status=status, retryable=retryable)


def resolve_effort_map(adapter: Adapter, default: dict[str, Any]) -> dict[str, Any]:
    m = dict(default)
    for src in (adapter.provider.extra, adapter.model.extra):
        em = (src or {}).get("effort_map")
        if isinstance(em, dict):
            m.update(em)
    return m


def check_logprobs_config(adapter: Adapter, config: RunConfig, mode: ExtractionMode) -> int | None:
    """Validate logprobs mode against capabilities; returns the top_logprobs to request."""
    if mode != ExtractionMode.LOGPROBS:
        if config.top_logprobs is not None:
            raise UnsupportedConfigError("top_logprobs is only valid in logprobs mode")
        return None
    caps = adapter.capabilities or adapter.default_capabilities()
    if not caps.logprobs:
        raise UnsupportedConfigError(
            f"{adapter.model.name}: logprobs not supported (probe or default capabilities)"
        )
    cap = caps.top_logprobs_max
    k = config.top_logprobs
    if k is None:
        k = min(DEFAULT_TOP_LOGPROBS, cap) if cap else DEFAULT_TOP_LOGPROBS
    if k < 1:
        raise UnsupportedConfigError("top_logprobs must be >= 1")
    if cap is not None and k > cap:
        raise UnsupportedConfigError(f"top_logprobs={k} exceeds provider cap {cap}")
    return k


def mode_temperature(config: RunConfig, mode: ExtractionMode) -> float | None:
    if mode == ExtractionMode.GREEDY:
        if config.temperature not in (None, 0, 0.0):
            raise UnsupportedConfigError("greedy mode requires temperature 0")
        return 0.0
    return config.temperature


# --------------------------------------------------------------------------- adapter


class OpenAICompatAdapter(Adapter):
    kind = "openai_compatible"
    default_base_url: str | None = None
    supports_n_default: bool = True
    max_tokens_field: str = "max_tokens"
    default_logprobs: bool = True  # unprobed guess; most servers support it
    default_top_logprobs_max: int | None = 20

    EFFORT_MAP: dict[str, Any] = {
        "off": None,
        "low": {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "medium"},
        "high": {"reasoning_effort": "high"},
        "max": None,
    }

    def __init__(self, provider, model, api_key: str | None = None, *,
                 client: httpx.AsyncClient | None = None):
        super().__init__(provider, model, api_key)
        base = provider.base_url or self.default_base_url
        if not base:
            raise ValueError(f"provider {provider.id!r} ({provider.kind}) needs base_url")
        self.base_url = base.rstrip("/")
        self._client = client
        self._own_client = client is None
        self.timeout_s = float(provider.extra.get("timeout_s", 120.0))

    # ---- config mapping

    def _effort_params(self, effort: str | None) -> dict[str, Any]:
        if effort is None:
            return {}
        emap = resolve_effort_map(self, self.EFFORT_MAP)
        if effort in emap:
            v = emap[effort]
            if v is None:
                raise UnsupportedConfigError(
                    f"effort={effort!r} has no mapping for {self.kind} model {self.model.name}; "
                    "set extra.effort_map in the registry"
                )
            return dict(v)
        if effort in NORMALIZED_EFFORTS:
            raise UnsupportedConfigError(f"effort={effort!r} unsupported")
        return {"reasoning_effort": effort}  # provider-native value

    def _check_probed_effort(self, effort: str | None) -> None:
        caps = self.capabilities
        if caps is None or effort is None or not caps.probed_at:
            return
        if not caps.effort_param:
            raise UnsupportedConfigError(f"probe found no effort parameter on {self.model.name}")
        if caps.supported_efforts and effort not in caps.supported_efforts:
            raise UnsupportedConfigError(
                f"effort={effort!r} not accepted by {self.model.name} "
                f"(probe: {caps.supported_efforts})"
            )

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        self._check_probed_effort(config.effort)
        native: dict[str, Any] = {"model": self.model.name}
        native.update(self._effort_params(config.effort))
        t = mode_temperature(config, mode)
        if t is not None:
            native["temperature"] = t
        k = check_logprobs_config(self, config, mode)
        if k is not None:
            native["logprobs"] = True
            native["top_logprobs"] = k
        if mode == ExtractionMode.SAMPLE and config.n_samples:
            native["n"] = config.n_samples
        if config.max_output_tokens is not None or is_thinking_effort(config.effort):
            native[self.max_tokens_field] = raise_for_thinking(
                config.max_output_tokens or 0, config.effort, forced=self.model.forced_thinking
            )
        if config.seed is not None:
            native["seed"] = config.seed
        return native

    def default_capabilities(self) -> Capabilities:
        return Capabilities(
            logprobs=self.default_logprobs,
            top_logprobs_max=self.default_top_logprobs_max if self.default_logprobs else None,
            effort_param=False,
            supported_efforts=[],
            max_concurrency=8,
            supports_batch=False,
            notes=["static guess; run the capability probe"],
        )

    # ---- HTTP

    def _client_or_new(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.provider.extra.get("headers", {}) or {})
        return h

    def _supports_n(self) -> bool:
        v = self.model.extra.get("supports_n", self.provider.extra.get("supports_n"))
        return self.supports_n_default if v is None else bool(v)

    def _build_body(self, prompt: str, system_prompt: str | None, params: CallParams,
                    n: int) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": self.model.name, "messages": messages}
        try:
            body.update(self._effort_params(params.effort))
        except UnsupportedConfigError as e:
            raise ProviderError(str(e), retryable=False) from e
        body[self.max_tokens_field] = raise_for_thinking(
            params.max_output_tokens, params.effort, forced=self.model.forced_thinking
        )
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = params.top_logprobs or 5
        if n > 1:
            body["n"] = n
        if params.seed is not None:
            body["seed"] = params.seed
        extra_body = self.model.extra.get("extra_body") or self.provider.extra.get("extra_body")
        if isinstance(extra_body, dict):
            for k, v in extra_body.items():
                body.setdefault(k, v)
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        client = self._client_or_new()
        url = f"{self.base_url}/chat/completions"
        try:
            resp = await client.post(url, json=body, headers=self._headers(),
                                     timeout=self.timeout_s)
        except httpx.TimeoutException as e:
            raise ProviderError(f"timeout: {e}", retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"transport error: {e}", retryable=True) from e
        if resp.status_code >= 400:
            text = resp.text[:500]
            raise http_error_to_provider_error(resp.status_code, text, resp.headers)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(f"invalid JSON from server: {resp.text[:200]}",
                                retryable=True) from e
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            status = code if isinstance(code, int) else None
            if status is not None:
                raise http_error_to_provider_error(status, msg)
            raise ProviderError(msg, retryable=False)
        return data

    def _parse_choice(self, choice: dict[str, Any], store_thinking: bool
                      ) -> tuple[str, str | None, str]:
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        answer, thinking = split_think(content)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            thinking = reasoning if not thinking else reasoning + "\n\n" + thinking
        return answer, (thinking if store_thinking else None), str(choice.get("finish_reason"))

    async def classify(self, prompt: str, system_prompt: str | None,
                       params: CallParams) -> ClassifyResult:
        n = max(1, params.n)
        t0 = time.perf_counter()
        if n == 1 or self._supports_n():
            bodies = [self._build_body(prompt, system_prompt, params, n)]
        else:
            bodies = [self._build_body(prompt, system_prompt, params, 1) for _ in range(n)]
        texts: list[str] = []
        thinking_texts: list[str | None] = []
        finish: list[str] = []
        usage = Usage()
        first_lp: dict[str, float] | None = None
        resolved = None
        raw: list[Any] = []
        for body in bodies:
            data = await self._post(body)
            raw.append(data)
            resolved = resolved or data.get("model")
            usage = usage + openai_usage(data.get("usage"))
            choices = sorted(data.get("choices") or [], key=lambda c: c.get("index", 0))
            if not choices:
                raise ProviderError("response has no choices", retryable=True)
            for c in choices:
                a, th, fr = self._parse_choice(c, params.store_thinking)
                texts.append(a)
                thinking_texts.append(th)
                finish.append(fr)
            if params.logprobs and first_lp is None:
                first_lp = first_token_logprobs_openai(choices[0].get("logprobs"))
        native = {k: v for k, v in bodies[0].items() if k != "messages"}
        if len(bodies) > 1:
            native["n_requests"] = len(bodies)
        return ClassifyResult(
            texts=texts,
            usage=usage,
            latency_s=time.perf_counter() - t0,
            finish_reasons=finish,
            first_token_logprobs=first_lp if params.logprobs else None,
            thinking_texts=thinking_texts,
            resolved_model=resolved,
            native_params=native,
            raw=raw,
        )

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            await self._client.aclose()
            self._client = None


__all__ = [
    "OpenAICompatAdapter",
    "split_think",
    "first_token_logprobs_openai",
    "openai_usage",
    "http_error_to_provider_error",
    "raise_for_thinking",
    "thinking_floor",
    "is_thinking_effort",
    "check_logprobs_config",
    "mode_temperature",
    "resolve_effort_map",
    "NORMALIZED_EFFORTS",
    "THINKING_MAX_TOKENS_FLOOR",
]
