"""Google Gemini adapter on the `google-genai` SDK (`client.aio.models.generate_content`).

Text only, no tools. Samples loop `n` times (candidate_count is not supported on every model).
Thinking parts (`part.thought`) are removed from the answer; `usage_metadata.thoughts_token_count`
is the thinking count and is added to output tokens, since Google bills thoughts as output but
reports them outside `candidates_token_count`. Logprobs via `response_logprobs` + `logprobs=k`
on models that support them (few do; the probe decides).

Effort mapping (normalized -> native; family from the model name, override `extra.gemini_family`):

| normalized | gemini-2.5-pro           | gemini-2.5-flash / flash-lite | gemini-3.x               | gemini-2.0 / 1.5 |
|------------|--------------------------|-------------------------------|--------------------------|------------------|
| off        | refused (min budget 128) | thinking_budget 0             | refused (cannot disable) | no-op            |
| low        | thinking_budget 1024     | thinking_budget 1024          | thinking_level LOW       | refused          |
| medium     | thinking_budget 8192     | thinking_budget 8192          | thinking_level MEDIUM    | refused          |
| high       | thinking_budget 24576    | thinking_budget 24576         | thinking_level HIGH      | refused          |
| max        | thinking_budget 32768    | thinking_budget 24576 (model max) | refused              | refused          |

Native strings: "dynamic" -> thinking_budget -1 on 2.5; "minimal" -> thinking_level MINIMAL on 3.x
(Flash only; the API rejects it elsewhere). Override with `extra.effort_map: {level: {thinking_config}|null}`.
"""

from __future__ import annotations

import re
import time
from typing import Any

from blindearth.providers.base import (
    Adapter,
    ProviderError,
    RateLimitError,
    UnsupportedConfigError,
)
from blindearth.providers.openai_compat import (
    _get,
    check_logprobs_config,
    mode_temperature,
    raise_for_thinking,
    resolve_effort_map,
)
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    RunConfig,
    Usage,
)

FAMILIES: dict[str, dict[str, Any]] = {
    "g25pro": {"map": {"off": None, "low": {"thinking_budget": 1024},
                       "medium": {"thinking_budget": 8192},
                       "high": {"thinking_budget": 24576},
                       "max": {"thinking_budget": 32768},
                       "dynamic": {"thinking_budget": -1}},
               "thinks_by_default": True, "logprobs": False},
    "g25flash": {"map": {"off": {"thinking_budget": 0}, "low": {"thinking_budget": 1024},
                         "medium": {"thinking_budget": 8192},
                         "high": {"thinking_budget": 24576},
                         "max": {"thinking_budget": 24576},
                         "dynamic": {"thinking_budget": -1}},
                 "thinks_by_default": True, "logprobs": False},
    "g3": {"map": {"off": None, "low": {"thinking_level": "LOW"},
                   "medium": {"thinking_level": "MEDIUM"},
                   "high": {"thinking_level": "HIGH"}, "max": None,
                   "minimal": {"thinking_level": "MINIMAL"}},
           "thinks_by_default": True, "logprobs": False},
    "g2": {"map": {"off": "__noop__", "low": None, "medium": None, "high": None, "max": None},
           "thinks_by_default": False, "logprobs": True},
}
DEFAULT_LOGPROBS_CAP = 5  # conservative static guess; probe records the real cap


def gemini_family(name: str) -> str:
    n = name.lower().split("/")[-1]
    if n.startswith("gemini-2.5-pro"):
        return "g25pro"
    if n.startswith("gemini-2.5"):
        return "g25flash"
    m = re.match(r"^gemini-(\d+)", n)
    if m and int(m.group(1)) >= 3:
        return "g3"
    return "g2"


def _finish_name(fr: Any) -> str:
    if fr is None:
        return "None"
    name = getattr(fr, "name", None)
    return str(name if name else fr).lower()


class GoogleAdapter(Adapter):
    kind = "google"

    def __init__(self, provider, model, api_key: str | None = None, *, client: Any = None):
        super().__init__(provider, model, api_key)
        self._client = client
        self.family = model.extra.get("gemini_family") or gemini_family(model.name)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown gemini_family {self.family!r}")
        self.profile = FAMILIES[self.family]

    @property
    def client(self):
        if self._client is None:
            from google import genai  # lazy: SDK import is heavy

            kw: dict[str, Any] = {}
            if self.api_key:
                kw["api_key"] = self.api_key
            if self.provider.extra.get("vertexai"):
                kw["vertexai"] = True
                for k in ("project", "location"):
                    if self.provider.extra.get(k):
                        kw[k] = self.provider.extra[k]
            self._client = genai.Client(**kw)
        return self._client

    # ---- mapping

    def _thinking_config(self, effort: str | None) -> dict[str, Any] | None:
        if effort is None:
            return None
        emap = resolve_effort_map(self, self.profile["map"])
        if effort not in emap:
            raise UnsupportedConfigError(f"{self.model.name}: effort={effort!r} unsupported")
        v = emap[effort]
        if v is None:
            raise UnsupportedConfigError(
                f"{self.model.name} ({self.family}): effort={effort!r} not supported")
        if v == "__noop__":
            return None
        return dict(v)

    def _thinks(self, effort: str | None, tc: dict[str, Any] | None) -> bool:
        if tc is not None and tc.get("thinking_budget") == 0:
            return False
        if effort is None:
            return bool(self.profile["thinks_by_default"]) or self.model.forced_thinking
        return effort != "off" and self.profile["thinks_by_default"]

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        tc = self._thinking_config(config.effort)
        native: dict[str, Any] = {"model": self.model.name}
        if tc:
            native["thinking_config"] = tc
        t = mode_temperature(config, mode)
        if t is not None:
            native["temperature"] = t
        k = check_logprobs_config(self, config, mode)
        if k is not None:
            native["response_logprobs"] = True
            native["logprobs"] = k
        thinks = self._thinks(config.effort, tc)
        if config.max_output_tokens is not None or thinks:
            native["max_output_tokens"] = raise_for_thinking(
                config.max_output_tokens or 0,
                config.effort if config.effort in ("low", "medium", "high", "max") else None,
                forced=thinks)
        if mode == ExtractionMode.SAMPLE and config.n_samples:
            native["n_requests"] = config.n_samples
        if config.seed is not None:
            native["seed"] = config.seed
        return native

    def default_capabilities(self) -> Capabilities:
        efforts = [k for k, v in self.profile["map"].items() if v is not None]
        lp = bool(self.profile["logprobs"])
        return Capabilities(
            logprobs=lp,
            top_logprobs_max=DEFAULT_LOGPROBS_CAP if lp else None,
            effort_param=self.family != "g2",
            supported_efforts=efforts,
            temperature_fixed_with_thinking=False,
            max_concurrency=16,
            supports_batch=False,
            notes=[f"static profile: family={self.family}"],
        )

    # ---- requests

    def _config(self, system_prompt: str | None, params: CallParams) -> dict[str, Any]:
        try:
            tc = self._thinking_config(params.effort)
        except UnsupportedConfigError as e:
            raise ProviderError(str(e), retryable=False) from e
        thinks = self._thinks(params.effort, tc)
        cfg: dict[str, Any] = {
            "max_output_tokens": raise_for_thinking(
                params.max_output_tokens,
                params.effort if params.effort in ("low", "medium", "high", "max") else None,
                forced=thinks),
        }
        if system_prompt:
            cfg["system_instruction"] = system_prompt
        if params.temperature is not None:
            cfg["temperature"] = params.temperature
        if params.seed is not None:
            cfg["seed"] = params.seed
        if params.logprobs:
            cfg["response_logprobs"] = True
            cfg["logprobs"] = params.top_logprobs or DEFAULT_LOGPROBS_CAP
        if tc or (params.store_thinking and thinks):
            tc = dict(tc or {})
            if params.store_thinking and thinks:
                tc["include_thoughts"] = True
            cfg["thinking_config"] = tc
        return cfg

    async def _call(self, prompt: str, cfg: dict[str, Any]) -> Any:
        try:
            return await self.client.aio.models.generate_content(
                model=self.model.name, contents=prompt, config=cfg)
        except Exception as e:  # noqa: BLE001
            raise map_google_error(e) from e

    @staticmethod
    def _first_token_logprobs(cand: Any) -> dict[str, float] | None:
        lr = _get(cand, "logprobs_result")
        tops = _get(lr, "top_candidates") or []
        chosen = _get(lr, "chosen_candidates") or []
        # skip whitespace-only leading tokens
        for i, top in enumerate(tops):
            ch = chosen[i] if i < len(chosen) else None
            ch_tok = _get(ch, "token") or ""
            if ch is not None and ch_tok.strip() == "":
                continue
            out: dict[str, float] = {}
            for c in _get(top, "candidates") or []:
                tok, lp = _get(c, "token"), _get(c, "log_probability")
                if tok is not None and lp is not None:
                    out[tok] = max(float(lp), out.get(tok, float("-inf")))
            if ch is not None and ch_tok not in out and _get(ch, "log_probability") is not None:
                out[ch_tok] = float(_get(ch, "log_probability"))
            return out or None
        return None

    async def classify(self, prompt: str, system_prompt: str | None,
                       params: CallParams) -> ClassifyResult:
        cfg = self._config(system_prompt, params)
        t0 = time.perf_counter()
        texts: list[str] = []
        thinking_texts: list[str | None] = []
        finish: list[str] = []
        usage = Usage()
        first_lp = None
        resolved = None
        raw = []
        for _ in range(max(1, params.n)):
            resp = await self._call(prompt, cfg)
            raw.append(resp)
            resolved = resolved or _get(resp, "model_version")
            um = _get(resp, "usage_metadata")
            thoughts_n = int(_get(um, "thoughts_token_count", 0) or 0)
            usage = usage + Usage(
                input_tokens=int(_get(um, "prompt_token_count", 0) or 0),
                output_tokens=int(_get(um, "candidates_token_count", 0) or 0) + thoughts_n,
                thinking_tokens=thoughts_n,
            )
            cands = _get(resp, "candidates") or []
            if not cands:
                fb = _get(resp, "prompt_feedback")
                texts.append("")
                thinking_texts.append(None)
                finish.append(f"blocked:{_finish_name(_get(fb, 'block_reason'))}")
                continue
            cand = cands[0]
            ans, th = [], []
            for part in _get(_get(cand, "content"), "parts") or []:
                txt = _get(part, "text")
                if not txt:
                    continue
                (th if _get(part, "thought") else ans).append(txt)
            texts.append("".join(ans).strip())
            thinking_texts.append("\n".join(th) if (params.store_thinking and th) else None)
            finish.append(_finish_name(_get(cand, "finish_reason")))
            if params.logprobs and first_lp is None:
                first_lp = self._first_token_logprobs(cand)
        native = {"model": self.model.name, **cfg}
        native.pop("system_instruction", None)
        native["n_requests"] = max(1, params.n)
        return ClassifyResult(
            texts=texts, usage=usage, latency_s=time.perf_counter() - t0,
            finish_reasons=finish, first_token_logprobs=first_lp if params.logprobs else None,
            thinking_texts=thinking_texts, resolved_model=resolved, native_params=native, raw=raw,
        )


def _retry_delay_from(e: Exception) -> float | None:
    details = getattr(e, "details", None)
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", str(details or e))
    return float(m.group(1)) if m else None


def map_google_error(e: Exception) -> Exception:
    if isinstance(e, ProviderError):
        return e
    try:
        from google.genai import errors as gerrors
    except ImportError:  # pragma: no cover
        gerrors = None
    code = getattr(e, "code", None)
    if gerrors is not None and isinstance(e, gerrors.APIError) or isinstance(code, int):
        status = int(code or 0)
        if status == 429:
            return RateLimitError(str(e), retry_after_s=_retry_delay_from(e))
        retryable = status in (408, 409) or status >= 500
        return ProviderError(str(e), status=status, retryable=retryable)
    name = type(e).__name__.lower()
    if "timeout" in name or "connect" in name or isinstance(e, (TimeoutError, ConnectionError)):
        return ProviderError(f"{type(e).__name__}: {e}", retryable=True)
    try:
        import httpx

        if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
            return ProviderError(f"{type(e).__name__}: {e}", retryable=True)
    except ImportError:  # pragma: no cover
        pass
    return e


__all__ = ["GoogleAdapter", "gemini_family", "FAMILIES", "map_google_error"]
