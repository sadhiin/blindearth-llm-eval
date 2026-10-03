"""Google Gemini adapter on the `google-genai` SDK (`client.aio.models.generate_content`).

Text only, no tools. Samples loop `n` times (candidate_count is not supported on every model).
Thinking parts (`part.thought`) are removed from the answer; `usage_metadata.thoughts_token_count`
is the thinking count and is added to output tokens, since Google bills thoughts as output but
reports them outside `candidates_token_count`. Logprobs via `GenerateContentConfig.response_logprobs`
(bool) + `logprobs` (int, top-k) on models that support them (few do; the probe decides); results
come back as `candidate.logprobs_result.{top_candidates[].candidates[], chosen_candidates[]}` with
`token` / `log_probability`.

Effort mapping (normalized -> `thinking_config`; family from the model name, override
`extra.gemini_family`):

| normalized | gemini-2.5-pro           | gemini-2.5-flash         | gemini-2.5-flash-lite     | gemini-3.x (levels per model) | gemini-2.0 / 1.5 |
|------------|--------------------------|--------------------------|---------------------------|-------------------------------|------------------|
| off        | refused (cannot disable) | thinking_budget 0        | thinking_budget 0         | refused (cannot disable)      | no-op            |
| low        | thinking_budget 1024     | thinking_budget 1024     | thinking_budget 1024      | thinking_level LOW            | refused          |
| medium     | thinking_budget 8192     | thinking_budget 8192     | thinking_budget 8192      | thinking_level MEDIUM *       | refused          |
| high       | thinking_budget 24576    | thinking_budget 24576    | thinking_budget 24576     | thinking_level HIGH           | refused          |
| max        | thinking_budget 32768    | thinking_budget 24576 (model max) | 24576 (model max) | refused                       | refused          |

Native strings: "dynamic" -> thinking_budget -1 on 2.5; "minimal" -> thinking_level MINIMAL on the
3.x models that list it (*). Per-model 3.x levels (from the Gemini API thinking page):
gemini-3-pro*: low, high (no medium); gemini-3.1+-pro*: low, medium, high; gemini-3-flash*,
gemini-3.5-flash*, gemini-3.6-flash*, any *flash-lite*: minimal, low, medium, high;
gemini-3.7+/other flash: low, medium, high. `gemini-*-latest` aliases are treated as 3.x with
low/medium/high. Override per model with `extra.thinking_levels: [..]` or
`extra.effort_map: {level: {thinking_config}|null}` (an explicit effort_map entry is trusted).
Budgets are range-checked per 2.5 family, and a config carrying both `thinking_level` and
`thinking_budget` is refused (the API rejects that combination). Default-thinking (no effort sent)
follows `blindearth.thinking_defaults` (2.5 Flash-Lite off; 2.5 Pro and 3.x always on) plus the
registry `extra.thinks_by_default` override.

Batch Mode (Gemini Developer API only; Vertex rejects inlined requests, so `supports_batch` is
False there): `client.aio.batches.create(model=..., src=[InlinedRequest dicts])`, polled with
`client.aio.batches.get(name=...)`; results from `job.dest.inlined_responses` (each has `response`
or `error`, plus `metadata`). Each inlined request carries `metadata={"key": "<custom_id>__s<k>"}`,
which the API echoes back on the matching `InlinedResponse.metadata`; that key is the primary
join. Responses are also documented to be "in the same order as the input requests", so when a
response lacks metadata the in-process submit order is used as a fallback (only while this process
still remembers the submission). Request configs are built by the same `_config` used by
`classify`, so batch and live calls send identical `GenerateContentConfig`s. Inline submissions
are capped at 20 MB, so large submissions are split across several jobs and the returned batch id
is the comma-joined job names; `poll_batch` returns None until every job is terminal. Job-level
failure (FAILED / CANCELLED / EXPIRED with no responses) never raises from `poll_batch` (that
would make the executor poll forever); the known items get `error` set instead. The 50% batch
price is applied by `pricing.cost_usd(batch=True)` in the planner/executor, not here.

Verified against (2026-10-03):
- https://ai.google.dev/gemini-api/docs/batch-mode : inline src list, 20 MB inline cap, JSONL
  `key` field (file mode only), `batches.get`, `dest.inlined_responses` with `.response` / `.error`,
  JOB_STATE_* names, 50% price.
- https://github.com/googleapis/python-genai/blob/main/google/genai/types.py : `InlinedRequest`
  {model, contents, metadata: dict[str, str], config}, `InlinedResponse` {response, metadata, error:
  JobError{code, message, details}}, `BatchJobDestination.inlined_responses` ordering note,
  `JobState` (QUEUED, PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLING, CANCELLED, PAUSED, EXPIRED,
  UPDATING, PARTIALLY_SUCCEEDED), `ThinkingConfig` {include_thoughts, thinking_budget (0 disabled,
  -1 automatic), thinking_level: MINIMAL|LOW|MEDIUM|HIGH}, `GenerateContentConfig.response_logprobs`
  / `logprobs`, `LogprobsResult` fields.
- https://github.com/googleapis/python-genai/blob/main/google/genai/batches.py : metadata is
  passed through on create and on responses; inlined_requests raise on Vertex.
- https://ai.google.dev/gemini-api/docs/thinking : per-model 3.x thinking levels and defaults.
- https://firebase.google.com/docs/ai-logic/thinking : 2.5 budgets (Pro 128..32768, cannot
  disable; Flash 0..24576; Flash-Lite 512..24576, off by default; -1 dynamic) and "setting both
  thinkingLevel and thinkingBudget returns an error".
"""

from __future__ import annotations

import json
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
from blindearth.thinking_defaults import always_thinks as td_always_thinks
from blindearth.thinking_defaults import thinks_by_default as td_thinks_by_default
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    RunConfig,
    Usage,
)

_G25_MAP = {"off": {"thinking_budget": 0}, "low": {"thinking_budget": 1024},
            "medium": {"thinking_budget": 8192}, "high": {"thinking_budget": 24576},
            "max": {"thinking_budget": 24576}, "dynamic": {"thinking_budget": -1}}

FAMILIES: dict[str, dict[str, Any]] = {
    # budget_range: (min, max) for positive budgets; 0 allowed only when "off" maps to 0.
    "g25pro": {"map": {**_G25_MAP, "off": None, "max": {"thinking_budget": 32768}},
               "budget_range": (128, 32768), "thinks_by_default": True, "logprobs": False},
    "g25flash": {"map": dict(_G25_MAP), "budget_range": (1, 24576),
                 "thinks_by_default": True, "logprobs": False},
    "g25flashlite": {"map": dict(_G25_MAP), "budget_range": (512, 24576),
                     "thinks_by_default": False, "logprobs": False},
    "g3": {"map": {"off": None, "low": {"thinking_level": "LOW"},
                   "medium": {"thinking_level": "MEDIUM"},
                   "high": {"thinking_level": "HIGH"}, "max": None,
                   "minimal": {"thinking_level": "MINIMAL"}},
           "thinks_by_default": True, "logprobs": False},
    "g2": {"map": {"off": "__noop__", "low": None, "medium": None, "high": None, "max": None},
           "thinks_by_default": False, "logprobs": True},
}
DEFAULT_LOGPROBS_CAP = 5  # conservative static guess; probe records the real cap

INLINE_BATCH_MAX_BYTES = 15_000_000  # API cap is 20 MB per inline batch request; keep headroom
_RUNNING_STATES = {"JOB_STATE_UNSPECIFIED", "JOB_STATE_QUEUED", "JOB_STATE_PENDING",
                   "JOB_STATE_RUNNING", "JOB_STATE_CANCELLING", "JOB_STATE_PAUSED",
                   "JOB_STATE_UPDATING"}
_SAMPLE_SEP = "__s"


def gemini_family(name: str) -> str:
    n = name.lower().split("/")[-1]
    if n.startswith("gemini-2.5-pro"):
        return "g25pro"
    if n.startswith("gemini-2.5-flash-lite"):
        return "g25flashlite"
    if n.startswith("gemini-2.5"):
        return "g25flash"
    m = re.match(r"^gemini-(\d+)", n)
    if m:
        # Versioned names (incl. versioned aliases like gemini-1.5-flash-latest) follow their version.
        return "g3" if int(m.group(1)) >= 3 else "g2"
    if n.startswith("gemini-") and n.endswith("-latest"):
        return "g3"  # moving alias for the current (thinking) generation
    return "g2"


def gemini3_levels(name: str) -> set[str]:
    """Thinking levels (lower-case) a Gemini 3.x model accepts, per the Gemini API thinking page."""
    n = name.lower().split("/")[-1]
    m = re.match(r"^gemini-(\d+)(?:\.(\d+))?-(.*)$", n)
    if not m:
        return {"low", "medium", "high"}
    ver = (int(m.group(1)), int(m.group(2) or 0))
    rest = m.group(3)
    full = {"minimal", "low", "medium", "high"}
    if rest.startswith("pro"):
        return {"low", "high"} if ver == (3, 0) else {"low", "medium", "high"}
    if "flash-lite" in rest:
        return full
    if rest.startswith("flash"):
        return full if ver in ((3, 0), (3, 5), (3, 6)) else {"low", "medium", "high"}
    return {"low", "medium", "high"}


def _finish_name(fr: Any) -> str:
    if fr is None:
        return "None"
    name = getattr(fr, "name", None)
    return str(name if name else fr).lower()


def _state_name(state: Any) -> str:
    name = getattr(state, "name", None)
    s = str(name if name else state or "JOB_STATE_UNSPECIFIED")
    return s if s.startswith("JOB_STATE_") else s.upper()


def _job_error_text(err: Any) -> str:
    if err is None:
        return ""
    code, msg = _get(err, "code"), _get(err, "message")
    if code is None and msg is None:
        return str(err)
    return f"{code}: {msg}" if code is not None else str(msg)


class GoogleAdapter(Adapter):
    kind = "google"
    supports_batch = True

    def __init__(self, provider, model, api_key: str | None = None, *, client: Any = None):
        super().__init__(provider, model, api_key)
        self._client = client
        self.family = model.extra.get("gemini_family") or gemini_family(model.name)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown gemini_family {self.family!r}")
        self.profile = FAMILIES[self.family]
        lv = model.extra.get("thinking_levels")
        self.levels: set[str] | None = (
            {str(x).lower() for x in lv} if lv else
            (gemini3_levels(model.name) if self.family == "g3" else None))
        # Inline batch requests are Gemini Developer API only (the SDK rejects them on Vertex).
        if provider.extra.get("vertexai"):
            self.supports_batch = False
        # job name -> ordered request keys, for the order fallback (in-process only)
        self._batch_keys: dict[str, list[str]] = {}

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

    def _user_mapped(self, effort: str) -> bool:
        return any(isinstance((src or {}).get("effort_map"), dict)
                   and effort in src["effort_map"]
                   for src in (self.provider.extra, self.model.extra))

    def _thinking_config(self, effort: str | None) -> dict[str, Any] | None:
        if effort is None:
            return None
        emap = resolve_effort_map(self, self.profile["map"])
        if effort not in emap:
            raise UnsupportedConfigError(
                f"{self.model.name} ({self.family}): effort={effort!r} unsupported; "
                f"accepted: {sorted(k for k, v in emap.items() if v is not None)}")
        v = emap[effort]
        if v is None:
            raise UnsupportedConfigError(
                f"{self.model.name} ({self.family}): effort={effort!r} not supported")
        if v == "__noop__":
            return None
        if not isinstance(v, dict):
            raise UnsupportedConfigError(
                f"{self.model.name}: effort_map[{effort!r}] must be a thinking_config dict")
        tc = dict(v)
        if "thinking_level" in tc and "thinking_budget" in tc:
            raise UnsupportedConfigError(
                f"{self.model.name}: thinking_level and thinking_budget cannot both be set")
        trusted = self._user_mapped(effort)
        lvl = tc.get("thinking_level")
        if lvl is not None and not trusted:
            if self.levels is not None and str(lvl).lower() not in self.levels:
                raise UnsupportedConfigError(
                    f"{self.model.name}: thinking_level {lvl} not supported "
                    f"(model accepts {sorted(self.levels)})")
        b = tc.get("thinking_budget")
        rng = self.profile.get("budget_range")
        if b is not None and rng and not trusted:
            b = int(b)
            off_ok = self.profile["map"].get("off") is not None
            if not (b == -1 or (b == 0 and off_ok) or rng[0] <= b <= rng[1]):
                raise UnsupportedConfigError(
                    f"{self.model.name}: thinking_budget {b} outside {rng[0]}..{rng[1]}"
                    + ("" if off_ok else " (thinking cannot be disabled)"))
        return tc

    def _thinks_by_default(self) -> bool:
        """Kept consistent with blindearth.thinking_defaults (which labels runs): a registry
        `extra.thinks_by_default` wins; otherwise the shared table OR this family's profile
        (the union only matters for max_output_tokens floors, where erring high is safe)."""
        ov = self.model.extra.get("thinks_by_default")
        if ov is not None:
            return bool(ov)
        return (bool(self.profile["thinks_by_default"])
                or td_thinks_by_default("google", self.model.name))

    def _thinks(self, effort: str | None, tc: dict[str, Any] | None) -> bool:
        if self.model.forced_thinking or td_always_thinks("google", self.model.name):
            return True  # such families refuse "off" in _thinking_config
        if tc is not None and tc.get("thinking_budget") == 0:
            return False
        if effort is None:
            return self._thinks_by_default()
        if effort == "off":
            return False
        return tc is not None

    def _supported_efforts(self) -> list[str]:
        out = []
        for k in resolve_effort_map(self, self.profile["map"]):
            try:
                self._thinking_config(k)
            except UnsupportedConfigError:
                continue
            out.append(k)
        return out

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
        lp = bool(self.profile["logprobs"])
        return Capabilities(
            logprobs=lp,
            top_logprobs_max=DEFAULT_LOGPROBS_CAP if lp else None,
            effort_param=self.family != "g2",
            supported_efforts=self._supported_efforts(),
            temperature_fixed_with_thinking=False,
            max_concurrency=16,
            supports_batch=bool(self.supports_batch),
            notes=[f"static profile: family={self.family}"]
            + ([f"thinking levels: {sorted(self.levels)}"] if self.levels else []),
        )

    # ---- requests

    def _config(self, system_prompt: str | None, params: CallParams) -> dict[str, Any]:
        """GenerateContentConfig dict; shared by `classify` and `submit_batch`."""
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

    def _parse_response(self, resp: Any, store_thinking: bool, want_logprobs: bool
                        ) -> tuple[str, str | None, str, Usage, dict[str, float] | None]:
        """-> (answer text, thinking text, finish reason, usage, first-token logprobs)."""
        um = _get(resp, "usage_metadata")
        thoughts_n = int(_get(um, "thoughts_token_count", 0) or 0)
        usage = Usage(
            input_tokens=int(_get(um, "prompt_token_count", 0) or 0),
            output_tokens=int(_get(um, "candidates_token_count", 0) or 0) + thoughts_n,
            thinking_tokens=thoughts_n,
        )
        cands = _get(resp, "candidates") or []
        if not cands:
            fb = _get(resp, "prompt_feedback")
            return "", None, f"blocked:{_finish_name(_get(fb, 'block_reason'))}", usage, None
        cand = cands[0]
        ans, th = [], []
        for part in _get(_get(cand, "content"), "parts") or []:
            txt = _get(part, "text")
            if not txt:
                continue
            (th if _get(part, "thought") else ans).append(txt)
        thinking = "\n".join(th) if (store_thinking and th) else None
        lp = self._first_token_logprobs(cand) if want_logprobs else None
        return "".join(ans).strip(), thinking, _finish_name(_get(cand, "finish_reason")), usage, lp

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
            text, th, fr, u, lp = self._parse_response(
                resp, params.store_thinking, params.logprobs and first_lp is None)
            texts.append(text)
            thinking_texts.append(th)
            finish.append(fr)
            usage = usage + u
            if first_lp is None:
                first_lp = lp
        native = {"model": self.model.name, **cfg}
        native.pop("system_instruction", None)
        native["n_requests"] = max(1, params.n)
        return ClassifyResult(
            texts=texts, usage=usage, latency_s=time.perf_counter() - t0,
            finish_reasons=finish, first_token_logprobs=first_lp if params.logprobs else None,
            thinking_texts=thinking_texts, resolved_model=resolved, native_params=native, raw=raw,
        )

    # ---- Batch Mode (inline requests)

    async def submit_batch(self, items: list[tuple[str, str, str | None, CallParams]]) -> str:
        """Inline batch job(s). Returns one job name, or several joined with ','."""
        if not self.supports_batch:
            raise ProviderError("Gemini inline batch requests are not available on Vertex AI",
                                retryable=False)
        requests: list[dict[str, Any]] = []
        sizes: list[int] = []
        for custom_id, prompt, system_prompt, params in items:
            cfg = self._config(system_prompt, params)
            for k in range(max(1, params.n)):
                req = {"contents": prompt, "config": cfg,
                       "metadata": {"key": f"{custom_id}{_SAMPLE_SEP}{k}"}}
                requests.append(req)
                sizes.append(len(json.dumps(req, default=str)) + 64)
        chunks: list[list[dict[str, Any]]] = []
        cur: list[dict[str, Any]] = []
        cur_bytes = 0
        for req, sz in zip(requests, sizes):
            if cur and cur_bytes + sz > INLINE_BATCH_MAX_BYTES:
                chunks.append(cur)
                cur, cur_bytes = [], 0
            cur.append(req)
            cur_bytes += sz
        if cur:
            chunks.append(cur)
        names: list[str] = []
        for i, chunk in enumerate(chunks):
            try:
                job = await self.client.aio.batches.create(
                    model=self.model.name, src=chunk,
                    config={"display_name": f"blindearth-{self.model.id}-{i}"[:120]})
            except Exception as e:  # noqa: BLE001
                if names:  # earlier jobs already exist and will bill; surface their names
                    raise ProviderError(
                        f"batch submit failed after creating {names}: {e}",
                        retryable=False) from e
                raise map_google_error(e) from e
            name = _get(job, "name")
            names.append(name)
            self._batch_keys[name] = [r["metadata"]["key"] for r in chunk]
        return ",".join(names)

    async def poll_batch(self, batch_id: str) -> dict[str, ClassifyResult] | None:
        jobs = []
        for name in [n for n in batch_id.split(",") if n]:
            try:
                job = await self.client.aio.batches.get(name=name)
            except Exception as e:  # noqa: BLE001
                raise map_google_error(e) from e
            if _state_name(_get(job, "state")) in _RUNNING_STATES:
                return None
            jobs.append((name, job))
        samples: dict[str, list[tuple[int, Any, str | None]]] = {}
        for name, job in jobs:
            self._collect(name, job, samples)
        out: dict[str, ClassifyResult] = {}
        for base, rows in samples.items():
            rows.sort(key=lambda x: x[0])
            res = ClassifyResult(texts=[], usage=Usage(), latency_s=0.0,
                                 native_params={"batch_id": batch_id, "n_requests": len(rows)})
            errors = []
            for _, resp, err in rows:
                if err is not None:
                    errors.append(err)
                    continue
                text, th, fr, u, lp = self._parse_response(resp, True, True)
                res.texts.append(text)
                res.thinking_texts.append(th)
                res.finish_reasons.append(fr)
                res.usage = res.usage + u
                res.resolved_model = res.resolved_model or _get(resp, "model_version")
                if res.first_token_logprobs is None:
                    res.first_token_logprobs = lp
            if errors:
                res.error = f"{len(errors)}/{len(rows)} batch samples failed: " + "; ".join(errors)
            out[base] = res
        return out

    def _collect(self, name: str, job: Any,
                 samples: dict[str, list[tuple[int, Any, str | None]]]) -> None:
        """Add (sample_idx, response, error) rows of one terminal job, keyed by custom_id."""
        state = _state_name(_get(job, "state"))
        responses = _get(_get(job, "dest"), "inlined_responses") or []
        known = self._batch_keys.get(name)

        def add(key: str, resp: Any, err: str | None) -> None:
            base, sep, k = key.rpartition(_SAMPLE_SEP)
            if not sep or not base:
                base, k = key, "0"
            samples.setdefault(base, []).append((int(k) if k.isdigit() else 0, resp, err))

        if not responses:
            # Job-level failure (or no results at all): mark what we know; the executor reports
            # anything we cannot name as "missing from batch output".
            msg = f"batch {state}" + (f": {_job_error_text(_get(job, 'error'))}"
                                      if _get(job, "error") is not None else "")
            for key in known or []:
                add(key, None, msg)
            return
        order_ok = known is not None and len(known) == len(responses)
        for i, r in enumerate(responses):
            key = (_get(r, "metadata") or {}).get("key")
            if key is None and order_ok:
                key = known[i]
            if key is None:
                continue  # unattributable; surfaces as missing
            err = _get(r, "error")
            resp = _get(r, "response")
            if err is not None or resp is None:
                add(key, None, _job_error_text(err) or f"no response ({state})")
            else:
                add(key, resp, None)


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


__all__ = ["GoogleAdapter", "gemini_family", "gemini3_levels", "FAMILIES", "map_google_error"]
