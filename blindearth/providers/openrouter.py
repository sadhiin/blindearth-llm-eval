"""OpenRouter adapter: the OpenAI-compatible adapter pointed at https://openrouter.ai/api/v1.

Reasoning is passed through with OpenRouter's unified `reasoning` object, which it translates
for the upstream provider (native effort for OpenAI/Grok, `thinkingLevel` for Gemini 3, a token
budget of `max_tokens * ratio` for Anthropic).

Effort mapping (normalized -> body):

| normalized | body                                                                |
|------------|---------------------------------------------------------------------|
| off        | `reasoning: {effort: "none"}`; refused when `forced_thinking` is set |
| low        | `reasoning: {effort: "low"}`                                        |
| medium     | `reasoning: {effort: "medium"}`                                     |
| high       | `reasoning: {effort: "high"}`                                       |
| max        | `reasoning: {effort: "max"}` (highest documented level, ~95% budget) |
| minimal    | `reasoning: {effort: "minimal"}` (provider-native)                  |
| xhigh      | `reasoning: {effort: "xhigh"}` (provider-native)                    |
| other str  | refused (OpenRouter's effort vocabulary is closed)                  |

`effort="none"` as a native string is refused in favour of `off` so the same request does not
get two run hashes. Override per provider/model with `extra.effort_map` as for the compat
adapter (e.g. `max: {reasoning: {max_tokens: 24000}}` for an explicit budget).

Silent drops: without `provider.require_parameters`, OpenRouter may route to an upstream that
"will ignore unknown parameters". We therefore add `provider: {require_parameters: true}`
whenever `reasoning` or `logprobs` is sent, so an upstream that cannot honour them is never
picked (OpenRouter errors instead). Consequence: for non-reasoning models leave `effort` unset
rather than `off`. Models whose reasoning is mandatory (`reasoning.mandatory` in
`GET /api/v1/models`) reject `effort: "none"`; mark them `forced_thinking` to refuse `off` at
plan time.

`top_logprobs` is 0..20 on OpenRouter. Thinking text comes back in `message.reasoning` and is
kept only with `store_thinking`; it is always excluded from the answer text. `resolved_model` is
the response `model`, with the upstream `provider` appended when present
(e.g. `anthropic/claude-opus-5-5@Anthropic`).

Docs checked 2026-10-03:
- https://openrouter.ai/docs/guides/best-practices/reasoning-tokens (effort values
  max|xhigh|high|medium|low|minimal|none, `effort: "none"` disables reasoning, `mandatory`,
  `enabled`, `max_tokens`, `exclude`, Anthropic budget formula)
- https://openrouter.ai/docs/api/reference/parameters (logprobs, top_logprobs 0..20)
- https://openrouter.ai/docs/guides/routing/provider-selection (require_parameters)
"""

from __future__ import annotations

from typing import Any

from blindearth.providers.base import ProviderError, UnsupportedConfigError
from blindearth.providers.openai_compat import (
    NORMALIZED_EFFORTS,
    OpenAICompatAdapter,
    resolve_effort_map,
)
from blindearth.types import CallParams, Capabilities, ClassifyResult, ExtractionMode, RunConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TOP_LOGPROBS_MAX = 20
# Documented `reasoning.effort` values beyond our normalized levels.
OPENROUTER_NATIVE_EFFORTS = ("minimal", "xhigh")


class OpenRouterAdapter(OpenAICompatAdapter):
    kind = "openrouter"
    default_base_url = OPENROUTER_BASE_URL
    supports_n_default = False  # many upstreams ignore `n`; loop instead
    default_logprobs = False
    default_top_logprobs_max = None

    EFFORT_MAP: dict[str, Any] = {
        "off": {"reasoning": {"effort": "none"}},
        "low": {"reasoning": {"effort": "low"}},
        "medium": {"reasoning": {"effort": "medium"}},
        "high": {"reasoning": {"effort": "high"}},
        "max": {"reasoning": {"effort": "max"}},
    }

    def _effort_params(self, effort: str | None) -> dict[str, Any]:
        if effort is None:
            return {}
        if effort == "none":
            raise UnsupportedConfigError("use effort='off' instead of the native 'none'")
        if effort == "off" and self.model.forced_thinking:
            raise UnsupportedConfigError(
                f"effort='off' refused: {self.model.name} always reasons (forced_thinking); "
                "OpenRouter rejects effort 'none' on mandatory-reasoning models"
            )
        emap = resolve_effort_map(self, self.EFFORT_MAP)
        if effort in emap:
            v = emap[effort]
            if v is None:
                raise UnsupportedConfigError(f"effort={effort!r} refused for {self.model.name}")
            return {k: (dict(x) if isinstance(x, dict) else x) for k, x in v.items()}
        if effort in NORMALIZED_EFFORTS:
            raise UnsupportedConfigError(f"effort={effort!r} unsupported")
        if effort in OPENROUTER_NATIVE_EFFORTS:
            return {"reasoning": {"effort": effort}}
        raise UnsupportedConfigError(
            f"effort={effort!r} is not an OpenRouter reasoning.effort value "
            f"(documented: max, xhigh, high, medium, low, minimal, none)"
        )

    @staticmethod
    def _needs_require_parameters(body: dict[str, Any]) -> bool:
        return bool(body.get("logprobs")) or "reasoning" in body

    @staticmethod
    def _add_require_parameters(body: dict[str, Any]) -> None:
        prov = dict(body.get("provider") or {})
        prov["require_parameters"] = True
        body["provider"] = prov

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        h.setdefault("HTTP-Referer", self.provider.extra.get("referer",
                                                             "https://github.com/blindearth"))
        h.setdefault("X-Title", self.provider.extra.get("title", "blindearth"))
        return h

    def _build_body(self, prompt: str, system_prompt: str | None, params: CallParams,
                    n: int) -> dict[str, Any]:
        if params.logprobs and (params.top_logprobs or 0) > OPENROUTER_TOP_LOGPROBS_MAX:
            raise ProviderError(
                f"top_logprobs={params.top_logprobs} exceeds OpenRouter max "
                f"{OPENROUTER_TOP_LOGPROBS_MAX}", retryable=False)
        body = super()._build_body(prompt, system_prompt, params, n)
        if self._needs_require_parameters(body):
            self._add_require_parameters(body)
        return body

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        native = super().map_config(config, mode)
        k = native.get("top_logprobs")
        if k is not None and k > OPENROUTER_TOP_LOGPROBS_MAX:
            raise UnsupportedConfigError(
                f"top_logprobs={k} exceeds OpenRouter max {OPENROUTER_TOP_LOGPROBS_MAX}")
        if self._needs_require_parameters(native):
            self._add_require_parameters(native)
        return native

    def default_capabilities(self) -> Capabilities:
        caps = super().default_capabilities()
        caps.effort_param = True
        caps.supported_efforts = ["off", "low", "medium", "high", "max", "minimal", "xhigh"]
        caps.notes = ["static guess; upstream-dependent, run the capability probe"]
        return caps

    async def classify(self, prompt: str, system_prompt: str | None,
                       params: CallParams) -> ClassifyResult:
        res = await super().classify(prompt, system_prompt, params)
        raws = res.raw if isinstance(res.raw, list) else []
        upstream = next((r.get("provider") for r in raws if isinstance(r, dict)
                         and r.get("provider")), None)
        if upstream and res.resolved_model:
            res.resolved_model = f"{res.resolved_model}@{upstream}"
        return res


__all__ = ["OpenRouterAdapter", "OPENROUTER_BASE_URL", "OPENROUTER_TOP_LOGPROBS_MAX"]
