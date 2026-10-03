"""OpenRouter adapter: the OpenAI-compatible adapter pointed at https://openrouter.ai/api/v1.

Reasoning is passed through with OpenRouter's unified `reasoning` object, which it translates
for the upstream provider (effort for OpenAI-style models, a token budget for Anthropic/Gemini).

Effort mapping (normalized -> body):

| normalized | body                                         |
|------------|----------------------------------------------|
| off        | `reasoning: {enabled: false}` (refused by upstream models that always reason) |
| low        | `reasoning: {effort: "low"}`                 |
| medium     | `reasoning: {effort: "medium"}`              |
| high       | `reasoning: {effort: "high"}`                |
| max        | `reasoning: {max_tokens: 32000}` (largest explicit budget we send) |
| other str  | `reasoning: {effort: <str>}`                 |

When logprobs are requested, `provider: {require_parameters: true}` is added so OpenRouter
routes only to upstreams that honour them instead of dropping the parameter silently.
Thinking text comes back in `message.reasoning` and is kept only with `store_thinking`; it is
always excluded from the answer text. `resolved_model` is the response `model`, with the
upstream `provider` appended when present (e.g. `anthropic/claude-opus-5-5@Anthropic`).
"""

from __future__ import annotations

from typing import Any

from blindearth.providers.base import UnsupportedConfigError
from blindearth.providers.openai_compat import (
    NORMALIZED_EFFORTS,
    OpenAICompatAdapter,
    resolve_effort_map,
)
from blindearth.types import CallParams, Capabilities, ClassifyResult, ExtractionMode, RunConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(OpenAICompatAdapter):
    kind = "openrouter"
    default_base_url = OPENROUTER_BASE_URL
    supports_n_default = False  # many upstreams ignore `n`; loop instead
    default_logprobs = False
    default_top_logprobs_max = None

    EFFORT_MAP: dict[str, Any] = {
        "off": {"reasoning": {"enabled": False}},
        "low": {"reasoning": {"effort": "low"}},
        "medium": {"reasoning": {"effort": "medium"}},
        "high": {"reasoning": {"effort": "high"}},
        "max": {"reasoning": {"max_tokens": 32000}},
    }

    def _effort_params(self, effort: str | None) -> dict[str, Any]:
        if effort is None:
            return {}
        emap = resolve_effort_map(self, self.EFFORT_MAP)
        if effort in emap:
            v = emap[effort]
            if v is None:
                raise UnsupportedConfigError(f"effort={effort!r} refused for {self.model.name}")
            return {k: (dict(x) if isinstance(x, dict) else x) for k, x in v.items()}
        if effort in NORMALIZED_EFFORTS:
            raise UnsupportedConfigError(f"effort={effort!r} unsupported")
        return {"reasoning": {"effort": effort}}

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        h.setdefault("HTTP-Referer", self.provider.extra.get("referer",
                                                             "https://github.com/blindearth"))
        h.setdefault("X-Title", self.provider.extra.get("title", "blindearth"))
        return h

    def _build_body(self, prompt: str, system_prompt: str | None, params: CallParams,
                    n: int) -> dict[str, Any]:
        body = super()._build_body(prompt, system_prompt, params, n)
        if params.logprobs:
            prov = dict(body.get("provider") or {})
            prov["require_parameters"] = True
            body["provider"] = prov
        return body

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        native = super().map_config(config, mode)
        if native.get("logprobs"):
            native["provider"] = {"require_parameters": True}
        return native

    def default_capabilities(self) -> Capabilities:
        caps = super().default_capabilities()
        caps.effort_param = True
        caps.supported_efforts = ["off", "low", "medium", "high", "max"]
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


__all__ = ["OpenRouterAdapter", "OPENROUTER_BASE_URL"]
