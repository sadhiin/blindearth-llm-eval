"""Run hash and the effective (fully resolved) run configuration.

The run hash covers everything that changes model output: the eval spec id (which already
covers mask hash, truth rule, grid, placement, coordinate format, prompt and spec system prompt),
provider and model name, quantization, the resolved model version, the normalized effective
config (temperature, samples, top_logprobs, max tokens, effort, config system prompt, seed),
the extraction mode and the repeat index. Rate-limit fields never enter the hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from blindearth.thinking_defaults import KNOWN_KINDS, always_thinks, thinks_by_default
from blindearth.types import ExtractionMode, ExtractionSpec, ModelSpec, RunConfig

HASH_VERSION = 1

# Floor for max_output_tokens on thinking runs, by effort, so the answer is not cut off.
THINKING_MIN_OUTPUT_TOKENS = {"low": 4096, "medium": 8192, "high": 16384, "max": 32768}
DEFAULT_THINKING_MIN_OUTPUT_TOKENS = 8192
DEFAULT_PLAIN_MAX_OUTPUT_TOKENS = 16
DEFAULT_TOP_LOGPROBS = 20


def infer_provider_kind(model: ModelSpec) -> str | None:
    """Best-effort provider kind for a model when the ProviderSpec is not at hand.

    The provider id usually equals its kind (`anthropic`, `openai`, ...); otherwise None, which
    makes the thinking table match the model name against every vendor's patterns.
    """
    pid = (model.provider or "").lower()
    return pid if pid in KNOWN_KINDS else None


def model_thinks_by_default(model: ModelSpec, provider_kind: str | None = None) -> bool:
    """Registry override `thinks_by_default: bool` first, then the built-in table."""
    override = (model.extra or {}).get("thinks_by_default")
    if isinstance(override, bool):
        return override
    kind = provider_kind or infer_provider_kind(model)
    return thinks_by_default(kind, model.name)


def model_always_thinks(model: ModelSpec, provider_kind: str | None = None) -> bool:
    """Reasoning cannot be turned off: registry `forced_thinking`, or the built-in table
    (unless the registry says `thinks_by_default: false`). Such runs are starred in reports."""
    if model.forced_thinking:
        return True
    if (model.extra or {}).get("thinks_by_default") is False:
        return False
    return always_thinks(provider_kind or infer_provider_kind(model), model.name)


def is_thinking(config: RunConfig, model: ModelSpec, provider_kind: str | None = None) -> bool:
    """Whether a run of `model` under `config` reasons before answering.

    - `forced_thinking` on the model, or a model that cannot turn reasoning off: always True
      (an explicit effort "off" stays thinking; the adapter refuses or maps it to the minimum).
    - effort set and not "off": True.
    - effort "off": False.
    - effort None (provider default): True if the model reasons by default
      (`blindearth.thinking_defaults`, overridable with `thinks_by_default` in the registry).

    `provider_kind` is the ProviderSpec.kind when the caller has it; otherwise it is inferred
    from `model.provider`.
    """
    if model_always_thinks(model, provider_kind):
        return True
    kind = provider_kind or infer_provider_kind(model)
    if config.effort is None:
        return model_thinks_by_default(model, kind)
    return config.effort != "off"


def thinking_floor(effort: str | None) -> int:
    return THINKING_MIN_OUTPUT_TOKENS.get((effort or "").lower(), DEFAULT_THINKING_MIN_OUTPUT_TOKENS)


def effective_config(
    config: RunConfig,
    extraction: ExtractionSpec,
    mode: ExtractionMode,
    *,
    thinking: bool,
    top_logprobs_cap: int | None = None,
) -> RunConfig:
    """Fill every output-affecting field so the stored config says exactly what was sent.

    Idempotent: applying it to its own result returns an equal config.
    """
    mode = ExtractionMode(mode)
    if mode == ExtractionMode.AUTO:
        raise ValueError("extraction mode must be resolved before building the effective config")
    c = dataclasses.replace(config)
    if mode == ExtractionMode.GREEDY:
        c.temperature = 0.0
        c.n_samples = 1
        c.top_logprobs = None
    elif mode == ExtractionMode.LOGPROBS:
        c.temperature = float(c.temperature if c.temperature is not None else extraction.temperature)
        c.n_samples = 1
        k = c.top_logprobs if c.top_logprobs is not None else DEFAULT_TOP_LOGPROBS
        if top_logprobs_cap is not None:
            k = min(k, top_logprobs_cap)
        c.top_logprobs = max(1, int(k))
    else:  # SAMPLE
        c.temperature = float(c.temperature if c.temperature is not None else extraction.temperature)
        c.n_samples = int(c.n_samples if c.n_samples is not None else extraction.n_samples)
        c.top_logprobs = None
    if thinking:
        c.max_output_tokens = max(int(c.max_output_tokens or 0), thinking_floor(c.effort))
    elif c.max_output_tokens is None:
        c.max_output_tokens = DEFAULT_PLAIN_MAX_OUTPUT_TOKENS
    return c


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def run_hash(
    spec_id: str,
    model: ModelSpec,
    resolved_version: str | None,
    config: RunConfig,
    extraction: ExtractionSpec,
    mode: ExtractionMode,
    repeat_idx: int,
) -> str:
    mode = ExtractionMode(mode)
    if mode == ExtractionMode.AUTO:
        raise ValueError("run_hash needs a resolved extraction mode, not AUTO")
    eff = effective_config(config, extraction, mode, thinking=is_thinking(config, model))
    payload = {
        "v": HASH_VERSION,
        "spec_id": spec_id,
        "provider": model.provider,
        "model_name": model.name,
        "quant": model.quant,
        "resolved_version": resolved_version or model.name,
        "config": eff.normalized(),
        "mode": mode.value,
        "repeat_idx": int(repeat_idx),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
