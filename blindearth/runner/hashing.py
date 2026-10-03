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

from blindearth.types import ExtractionMode, ExtractionSpec, ModelSpec, RunConfig

HASH_VERSION = 1

# Floor for max_output_tokens on thinking runs, by effort, so the answer is not cut off.
THINKING_MIN_OUTPUT_TOKENS = {"low": 4096, "medium": 8192, "high": 16384, "max": 32768}
DEFAULT_THINKING_MIN_OUTPUT_TOKENS = 8192
DEFAULT_PLAIN_MAX_OUTPUT_TOKENS = 16
DEFAULT_TOP_LOGPROBS = 20


def is_thinking(config: RunConfig, model: ModelSpec) -> bool:
    return bool(model.forced_thinking) or config.effort not in (None, "off")


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
