from __future__ import annotations

import pytest

from blindearth.runner.hashing import (
    DEFAULT_PLAIN_MAX_OUTPUT_TOKENS,
    THINKING_MIN_OUTPUT_TOKENS,
    effective_config,
    is_thinking,
    run_hash,
)
from blindearth.types import ExtractionMode, ExtractionSpec, ModelSpec, RunConfig

M = ModelSpec(id="opus-5-5", provider="anthropic", name="claude-opus-5-5")
X = ExtractionSpec()


def h(config=None, *, model=M, resolved=None, mode=ExtractionMode.SAMPLE, repeat=0, spec="spec1", extraction=X):
    return run_hash(spec, model, resolved, config or RunConfig(), extraction, mode, repeat)


def test_deterministic_and_hex():
    a, b = h(), h()
    assert a == b
    assert len(a) == 64 and int(a, 16) >= 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spec": "spec2"},
        {"repeat": 1},
        {"mode": ExtractionMode.GREEDY},
        {"resolved": "claude-opus-5-5-20261001"},
        {"config": RunConfig(effort="low")},
        {"config": RunConfig(system_prompt="be terse")},
        {"config": RunConfig(seed=7)},
        {"model": ModelSpec(id="opus-5-5", provider="anthropic", name="claude-opus-5-5", quant="int4")},
        {"model": ModelSpec(id="opus-5-5", provider="openrouter", name="claude-opus-5-5")},
        {"extraction": ExtractionSpec(n_samples=8)},
    ],
)
def test_hash_changes_with_output_affecting_inputs(kwargs):
    assert h(**kwargs) != h()


def test_rate_limit_fields_do_not_change_hash():
    assert h(RunConfig(concurrency=4, rpm=100, tpm=10_000)) == h(RunConfig())


def test_unknown_version_equals_model_name():
    assert h(resolved=None) == h(resolved=M.name)


def test_explicit_defaults_hash_like_implicit():
    explicit = RunConfig(temperature=X.temperature, n_samples=X.n_samples, max_output_tokens=DEFAULT_PLAIN_MAX_OUTPUT_TOKENS)
    assert h(explicit) == h(RunConfig())


def test_auto_mode_rejected():
    with pytest.raises(ValueError):
        h(mode=ExtractionMode.AUTO)


def test_effective_config_greedy_and_logprobs():
    g = effective_config(RunConfig(temperature=1.3, n_samples=8), X, ExtractionMode.GREEDY, thinking=False)
    assert g.temperature == 0.0 and g.n_samples == 1 and g.top_logprobs is None
    lp = effective_config(RunConfig(), X, ExtractionMode.LOGPROBS, thinking=False, top_logprobs_cap=5)
    assert lp.n_samples == 1 and lp.top_logprobs == 5


def test_effective_config_thinking_raises_max_tokens_and_is_idempotent():
    cfg = RunConfig(effort="high", max_output_tokens=50)
    eff = effective_config(cfg, X, ExtractionMode.SAMPLE, thinking=True)
    assert eff.max_output_tokens == THINKING_MIN_OUTPUT_TOKENS["high"]
    assert effective_config(eff, X, ExtractionMode.SAMPLE, thinking=True) == eff
    big = effective_config(RunConfig(effort="low", max_output_tokens=100_000), X, ExtractionMode.SAMPLE, thinking=True)
    assert big.max_output_tokens == 100_000


def test_is_thinking():
    assert not is_thinking(RunConfig(), M)
    assert not is_thinking(RunConfig(effort="off"), M)
    assert is_thinking(RunConfig(effort="low"), M)
    forced = ModelSpec(id="x", provider="p", name="x", forced_thinking=True)
    assert is_thinking(RunConfig(), forced)
