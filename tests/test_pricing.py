from __future__ import annotations

import pytest

from blindearth.pricing import ENV_OVERRIDE, cost_usd, price_for
from blindearth.types import ModelSpec, ProviderSpec, Usage


def _m(name, provider="p", **extra):
    return ModelSpec(id=name.replace("/", "-"), provider=provider, name=name, extra=extra)


ANTHROPIC = ProviderSpec(id="anthropic", kind="anthropic")
OPENAI = ProviderSpec(id="openai", kind="openai")
VLLM = ProviderSpec(id="my-vllm", kind="openai_compatible", base_url="http://x/v1")
OLLAMA = ProviderSpec(id="ollama", kind="ollama")


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)


def test_exact_default_price():
    p = price_for(_m("claude-opus-5-5", "anthropic"), ANTHROPIC)
    assert p is not None
    assert (p.input_per_m, p.output_per_m) == (4.0, 20.0)
    assert p.verify is True
    assert p.batch_discount == 0.5


def test_longest_prefix_match_for_dated_snapshot():
    p = price_for(_m("gpt-4o-mini-2024-07-18", "openai"), OPENAI)
    assert (p.input_per_m, p.output_per_m) == (0.15, 0.6)
    p = price_for(_m("gpt-4o-2024-08-06", "openai"), OPENAI)
    assert (p.input_per_m, p.output_per_m) == (2.5, 10.0)


def test_openrouter_style_name_matches_vendor_entry():
    prov = ProviderSpec(id="openrouter", kind="openrouter")
    p = price_for(_m("anthropic/claude-sonnet-5-5", "openrouter"), prov)
    assert p is not None and p.input_per_m == 2.0


def test_cost_includes_output_and_batch_discount():
    m = _m("claude-opus-5-5", "anthropic")
    u = Usage(input_tokens=1_000_000, output_tokens=500_000, thinking_tokens=400_000)
    assert cost_usd(m, ANTHROPIC, u) == pytest.approx(4.0 + 10.0)
    assert cost_usd(m, ANTHROPIC, u, batch=True) == pytest.approx(7.0)


def test_unknown_compat_model_is_none_and_local_is_free():
    assert price_for(_m("Qwen/Qwen2.5-7B-Instruct", "my-vllm"), VLLM) is None
    assert cost_usd(_m("Qwen/Qwen2.5-7B-Instruct", "my-vllm"), VLLM, Usage(10, 10)) is None
    p = price_for(_m("llama3.2:3b", "ollama"), OLLAMA)
    assert p.input_per_m == 0 and p.output_per_m == 0
    assert cost_usd(_m("llama3.2:3b", "ollama"), OLLAMA, Usage(100, 100)) == 0.0


def test_registry_price_wins():
    m = _m("Qwen/Qwen2.5-7B-Instruct", "my-vllm", pricing={"input": 0.2, "output": 0.4})
    p = price_for(m, VLLM)
    assert p.source == "registry" and p.input_per_m == 0.2


def test_user_override_file(tmp_path, monkeypatch):
    f = tmp_path / "prices.yaml"
    f.write_text(
        "batch_discount: 0.4\n"
        "models:\n"
        "  claude-opus-5-5: {input: 1.0, output: 2.0}\n"
        "  my-vllm/Qwen/Qwen2.5-7B-Instruct: {input: 0.05, output: 0.05}\n"
    )
    monkeypatch.setenv(ENV_OVERRIDE, str(f))
    p = price_for(_m("claude-opus-5-5", "anthropic"), ANTHROPIC)
    assert (p.input_per_m, p.output_per_m, p.source) == (1.0, 2.0, "override")
    assert p.batch_discount == 0.4
    q = price_for(_m("Qwen/Qwen2.5-7B-Instruct", "my-vllm"), VLLM)
    assert q.input_per_m == 0.05
    # entries missing from the override fall back to defaults
    r = price_for(_m("claude-haiku-4-5", "anthropic"), ANTHROPIC)
    assert r.source == "default" and r.input_per_m == 1.0


def test_missing_override_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError):
        price_for(_m("claude-opus-5-5", "anthropic"), ANTHROPIC)
