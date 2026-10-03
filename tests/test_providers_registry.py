from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from blindearth.providers.registry import (
    RegistryError,
    build_adapter,
    load_registry,
    parse_registry,
    resolve_api_key,
)
from blindearth.types import ProviderSpec

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "providers.yaml"

SPEC_YAML = """
providers:
  - id: anthropic
    kind: anthropic
    api_key_env: ANTHROPIC_API_KEY
  - id: my-vllm
    kind: openai_compatible
    base_url: http://localhost:8000/v1
    api_key_env: VLLM_KEY
    supports_n: false
models:
  - id: opus-5-5
    provider: anthropic
    name: claude-opus-5-5
  - id: qwen-local
    provider: my-vllm
    name: Qwen/Qwen2.5-7B-Instruct
    quant: awq-int4
    release_date: 2024-09-19
    effort_map: {off: {chat_template_kwargs: {enable_thinking: false}}}
"""


def _reg(text=SPEC_YAML, tmp_path=None):
    import yaml

    return parse_registry(yaml.safe_load(text))


def test_load_spec_example(tmp_path):
    f = tmp_path / "providers.yaml"
    f.write_text(SPEC_YAML)
    reg = load_registry(f)
    assert set(reg.providers) == {"anthropic", "my-vllm"}
    q = reg.models["qwen-local"]
    assert q.quant == "awq-int4"
    assert q.release_date == "2024-09-19"
    assert q.extra["effort_map"]["off"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert reg.providers["my-vllm"].extra == {"supports_n": False}
    assert reg.models["opus-5-5"].vendor == "anthropic"


def test_example_file_loads_and_builds_adapters(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
                "VLLM_KEY"):
        monkeypatch.setenv(var, "test-key")
    fake = types.ModuleType("keyring")  # never touch the real OS keychain in tests
    fake.get_password = lambda service, user: None
    monkeypatch.setitem(sys.modules, "keyring", fake)
    reg = load_registry(EXAMPLE)
    kinds = {p.kind for p in reg.providers.values()}
    assert {"ollama", "openai_compatible", "llamacpp", "transformers"} <= kinds
    expected = {
        "anthropic": "AnthropicAdapter", "openai": "OpenAIAdapter", "google": "GoogleAdapter",
        "openrouter": "OpenRouterAdapter", "openai_compatible": "OpenAICompatAdapter",
        "ollama": "OllamaAdapter", "llamacpp": "LlamaCppAdapter",
        "transformers": "TransformersAdapter",
    }
    for m in reg.models.values():
        prov = reg.provider_of(m)
        a = build_adapter(prov, m)
        assert type(a).__name__ == expected[prov.kind]
        assert a.model is m


def test_model_ref_resolution():
    reg = _reg()
    assert reg.model("opus-5-5").name == "claude-opus-5-5"
    assert reg.model("anthropic/opus-5-5").id == "opus-5-5"
    assert reg.model("my-vllm/qwen-local").id == "qwen-local"
    assert reg.model("my-vllm/Qwen/Qwen2.5-7B-Instruct").id == "qwen-local"
    assert reg.model("claude-opus-5-5").id == "opus-5-5"
    with pytest.raises(KeyError):
        reg.model("anthropic/qwen-local")
    with pytest.raises(KeyError):
        reg.model("nope")


def test_validation_errors():
    with pytest.raises(RegistryError):
        _reg("providers: [{id: x, kind: bogus}]")
    with pytest.raises(RegistryError):
        _reg("providers: [{id: x, kind: anthropic, api_key: sk-123}]")
    with pytest.raises(RegistryError):
        _reg("providers: [{id: x, kind: anthropic}]\nmodels: [{id: m, provider: y, name: n}]")
    with pytest.raises(RegistryError):
        _reg("providers: [{id: x, kind: anthropic}, {id: x, kind: openai}]")


def test_resolve_api_key_env_first(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    fake = types.ModuleType("keyring")
    fake.get_password = lambda service, user: "from-keyring"
    monkeypatch.setitem(sys.modules, "keyring", fake)
    p = ProviderSpec(id="x", kind="openai", api_key_env="MY_KEY")
    assert resolve_api_key(p) == "from-env"


def test_resolve_api_key_keyring_fallback(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    seen = {}

    def get_password(service, user):
        seen["args"] = (service, user)
        return "from-keyring"

    fake = types.ModuleType("keyring")
    fake.get_password = get_password
    monkeypatch.setitem(sys.modules, "keyring", fake)
    p = ProviderSpec(id="openai", kind="openai", api_key_env="MY_KEY")
    assert resolve_api_key(p) == "from-keyring"
    assert seen["args"] == ("blindearth", "openai")


def test_resolve_api_key_keyring_failure_is_none(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)

    def boom(service, user):
        raise RuntimeError("no backend")

    fake = types.ModuleType("keyring")
    fake.get_password = boom
    monkeypatch.setitem(sys.modules, "keyring", fake)
    assert resolve_api_key(ProviderSpec(id="x", kind="ollama", api_key_env="MY_KEY")) is None
