"""Provider/model registry loaded from YAML (spec format), API key resolution, adapter factory.

```yaml
providers:
  - id: anthropic
    kind: anthropic
    api_key_env: ANTHROPIC_API_KEY
  - id: my-vllm
    kind: openai_compatible
    base_url: http://localhost:8000/v1
    api_key_env: VLLM_KEY
models:
  - id: opus-5-5
    provider: anthropic
    name: claude-opus-5-5
```

Unknown keys on a provider or model go into its `extra` dict (adapters read options such as
`effort_map`, `timeout_s`, `supports_n` from there). Keys are read from the environment variable
named by `api_key_env`, then from the OS keychain (`keyring`, service `blindearth`, username =
provider id, or `keyring_service`/`keyring_user` in the provider's extra). Keys are never stored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args

import yaml

from blindearth.providers.base import Adapter
from blindearth.types import ModelSpec, ProviderKind, ProviderSpec

KEYRING_SERVICE = "blindearth"
VALID_KINDS = set(get_args(ProviderKind))
_VENDOR_BY_KIND = {"anthropic": "anthropic", "openai": "openai", "google": "google"}
LOCAL_KINDS = {"ollama", "llamacpp", "transformers"}


class RegistryError(ValueError):
    pass


@dataclass
class Registry:
    providers: dict[str, ProviderSpec] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)  # keyed by ModelSpec.id

    def model(self, ref: str) -> ModelSpec:
        """Resolve "provider/model_id" or "model_id" (also accepts "provider/<name as sent>")."""
        if ref in self.models and "/" not in ref:
            return self.models[ref]
        if "/" in ref:
            prov, _, mid = ref.partition("/")
            if prov in self.providers:
                m = self.models.get(mid)
                if m is not None and m.provider == prov:
                    return m
                by_name = [m for m in self.models.values() if m.provider == prov and m.name == mid]
                if len(by_name) == 1:
                    return by_name[0]
            if ref in self.models:  # an id that itself contains "/"
                return self.models[ref]
        else:
            by_name = [m for m in self.models.values() if m.name == ref]
            if len(by_name) == 1:
                return by_name[0]
        known = ", ".join(sorted(m.ref for m in self.models.values()))
        raise KeyError(f"unknown model {ref!r}; registry has: {known}")

    def provider_of(self, model: ModelSpec) -> ProviderSpec:
        try:
            return self.providers[model.provider]
        except KeyError:
            raise KeyError(f"model {model.id!r} references unknown provider {model.provider!r}")

    def models_of(self, provider_id: str) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.provider == provider_id]


def _split(d: dict[str, Any], cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
    names = {f.name for f in fields(cls)}
    known = {k: v for k, v in d.items() if k in names and k != "extra"}
    extra = dict(d.get("extra") or {})
    extra.update({k: v for k, v in d.items() if k not in names})
    return known, extra


def parse_registry(data: dict[str, Any]) -> Registry:
    if not isinstance(data, dict):
        raise RegistryError("registry YAML must be a mapping with providers: and models:")
    reg = Registry()
    for i, p in enumerate(data.get("providers") or []):
        if not isinstance(p, dict) or "id" not in p or "kind" not in p:
            raise RegistryError(f"providers[{i}] needs id and kind")
        if p["kind"] not in VALID_KINDS:
            raise RegistryError(
                f"provider {p['id']!r}: unknown kind {p['kind']!r}; valid: {sorted(VALID_KINDS)}")
        if p["id"] in reg.providers:
            raise RegistryError(f"duplicate provider id {p['id']!r}")
        if "api_key" in p:
            raise RegistryError(
                f"provider {p['id']!r}: put keys in an env var (api_key_env) or the OS keychain, "
                "not in the registry file")
        known, extra = _split(p, ProviderSpec)
        known["id"] = str(known["id"])
        reg.providers[known["id"]] = ProviderSpec(**known, extra=extra)
    for i, m in enumerate(data.get("models") or []):
        if not isinstance(m, dict) or not {"id", "provider", "name"} <= set(m):
            raise RegistryError(f"models[{i}] needs id, provider and name")
        if m["provider"] not in reg.providers:
            raise RegistryError(f"model {m['id']!r}: unknown provider {m['provider']!r}")
        if m["id"] in reg.models:
            raise RegistryError(f"duplicate model id {m['id']!r}")
        known, extra = _split(m, ModelSpec)
        known["id"] = str(known["id"])
        if known.get("release_date") is not None:
            known["release_date"] = str(known["release_date"])  # YAML parses dates
        if not known.get("vendor"):
            known["vendor"] = _VENDOR_BY_KIND.get(reg.providers[m["provider"]].kind)
        reg.models[known["id"]] = ModelSpec(**known, extra=extra)
    return reg


def load_registry(path: str | Path) -> Registry:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return parse_registry(data)


def resolve_api_key(provider: ProviderSpec) -> str | None:
    """Env var first, then the OS keychain. Returns None when neither has a key."""
    if provider.api_key_env:
        v = os.environ.get(provider.api_key_env)
        if v:
            return v
    service = provider.extra.get("keyring_service", KEYRING_SERVICE)
    user = provider.extra.get("keyring_user", provider.id)
    try:
        import keyring  # lazy: optional backend, may be slow to import
    except Exception:  # noqa: BLE001
        return None
    try:
        return keyring.get_password(service, user) or None
    except Exception:  # noqa: BLE001 - no backend / locked keychain
        return None


def adapter_class(kind: str) -> type[Adapter]:
    if kind == "anthropic":
        from blindearth.providers.anthropic import AnthropicAdapter
        return AnthropicAdapter
    if kind == "openai":
        from blindearth.providers.openai_adapter import OpenAIAdapter
        return OpenAIAdapter
    if kind == "google":
        from blindearth.providers.google import GoogleAdapter
        return GoogleAdapter
    if kind == "openrouter":
        from blindearth.providers.openrouter import OpenRouterAdapter
        return OpenRouterAdapter
    if kind == "openai_compatible":
        from blindearth.providers.openai_compat import OpenAICompatAdapter
        return OpenAICompatAdapter
    if kind == "ollama":
        from blindearth.providers.local import OllamaAdapter
        return OllamaAdapter
    if kind == "llamacpp":
        from blindearth.providers.local import LlamaCppAdapter
        return LlamaCppAdapter
    if kind == "transformers":
        from blindearth.providers.local import TransformersAdapter
        return TransformersAdapter
    raise RegistryError(f"unknown provider kind {kind!r}")


def build_adapter(provider: ProviderSpec, model: ModelSpec) -> Adapter:
    if model.provider != provider.id:
        raise RegistryError(f"model {model.id!r} belongs to {model.provider!r}, not {provider.id!r}")
    cls = adapter_class(provider.kind)
    return cls(provider, model, api_key=resolve_api_key(provider))


__all__ = [
    "Registry",
    "RegistryError",
    "load_registry",
    "parse_registry",
    "resolve_api_key",
    "build_adapter",
    "adapter_class",
    "LOCAL_KINDS",
]
