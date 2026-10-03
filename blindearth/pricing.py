"""Token prices (USD per 1M tokens) and cost calculation.

Defaults ship in `blindearth/data/pricing.yaml` and are marked as defaults to verify: prices
change, so check them against the vendor's pricing page. A user override file (same format) is
read from the path in env `BLINDEARTH_PRICING`; its entries win over the defaults. A model can
also carry its own price in the registry: `extra.pricing: {input: 4.0, output: 20.0}`.

Lookup keys, first match wins, in each source (override file, then defaults):
`<provider.id>/<model.name>`, `<provider.kind>/<model.name>`, `<model.name>`, `<model.id>`,
then the longest key that is a prefix of the model name (so dated snapshots such as
`gpt-4o-2024-08-06` match `gpt-4o`). Local kinds (ollama, llamacpp, transformers) cost 0 unless
priced explicitly.

Thinking tokens are billed as output: `Usage.output_tokens` already includes them wherever the
provider bills them that way (adapters normalize this), so cost = in*p_in + out*p_out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from blindearth.types import ModelSpec, ProviderSpec, Usage

DEFAULTS_PATH = Path(__file__).parent / "data" / "pricing.yaml"
ENV_OVERRIDE = "BLINDEARTH_PRICING"
LOCAL_KINDS = {"ollama", "llamacpp", "transformers"}


@dataclass(frozen=True)
class Price:
    input_per_m: float  # USD per 1M input tokens
    output_per_m: float  # USD per 1M output tokens (thinking billed here)
    batch_discount: float = 0.5  # multiplier applied for batch API calls
    source: str = "default"  # default | override | registry | local
    verify: bool = True  # True for shipped defaults that should be checked


@dataclass
class PriceTable:
    entries: dict[str, dict[str, Any]]
    batch_discount: float = 0.5
    source: str = "default"

    @classmethod
    def from_yaml(cls, path: str | Path, source: str) -> "PriceTable":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        entries = {str(k): v for k, v in (data.get("models") or {}).items() if isinstance(v, dict)}
        return cls(entries=entries, batch_discount=float(data.get("batch_discount", 0.5)),
                   source=source)

    def lookup(self, keys: list[str], name: str) -> Price | None:
        hit = next((self.entries[k] for k in keys if k in self.entries), None)
        if hit is None:
            cands = [k for k in self.entries if name.startswith(k) or
                     name.split("/")[-1].startswith(k)]
            if cands:
                hit = self.entries[max(cands, key=len)]
        if hit is None:
            return None
        return Price(
            input_per_m=float(hit["input"]),
            output_per_m=float(hit["output"]),
            batch_discount=float(hit.get("batch_discount", self.batch_discount)),
            source=self.source,
            verify=bool(hit.get("verify", self.source == "default")),
        )


@lru_cache(maxsize=8)
def _load(path: str, source: str, mtime: float) -> PriceTable:
    return PriceTable.from_yaml(path, source)


def _tables() -> list[PriceTable]:
    tables = []
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{ENV_OVERRIDE}={override} does not exist")
        tables.append(_load(str(p), "override", p.stat().st_mtime))
    if DEFAULTS_PATH.exists():
        tables.append(_load(str(DEFAULTS_PATH), "default", DEFAULTS_PATH.stat().st_mtime))
    return tables


def price_for(model: ModelSpec, provider: ProviderSpec) -> Price | None:
    """USD per 1M tokens for this model, or None when unknown."""
    reg = model.extra.get("pricing") if model.extra else None
    if isinstance(reg, dict) and "input" in reg and "output" in reg:
        return Price(float(reg["input"]), float(reg["output"]),
                     float(reg.get("batch_discount", 0.5)), source="registry", verify=False)
    keys = [f"{provider.id}/{model.name}", f"{provider.kind}/{model.name}", model.name, model.id]
    for t in _tables():
        p = t.lookup(keys, model.name)
        if p is not None:
            return p
    if provider.kind in LOCAL_KINDS:
        return Price(0.0, 0.0, 1.0, source="local", verify=False)
    return None


def cost_usd(model: ModelSpec, provider: ProviderSpec, usage: Usage, *,
             batch: bool = False) -> float | None:
    p = price_for(model, provider)
    if p is None:
        return None
    c = (usage.input_tokens * p.input_per_m + usage.output_tokens * p.output_per_m) / 1e6
    return c * p.batch_discount if batch else c


__all__ = ["Price", "PriceTable", "price_for", "cost_usd", "DEFAULTS_PATH", "ENV_OVERRIDE"]
