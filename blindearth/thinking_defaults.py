"""Which models reason (think) when the request sets no effort, and which can never stop.

Used by `runner.hashing.is_thinking` to label runs thinking / non-thinking when the run config
leaves `effort` unset (None = "provider default"). Pure lookup on the model name; no network, no
SDK imports (the provider adapters keep their own, richer family tables for request mapping).

Two questions per (provider kind, model name):

- `thinks_by_default`: with no effort/thinking parameter sent, does the model reason?
- `always_thinks`: is there no way to turn reasoning off? (Then effort "off" is still a
  thinking run; the adapter is responsible for refusing "off" or mapping it to the minimum.)
  Every always-thinking model also thinks by default.

Names are normalized before matching: lower-cased, any `vendor/` or `models/` prefix dropped
(OpenRouter, Hugging Face, Gemini `models/...`), cloud prefixes before `claude-` dropped
(`us.anthropic.claude-...`), and dots in Claude names turned into dashes (OpenRouter writes
`claude-opus-4.5`, Anthropic writes `claude-opus-4-5`). Patterns are `re.match`-ed (anchored at
the start) against the normalized name.

A registry entry can override the table with `thinks_by_default: true|false` (kept in
`ModelSpec.extra`); `forced_thinking: true` on the model marks it always-thinking.

Facts below were checked against the vendor docs on 2026-10-03 unless marked UNVERIFIED.

Anthropic, https://platform.claude.com/docs/en/build-with-claude/thinking (per-model table
"No `thinking` field" column, and the "turn it off" paragraphs):
  - "No thinking field" -> adaptive thinking on Opus 5.5, Sonnet 5.5, Fable 5 / 5.1,
    Mythos 5 / 5.1, Opus 5, Sonnet 5, Mythos Preview.
  - "Thinking off" with no field on Opus 4.8, 4.7, 4.6, Sonnet 4.6, Opus 4.5, Sonnet 4.5,
    Haiku 4.5 (and older models, which only have opt-in extended thinking).
  - "Claude Fable 5.1, Claude Mythos 5.1, Claude Fable 5, Claude Mythos 5, Claude Opus 5.5,
    and Claude Mythos Preview reject `thinking: {type: "disabled"}`. Thinking can't be turned
    off on these models."
  - Opus 5 accepts `disabled` at effort high or below; Sonnet 5 accepts `disabled`; Sonnet 5.5
    rejects `disabled` but `between_tools` turns off up-front thinking (with no tools the
    response is text only), so neither is always-thinking for this eval.
  - Default effort is `high` on most models, `medium` on Opus 5.5
    (https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost).
    Adaptive thinking may skip thinking on a trivial request; we still label the run thinking,
    because the model is allowed to and is billed for it when it does.
  - UNVERIFIED extrapolation: later Opus/Sonnet 5.x names (e.g. a hypothetical opus-5-6) are
    assumed to think by default like 5 / 5.5; Haiku 5.x is not listed (not documented).

OpenAI:
  - o-series and gpt-5 / gpt-5-mini / gpt-5-nano: reasoning models without a `none` effort;
    "All models before gpt-5.1 default to medium reasoning effort, and do not support none"
    (openai-python `types/shared/reasoning.py` docstring, older SDK releases; gpt-5 supports
    "minimal, low, medium, and high": https://developers.openai.com/api/docs/models/gpt-5).
    `minimal` still reasons a little, so gpt-5 is always-thinking here.
  - gpt-5.1: "none (default), low, medium, and high"
    (https://developers.openai.com/api/docs/models/gpt-5.1) -> no reasoning by default.
  - gpt-5.2: "none (default), low, medium, high and xhigh"
    (https://developers.openai.com/api/docs/models/gpt-5.2) -> no reasoning by default.
  - gpt-5.4: "none (default), low, medium, high and xhigh"
    (https://developers.openai.com/api/docs/models/gpt-5.4) -> no reasoning by default.
  - gpt-5.3: no model page (404). UNVERIFIED; treated like 5.2 / 5.4 (no reasoning by default).
  - gpt-5.5: "none, low, medium (default), high and xhigh"
    (https://developers.openai.com/api/docs/models/gpt-5.5) -> reasons by default.
  - gpt-5.6: "none, low, medium (default), high, xhigh, and max"
    (https://developers.openai.com/api/docs/models/gpt-5.6) -> reasons by default.
  - gpt-6: "GPT-6.1 Sol, GPT-6 Sol, and GPT-6 Luna also default to medium reasoning effort";
    "GPT-6 Astra does not support none reasoning effort"; "GPT-6.1 Sol does not support none or
    minimal" (https://developers.openai.com/api/docs/guides/reasoning). gpt-6-astra and
    gpt-6.1-sol are always-thinking; whether gpt-6-sol / gpt-6-luna accept `none` is
    UNVERIFIED (treated as default-on, not always-on). Astra's default effort is not stated.
  - `*-chat*` models (e.g. gpt-5-chat-latest) and gpt-4.x / gpt-4o / gpt-3.5 do not reason.

Google Gemini, https://ai.google.dev/gemini-api/docs/thinking (default-thinking table) and
https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking (budgets):
  - gemini-2.5-pro: thinking on by default; budget 128..32768, cannot be turned off.
  - gemini-2.5-flash: on by default; `thinking_budget: 0` turns it off.
  - gemini-2.5-flash-lite: "Off" by default.
  - Gemini 3.x (3-pro-preview, 3-flash-preview, 3.1-pro-preview, 3.5-flash, 3.5-flash-lite,
    3.6/3.7/3.8-flash): all "On" by default; supported levels never include an off level
    (the lowest is `minimal` or `low`), so treated as always-thinking. Whether `minimal`
    yields zero thought tokens is not stated in the docs.
  - gemini-2.0 / 1.5: no thinking.

Open-weights models (OpenRouter, vLLM, Ollama, llama.cpp, transformers):
  - Qwen3 hybrid models (Qwen3-8B etc.): "By default, Qwen3 has thinking capabilities
    enabled" (https://huggingface.co/Qwen/Qwen3-8B). Qwen3-*-Instruct-2507 "supports only
    non-thinking mode" (https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507); Qwen3-*-Thinking-2507
    and QwQ are reasoning-only (UNVERIFIED: not re-fetched). Qwen3-Coder is non-thinking
    (UNVERIFIED).
  - DeepSeek-R1 (and distills): reasoning model with no off switch; the card notes it may
    occasionally emit an empty `<think>` block (https://huggingface.co/deepseek-ai/DeepSeek-R1).
  - gpt-oss: reasoning effort low/medium/high, no off (UNVERIFIED: from the model card, not
    re-fetched).
  - OpenRouter `:thinking` model variants: always thinking (UNVERIFIED naming convention).
"""

from __future__ import annotations

import re
from typing import Iterable

# (pattern, note). Patterns are matched with re.match on the normalized name.
Rule = tuple[str, str]

# ---------------------------------------------------------------------------- Anthropic
ANTHROPIC_ALWAYS: tuple[Rule, ...] = (
    (r"claude-opus-5-5(?:$|[-@:])", "Opus 5.5 rejects thinking disabled"),
    (r"claude-(?:fable|mythos)-", "Fable / Mythos 5, 5.1 and Mythos Preview reject disabled"),
)
ANTHROPIC_DEFAULT: tuple[Rule, ...] = ANTHROPIC_ALWAYS + (
    (r"claude-(?:opus|sonnet)-5(?:$|[-@:])", "Opus/Sonnet 5 and 5.5: adaptive with no field"),
)

# ---------------------------------------------------------------------------- OpenAI
# Non-reasoning OpenAI names; checked before the reasoning rules.
OPENAI_NEVER: tuple[Rule, ...] = (
    (r".*-chat", "chat snapshot of a reasoning family, e.g. gpt-5-chat-latest"),
    (r"(?:gpt-4|gpt-3\.5|chatgpt)", "pre-reasoning chat models"),
)
OPENAI_ALWAYS: tuple[Rule, ...] = (
    (r"o\d", "o-series: reasoning only, default medium"),
    (r"gpt-5(?:$|-)", "gpt-5 / mini / nano: lowest effort is minimal, default medium"),
    (r"gpt-6-astra", "no none effort"),
    (r"gpt-6\.1-sol", "no none or minimal effort"),
)
OPENAI_DEFAULT: tuple[Rule, ...] = OPENAI_ALWAYS + (
    (r"gpt-5\.[56](?:$|[-:])", "gpt-5.5 / 5.6: default medium, none available"),
    (r"gpt-6", "gpt-6 family: default medium"),
)

# ---------------------------------------------------------------------------- Google
GOOGLE_NEVER: tuple[Rule, ...] = (
    (r"gemini-2\.5-flash-lite", "2.5 Flash-Lite: thinking off by default"),
)
GOOGLE_ALWAYS: tuple[Rule, ...] = (
    (r"gemini-2\.5-pro", "2.5 Pro: budget 128..32768, cannot disable"),
    (r"gemini-(?:[3-9]|\d\d)", "Gemini 3.x: no off level"),
    # Unversioned moving aliases (gemini-pro-latest, gemini-flash-latest,
    # gemini-flash-lite-latest): providers/google.py maps them to its Gemini 3.x ("g3")
    # profile, so they are labeled like Gemini 3.x here. UNVERIFIED which model each alias
    # currently points to; override per model with `thinks_by_default` if that changes.
    (r"gemini-[a-z][a-z-]*-latest$", "gemini-*-latest alias: Gemini 3.x profile"),
)
GOOGLE_DEFAULT: tuple[Rule, ...] = GOOGLE_ALWAYS + (
    (r"gemini-2\.5-flash", "2.5 Flash: on by default, budget 0 disables"),
)

# ---------------------------------------------------------------------------- open weights
OPEN_NEVER: tuple[Rule, ...] = (
    (r"qwen3.*(?:instruct|coder)", "Qwen3 Instruct-2507 / Coder: non-thinking only"),
)
OPEN_ALWAYS: tuple[Rule, ...] = (
    (r"deepseek-r1", "DeepSeek-R1 and distills: reasoning model"),
    (r"qwq", "QwQ: reasoning model"),
    (r"qwen3.*thinking", "Qwen3 Thinking-2507: thinking only"),
    (r"gpt-oss", "gpt-oss: effort low/medium/high, no off"),
    (r".*:thinking$", "OpenRouter :thinking variant"),
)
OPEN_DEFAULT: tuple[Rule, ...] = OPEN_ALWAYS + (
    (r"qwen3(?:$|[-:_])", "Qwen3 hybrid: enable_thinking defaults to True"),
)


# family -> (never, always, default)
_TABLES: dict[str, tuple[tuple[Rule, ...], tuple[Rule, ...], tuple[Rule, ...]]] = {
    "anthropic": ((), ANTHROPIC_ALWAYS, ANTHROPIC_DEFAULT),
    "openai": (OPENAI_NEVER, OPENAI_ALWAYS, OPENAI_DEFAULT),
    "google": (GOOGLE_NEVER, GOOGLE_ALWAYS, GOOGLE_DEFAULT),
    "open": (OPEN_NEVER, OPEN_ALWAYS, OPEN_DEFAULT),
}
# First-party kinds only serve their own vendor's models; every other kind (OpenRouter,
# OpenAI-compatible servers, local runtimes, unknown) may serve anything.
_FIRST_PARTY = ("anthropic", "openai", "google")
KNOWN_KINDS = frozenset(
    {"anthropic", "openai", "google", "openrouter", "openai_compatible", "ollama", "llamacpp",
     "transformers"}
)


def normalize_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    i = n.find("claude-")
    if i >= 0:  # "anthropic/claude-opus-4.5", "us.anthropic.claude-opus-4-5-v1:0"
        n = n[i:].replace(".", "-")
    else:
        n = n.rsplit("/", 1)[-1]  # "openai/gpt-5", "models/gemini-2.5-pro", "Qwen/Qwen3-8B"
    if n.startswith("ft:"):  # OpenAI fine-tunes: "ft:gpt-4.1-mini:org::id"
        n = n[3:]
    return n


def _families(provider_kind: str | None) -> Iterable[str]:
    kind = (provider_kind or "").lower()
    if kind in _FIRST_PARTY:
        return (kind,)
    return tuple(_TABLES)


def _family_of(provider_kind: str | None, name: str) -> tuple[str, str] | None:
    """(family, verdict) of the first family with a matching rule; verdict in never/always/default."""
    for fam in _families(provider_kind):
        never, always, default = _TABLES[fam]
        if any(re.match(p, name) for p, _ in never):
            return fam, "never"
        if any(re.match(p, name) for p, _ in always):
            return fam, "always"
        if any(re.match(p, name) for p, _ in default):
            return fam, "default"
    return None


def always_thinks(provider_kind: str | None, model_name: str) -> bool:
    """True if the model cannot be run without reasoning (effort "off" is still thinking)."""
    hit = _family_of(provider_kind, normalize_model_name(model_name))
    return hit is not None and hit[1] == "always"


def thinks_by_default(provider_kind: str | None, model_name: str) -> bool:
    """True if the model reasons when the request carries no effort / thinking parameter."""
    hit = _family_of(provider_kind, normalize_model_name(model_name))
    return hit is not None and hit[1] in ("always", "default")


__all__ = [
    "KNOWN_KINDS",
    "always_thinks",
    "normalize_model_name",
    "thinks_by_default",
]
