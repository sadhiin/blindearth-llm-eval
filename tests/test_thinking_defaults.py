from __future__ import annotations

import pytest

from blindearth.thinking_defaults import always_thinks, normalize_model_name, thinks_by_default


@pytest.mark.parametrize(
    "raw, norm",
    [
        ("anthropic/claude-opus-4.5", "claude-opus-4-5"),
        ("us.anthropic.claude-opus-5-5-v1:0", "claude-opus-5-5-v1:0"),
        ("openai/gpt-5.1", "gpt-5.1"),
        ("models/gemini-2.5-pro", "gemini-2.5-pro"),
        ("Qwen/Qwen3-8B", "qwen3-8b"),
        ("ft:gpt-4.1-mini:org::abc", "gpt-4.1-mini:org::abc"),
    ],
)
def test_normalize(raw, norm):
    assert normalize_model_name(raw) == norm


# (kind, name, thinks_by_default, always_thinks)
CASES = [
    # Anthropic
    ("anthropic", "claude-opus-5-5", True, True),
    ("anthropic", "claude-fable-5-1", True, True),
    ("anthropic", "claude-mythos-5", True, True),
    ("anthropic", "claude-mythos-preview", True, True),
    ("anthropic", "claude-opus-5", True, False),
    ("anthropic", "claude-sonnet-5", True, False),
    ("anthropic", "claude-sonnet-5-5", True, False),
    ("anthropic", "claude-opus-4-8", False, False),
    ("anthropic", "claude-opus-4-6", False, False),
    ("anthropic", "claude-sonnet-4-6", False, False),
    ("anthropic", "claude-haiku-4-5", False, False),
    ("anthropic", "claude-3-7-sonnet-20250219", False, False),
    # OpenAI
    ("openai", "o3", True, True),
    ("openai", "o4-mini", True, True),
    ("openai", "gpt-5", True, True),
    ("openai", "gpt-5-mini", True, True),
    ("openai", "gpt-5-chat-latest", False, False),
    ("openai", "gpt-5.1", False, False),
    ("openai", "gpt-5.2", False, False),
    ("openai", "gpt-5.4", False, False),
    ("openai", "gpt-5.5", True, False),
    ("openai", "gpt-5.6", True, False),
    ("openai", "gpt-6-astra", True, True),
    ("openai", "gpt-6.1-sol", True, True),
    ("openai", "gpt-6-luna", True, False),
    ("openai", "gpt-4.1-mini", False, False),
    ("openai", "gpt-4o", False, False),
    # Google
    ("google", "gemini-2.5-pro", True, True),
    ("google", "gemini-2.5-flash", True, False),
    ("google", "gemini-2.5-flash-lite", False, False),
    ("google", "gemini-3-pro-preview", True, True),
    ("google", "gemini-3.5-flash-lite", True, True),
    ("google", "gemini-2.0-flash", False, False),
    ("google", "gemini-pro-latest", True, True),
    ("google", "gemini-flash-latest", True, True),
    ("google", "gemini-flash-lite-latest", True, True),
    ("google", "models/gemini-flash-latest", True, True),
    ("openrouter", "google/gemini-pro-latest", True, True),
    ("google", "gemini-1.5-flash-latest", False, False),  # versioned 1.5 alias: no thinking
    # Routed / local
    ("openrouter", "anthropic/claude-opus-5.5", True, True),
    ("openrouter", "anthropic/claude-sonnet-4.6", False, False),
    ("openrouter", "openai/gpt-5.1", False, False),
    ("openrouter", "google/gemini-2.5-flash-lite", False, False),
    ("openrouter", "deepseek/deepseek-r1", True, True),
    ("openrouter", "anthropic/claude-3.7-sonnet:thinking", True, True),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct", False, False),
    ("openai_compatible", "Qwen/Qwen3-8B", True, False),
    ("openai_compatible", "Qwen/Qwen2.5-7B-Instruct", False, False),
    ("ollama", "qwen3:8b", True, False),
    ("ollama", "llama3.2:3b", False, False),
    ("llamacpp", "qwen3-4b-instruct", False, False),
    ("llamacpp", "qwen3-30b-a3b-thinking-2507", True, True),
    ("transformers", "openai/gpt-oss-20b", True, True),
    (None, "gpt-5.5", True, False),
    (None, "something-unknown", False, False),
]


@pytest.mark.parametrize("kind, name, default, always", CASES)
def test_table(kind, name, default, always):
    assert thinks_by_default(kind, name) is default
    assert always_thinks(kind, name) is always


def test_always_implies_default():
    for kind, name, _, _ in CASES:
        if always_thinks(kind, name):
            assert thinks_by_default(kind, name)


def test_first_party_kind_restricts_vendor_table():
    # A Gemini name on the Anthropic API is not a Gemini model; no cross-vendor matches.
    assert not thinks_by_default("anthropic", "gemini-2.5-pro")
    assert not thinks_by_default("google", "o3")
    assert thinks_by_default("openrouter", "google/gemini-2.5-pro")
