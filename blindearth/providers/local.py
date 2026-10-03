"""Local model adapters.

- `OllamaAdapter`: Ollama's OpenAI-compatible endpoint (default http://localhost:11434/v1).
- `LlamaCppAdapter`: llama.cpp `llama-server` OpenAI-compatible endpoint
  (default http://localhost:8080/v1).
- `TransformersAdapter`: Hugging Face transformers in-process. One forward pass gives exact
  first-token log-probabilities from a softmax over the full vocabulary; generation (greedy or
  sampled) gives the answer texts. torch/transformers are imported lazily.

Effort mapping (normalized -> native):

| normalized | ollama                         | llama.cpp                                         | transformers                          |
|------------|--------------------------------|---------------------------------------------------|---------------------------------------|
| off        | `reasoning_effort: "none"`     | `chat_template_kwargs: {enable_thinking: false}`  | chat template `enable_thinking=False` |
| low        | `reasoning_effort: "low"`      | refused                                           | refused                               |
| medium     | `reasoning_effort: "medium"`   | refused                                           | refused                               |
| high       | `reasoning_effort: "high"`     | refused                                           | refused                               |
| max        | refused                        | refused                                           | refused                               |

llama.cpp has no per-request reasoning budget and transformers has no effort concept beyond the
chat template switch, so those levels are refused unless the registry sets
`extra.effort_map` (for transformers the values are extra chat-template kwargs).
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from blindearth.providers.base import Adapter, ProviderError, UnsupportedConfigError
from blindearth.providers.openai_compat import (
    NORMALIZED_EFFORTS,
    OpenAICompatAdapter,
    mode_temperature,
    raise_for_thinking,
    resolve_effort_map,
    split_think,
)
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    ExtractionMode,
    RunConfig,
    Usage,
)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLAMACPP_BASE_URL = "http://localhost:8080/v1"
LAND_WATER_WORDS = ("land", "water")
TOP_K_DEFAULT = 20


class OllamaAdapter(OpenAICompatAdapter):
    kind = "ollama"
    default_base_url = OLLAMA_BASE_URL
    supports_n_default = False
    default_logprobs = False  # only recent Ollama versions return logprobs; the probe decides
    default_top_logprobs_max = None
    EFFORT_MAP: dict[str, Any] = {
        "off": {"reasoning_effort": "none"},
        "low": {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "medium"},
        "high": {"reasoning_effort": "high"},
        "max": None,
    }

    def default_capabilities(self) -> Capabilities:
        caps = super().default_capabilities()
        caps.max_concurrency = 1
        return caps


class LlamaCppAdapter(OpenAICompatAdapter):
    kind = "llamacpp"
    default_base_url = LLAMACPP_BASE_URL
    supports_n_default = False
    default_logprobs = True
    default_top_logprobs_max = 20
    EFFORT_MAP: dict[str, Any] = {
        "off": {"chat_template_kwargs": {"enable_thinking": False}},
        "low": None,
        "medium": None,
        "high": None,
        "max": None,
    }

    def default_capabilities(self) -> Capabilities:
        caps = super().default_capabilities()
        caps.max_concurrency = 1
        return caps


# --------------------------------------------------------------------------- transformers


def normalize_piece(piece: str) -> str:
    """Raw vocab piece -> surface string (byte-level BPE and SentencePiece space markers)."""
    return piece.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")


def is_land_water_piece(surface: str) -> bool:
    """Case-insensitive prefix match against Land/Water, with or without one leading space.

    True when the piece is a prefix of the word ("La", " wat") or the word is a prefix of the
    piece ("Land", " Water."). The extractor applies its own rule on the returned dict.
    """
    core = surface[1:] if surface.startswith(" ") else surface
    if not core or core[0].isspace():
        return False
    c = core.lower()
    return any(w.startswith(c) or c.startswith(w) for w in LAND_WATER_WORDS)


def land_water_token_ids(pieces: list[str | None]) -> list[int]:
    return [i for i, p in enumerate(pieces) if p is not None
            and is_land_water_piece(normalize_piece(p))]


def logprob_dict(ids: list[int], logps: list[float], pieces: list[str | None]) -> dict[str, float]:
    """Combine token ids into {surface: logprob}; duplicates (same surface) are log-summed."""
    out: dict[str, float] = {}
    seen: set[int] = set()
    for i, lp in zip(ids, logps):
        if i in seen or i >= len(pieces) or pieces[i] is None:
            continue
        seen.add(i)
        s = normalize_piece(pieces[i])
        if s in out:
            a, b = out[s], float(lp)
            m = max(a, b)
            out[s] = m + math.log(math.exp(a - m) + math.exp(b - m)) if m > -math.inf else m
        else:
            out[s] = float(lp)
    return out


class TransformersAdapter(Adapter):
    kind = "transformers"

    EFFORT_MAP: dict[str, Any] = {
        "off": {"enable_thinking": False},
        "low": None,
        "medium": None,
        "high": None,
        "max": None,
    }

    def __init__(self, provider, model, api_key: str | None = None):
        super().__init__(provider, model, api_key)
        self._tok = None
        self._model = None
        self._device = None
        self._pieces: list[str | None] | None = None
        self._lw_ids: list[int] | None = None
        self._lock = asyncio.Lock()
        self.top_k = int(model.extra.get("top_k_logprobs", TOP_K_DEFAULT))

    # ---- mapping

    def _template_kwargs(self, effort: str | None) -> dict[str, Any]:
        if effort is None:
            return {}
        emap = resolve_effort_map(self, self.EFFORT_MAP)
        v = emap.get(effort, "__missing__")
        if v == "__missing__" and effort not in NORMALIZED_EFFORTS:
            raise UnsupportedConfigError(f"unknown effort {effort!r} for transformers")
        if v is None or v == "__missing__":
            raise UnsupportedConfigError(
                f"transformers: effort={effort!r} has no mapping; set extra.effort_map")
        return dict(v)

    def map_config(self, config: RunConfig, mode: ExtractionMode) -> dict[str, Any]:
        tk = self._template_kwargs(config.effort)
        thinking = config.effort not in (None, "off") or self.model.forced_thinking
        if mode == ExtractionMode.LOGPROBS and thinking:
            raise UnsupportedConfigError(
                "first-token logprobs are meaningless when the model thinks first; "
                "use effort=off or sample mode")
        if config.top_logprobs is not None and mode != ExtractionMode.LOGPROBS:
            raise UnsupportedConfigError("top_logprobs is only valid in logprobs mode")
        native: dict[str, Any] = {"model": self.model.name, "quant": self.model.quant,
                                  "chat_template_kwargs": tk}
        t = mode_temperature(config, mode)
        if t is not None:
            native["do_sample"] = t > 0
            native["temperature"] = t
        if mode == ExtractionMode.LOGPROBS:
            native["first_token_logprobs"] = "exact full-vocab softmax"
            native["top_k"] = config.top_logprobs or self.top_k
        if config.max_output_tokens is not None:
            native["max_new_tokens"] = config.max_output_tokens
        if mode == ExtractionMode.SAMPLE and config.n_samples:
            native["num_return_sequences"] = config.n_samples
        if config.seed is not None:
            native["seed"] = config.seed
        return native

    def default_capabilities(self) -> Capabilities:
        return Capabilities(
            logprobs=True, top_logprobs_max=None, effort_param=False,
            supported_efforts=["off"], max_concurrency=1, supports_batch=False,
            notes=["exact full-vocabulary softmax; effort only via chat template"],
        )

    # ---- model loading (blocking; runs in a worker thread)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ProviderError(
                "transformers adapter needs `pip install blindearth[local]` (torch, transformers)",
                retryable=False) from e
        extra = self.model.extra
        tok_kw = dict(extra.get("tokenizer_kwargs") or {})
        load_kw = dict(extra.get("load_kwargs") or {})
        load_kw.setdefault("torch_dtype", "auto")
        if extra.get("revision"):
            tok_kw.setdefault("revision", extra["revision"])
            load_kw.setdefault("revision", extra["revision"])
        self._tok = AutoTokenizer.from_pretrained(self.model.name, **tok_kw)
        if extra.get("device_map"):
            load_kw["device_map"] = extra["device_map"]
        model = AutoModelForCausalLM.from_pretrained(self.model.name, **load_kw)
        if "device_map" not in load_kw:
            device = extra.get("device")
            if not device:
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            model = model.to(device)
        model.eval()
        self._device = next(model.parameters()).device
        self._model = model
        vocab = len(self._tok)
        self._pieces = self._tok.convert_ids_to_tokens(list(range(vocab)))
        self._lw_ids = land_water_token_ids(self._pieces)

    def _render(self, prompt: str, system_prompt: str | None, template_kwargs: dict) -> str:
        tok = self._tok
        if getattr(tok, "chat_template", None):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **template_kwargs)
        return (system_prompt + "\n\n" if system_prompt else "") + prompt

    def _run_sync(self, prompt: str, system_prompt: str | None, params: CallParams,
                  template_kwargs: dict) -> dict[str, Any]:
        import torch

        self._load()
        tok, model = self._tok, self._model
        text = self._render(prompt, system_prompt, template_kwargs)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self._device)
        attn = enc.get("attention_mask")
        attn = attn.to(self._device) if attn is not None else None
        n_in = int(input_ids.shape[-1])
        first_lp = None
        if params.seed is not None:
            torch.manual_seed(params.seed)
        with torch.no_grad():
            if params.logprobs:
                logits = model(input_ids=input_ids, attention_mask=attn).logits[0, -1].float()
                logp = torch.log_softmax(logits, dim=-1)
                vocab_n = logp.shape[-1]
                k = min(params.top_logprobs or self.top_k, vocab_n)
                top = torch.topk(logp, k)
                ids = top.indices.tolist()
                lw = [i for i in (self._lw_ids or []) if i < vocab_n]
                ids_all = ids + lw
                lps = logp[torch.tensor(ids_all, device=logp.device)].tolist() if ids_all else []
                first_lp = logprob_dict(ids_all, lps, self._pieces or [])
            n = max(1, params.n)
            max_new = raise_for_thinking(params.max_output_tokens, params.effort
                                         if params.effort not in (None, "off") else None,
                                         forced=self.model.forced_thinking)
            temp = params.temperature
            do_sample = bool(temp and temp > 0)
            gen_kw: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": do_sample,
                                      "num_return_sequences": n}
            if do_sample:
                gen_kw["temperature"] = float(temp)
                gen_kw["top_p"] = 1.0
                gen_kw["top_k"] = 0
            elif n > 1:
                gen_kw["num_return_sequences"] = 1  # greedy: identical samples
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            out = model.generate(input_ids=input_ids, attention_mask=attn, pad_token_id=pad,
                                 **gen_kw)
        seqs = out[:, n_in:]
        texts, thinking, finish = [], [], []
        n_out = 0
        n_think = 0
        eos = tok.eos_token_id
        eos_set = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        for row in seqs.tolist():
            # trim padding after EOS
            cut = len(row)
            for j, t in enumerate(row):
                if t in eos_set:
                    cut = j + 1
                    break
            row = row[:cut]
            n_out += len(row)
            raw = tok.decode(row, skip_special_tokens=True)
            ans, th = split_think(raw)
            if th:
                n_think += len(tok.encode(th, add_special_tokens=False))
            texts.append(ans)
            thinking.append(th if params.store_thinking else None)
            finish.append("stop" if (row and row[-1] in eos_set) else
                          ("length" if len(row) >= max_new else "stop"))
        if not do_sample and n > 1:  # replicate the greedy answer for each requested sample
            texts, thinking, finish = texts * n, thinking * n, finish * n
        return {"texts": texts, "thinking": thinking, "finish": finish, "first_lp": first_lp,
                "n_in": n_in, "n_out": n_out, "n_think": n_think, "gen_kw": gen_kw}

    async def classify(self, prompt: str, system_prompt: str | None,
                       params: CallParams) -> ClassifyResult:
        try:
            tk = self._template_kwargs(params.effort)
        except UnsupportedConfigError as e:
            raise ProviderError(str(e), retryable=False) from e
        t0 = time.perf_counter()
        async with self._lock:
            try:
                r = await asyncio.to_thread(self._run_sync, prompt, system_prompt, params, tk)
            except ProviderError:
                raise
            except Exception as e:  # noqa: BLE001  (e.g. CUDA OOM): not retryable blindly
                raise ProviderError(f"{type(e).__name__}: {e}", retryable=False) from e
        return ClassifyResult(
            texts=r["texts"],
            usage=Usage(input_tokens=r["n_in"], output_tokens=r["n_out"],
                        thinking_tokens=r["n_think"]),
            latency_s=time.perf_counter() - t0,
            finish_reasons=r["finish"],
            first_token_logprobs=r["first_lp"] if params.logprobs else None,
            thinking_texts=r["thinking"],
            resolved_model=self.model.name + (f"@{self.model.extra['revision']}"
                                              if self.model.extra.get("revision") else ""),
            native_params={"chat_template_kwargs": tk, "quant": self.model.quant, **r["gen_kw"]},
        )

    async def aclose(self) -> None:
        self._model = None
        self._tok = None


__all__ = [
    "OllamaAdapter",
    "LlamaCppAdapter",
    "TransformersAdapter",
    "normalize_piece",
    "is_land_water_piece",
    "land_water_token_ids",
    "logprob_dict",
    "OLLAMA_BASE_URL",
    "LLAMACPP_BASE_URL",
]
