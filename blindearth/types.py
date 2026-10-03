"""Shared data types. Every module imports from here; do not redefine these elsewhere.

Changing a field here is a cross-module contract change. Add fields with defaults only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

CoordFormat = Literal["hemisphere", "signed_decimal", "dms"]
Placement = Literal["cell_center", "cell_corner"]
TruthRule = Literal["cell_center", "majority"]
ProviderKind = Literal[
    "anthropic",
    "openai",
    "google",
    "openrouter",
    "openai_compatible",
    "ollama",
    "llamacpp",
    "transformers",
]


class ExtractionMode(str, Enum):
    AUTO = "auto"  # resolved to LOGPROBS or SAMPLE at plan time; never stored on a run
    LOGPROBS = "logprobs"
    SAMPLE = "sample"
    GREEDY = "greedy"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"


# --------------------------------------------------------------------------- eval spec


@dataclass(frozen=True)
class GridSpec:
    step_deg: float = 2.0
    placement: Placement = "cell_center"
    subset_frac: float | None = None  # None = all points; e.g. 0.1 for a seeded 10% smoke test
    seed: int = 0  # point order and subset sampling


@dataclass(frozen=True)
class MaskSpec:
    id: str = "natural-earth-land"  # natural-earth-land | gshhg | modis-mod44w | upload
    path: str | None = None  # required for id == "upload"
    truth_rule: TruthRule = "cell_center"
    resolution_km: float = 1.0
    projection: Literal["equirectangular", "web_mercator"] = "equirectangular"
    invert: bool = False  # uploads: True when white = water
    threshold: float | None = None  # uploads: None = Otsu


@dataclass(frozen=True)
class PromptSpec:
    id: str = "default-land-water"
    # Placeholders: {coord} (formatted per coord_format), {lat}, {lon} (signed decimals).
    template: str = (
        "If this location is over land, say 'Land'. If this location is over water, "
        "say 'Water'. Do not say anything else. {coord}"
    )
    system_prompt: str | None = None


@dataclass(frozen=True)
class EvalSpec:
    mask: MaskSpec = field(default_factory=MaskSpec)
    grid: GridSpec = field(default_factory=GridSpec)
    coord_format: CoordFormat = "hemisphere"
    prompt: PromptSpec = field(default_factory=PromptSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Point:
    idx: int  # stable index in row-major grid order (north->south, west->east)
    lat: float
    lon: float
    weight: float  # cos(lat), for area weighting


@dataclass
class ExtractionSpec:
    mode: ExtractionMode = ExtractionMode.AUTO
    n_samples: int = 4
    temperature: float = 1.0


# --------------------------------------------------------------------------- providers / models


@dataclass
class ProviderSpec:
    id: str
    kind: ProviderKind
    api_key_env: str | None = None
    base_url: str | None = None
    use_batch: bool = False  # use provider batch API when available
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    id: str  # registry id, e.g. "opus-5-5"
    provider: str  # ProviderSpec.id
    name: str  # model name as sent to the API
    quant: str | None = None  # local models
    vendor: str | None = None  # "anthropic", "openai", "google", "meta", ... for scope grouping
    release_date: str | None = None  # ISO date, for the lineage view
    forced_thinking: bool = False  # model always thinks (starred in the original chart)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.id}"


@dataclass
class Capabilities:
    logprobs: bool = False
    top_logprobs_max: int | None = None
    effort_param: bool = False
    supported_efforts: list[str] = field(default_factory=list)
    temperature_fixed_with_thinking: bool = False
    max_concurrency: int | None = None
    supports_batch: bool = False
    probed_at: str | None = None  # ISO timestamp
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- run configuration


@dataclass
class RunConfig:
    """Normalized per-variant configuration (second selection level)."""

    effort: str | None = None  # off|low|medium|high|max|<provider-native>; None = provider default
    temperature: float | None = None  # None = take ExtractionSpec.temperature
    n_samples: int | None = None  # None = take ExtractionSpec.n_samples
    top_logprobs: int | None = None
    max_output_tokens: int | None = None
    system_prompt: str | None = None  # overrides PromptSpec.system_prompt; part of run hash
    concurrency: int | None = None
    rpm: int | None = None
    tpm: int | None = None
    seed: int | None = None
    repeats: int = 1

    # Fields that change model output (and so the run hash). Rate-limit fields do not.
    HASHED_FIELDS = (
        "effort",
        "temperature",
        "n_samples",
        "top_logprobs",
        "max_output_tokens",
        "system_prompt",
        "seed",
    )

    def normalized(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.HASHED_FIELDS if getattr(self, k) is not None}

    def variant_name(self, model_id: str) -> str:
        """`opus-5-5@effort=low,temperature=1.0`, or plain model id when nothing is set."""
        norm = self.normalized()
        norm.pop("system_prompt", None)
        if not norm:
            return model_id
        return model_id + "@" + ",".join(f"{k}={v}" for k, v in sorted(norm.items()))


@dataclass
class CallParams:
    """Resolved per-call parameters handed to an adapter."""

    temperature: float | None
    max_output_tokens: int
    effort: str | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    n: int = 1  # samples wanted; adapters loop if the API has no `n`
    seed: int | None = None
    store_thinking: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens where the provider bills them as output
    thinking_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.thinking_tokens + other.thinking_tokens,
        )


@dataclass
class ClassifyResult:
    """What an adapter returns for one prompt (possibly several samples)."""

    texts: list[str]  # final answer text per sample, thinking removed
    usage: Usage
    latency_s: float
    finish_reasons: list[str] = field(default_factory=list)
    # First-token alternatives of the final answer: token string -> logprob (natural log).
    # Only set in logprobs mode. Tokens are raw (may include leading space).
    first_token_logprobs: dict[str, float] | None = None
    thinking_texts: list[str | None] = field(default_factory=list)
    resolved_model: str | None = None  # model version the API reports back
    native_params: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # not persisted
    error: str | None = None


@dataclass
class Extracted:
    p_land: float | None  # None when no valid answer
    n_valid: int
    n_samples: int
    validity_mass: float | None  # logprobs mode: P(Land)+P(Water) from raw distribution
    answer_text: str  # first sample's raw final text (truncated to 200 chars)
    invalid: bool


@dataclass
class PointResult:
    run_id: str
    idx: int
    lat: float
    lon: float
    truth: int | None  # 1 land, 0 water
    p_land: float | None
    n_valid: int
    n_samples: int
    validity_mass: float | None
    answer_text: str
    finish_reason: str | None
    latency_s: float
    usage: Usage
    thinking_text: str | None = None
    error: str | None = None


@dataclass
class RunRecord:
    id: str
    run_hash: str
    spec_id: str
    model_id: str
    config_id: str
    variant: str  # RunConfig.variant_name(...)
    extraction_mode: ExtractionMode  # resolved, never AUTO
    status: RunStatus = RunStatus.QUEUED
    started_at: str | None = None
    ended_at: str | None = None
    runner_version: str = ""
    seed: int = 0
    repeat_idx: int = 0
    resolved_model_version: str | None = None
    forced_thinking: bool = False
    thinking: bool = False  # effort not in (None, "off") or forced_thinking
    totals: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    n_points_total: int = 0
    n_points_done: int = 0
