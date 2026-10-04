# blindearth module contracts

Product spec: `Blind Earth Eval Runner Product Spec.md`. Shared types: `blindearth/types.py`.
Adapter base: `blindearth/providers/base.py`. Treat both as stable: add a new field only with a
default value, so existing code keeps working.

Every public function below exists with this signature. Modules import each other only through
these names. The letters A to E group the modules by area.

## Conventions

- Python 3.11+, `from __future__ import annotations`, type hints, dataclasses from `types.py`.
- Grid order: `Point.idx` is row-major, north to south, west to east. Cell-center lats at 2° are
  89, 87, …, -89; lons -179, …, 179. Corner placement: lats 90…-88, lons -180…178.
- Masks are `numpy.ndarray[bool]` of shape (H, W), equirectangular, row 0 = 90°N, col 0 = 180°W,
  True = land.
- Points tables are `pandas.DataFrame` with columns
  `idx, lat, lon, weight, truth, p_land, n_valid, n_samples, validity_mass, answer_text,
  finish_reason, latency_s, input_tokens, output_tokens, thinking_tokens, error`.
- Missing p_land (invalid/error) counts as wrong in the headline accuracy.
- All timestamps are ISO-8601 UTC strings. Ids are `uuid4().hex` unless stated.
- No network at import time. Heavy optional deps (torch, transformers, gradio, keyring) are
  imported lazily inside functions.

## A. Providers (`blindearth/providers/*`, `blindearth/ratelimit.py`, `blindearth/pricing.py`)

```python
# providers/registry.py
@dataclass
class Registry:
    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]          # keyed by ModelSpec.id
    def model(self, ref: str) -> ModelSpec          # "provider/model_id" or "model_id"
    def provider_of(self, model: ModelSpec) -> ProviderSpec
def load_registry(path: str | Path) -> Registry        # YAML: providers:[], models:[] (spec format)
def resolve_api_key(provider: ProviderSpec) -> str | None   # env var, then OS keychain
def build_adapter(provider: ProviderSpec, model: ModelSpec) -> Adapter

# providers/probe.py
async def probe(adapter: Adapter, n_calls: int = 20) -> Capabilities

# providers/{anthropic,openai_adapter,google,openrouter,openai_compat,local}.py
# one Adapter subclass each; local.py has OllamaAdapter, LlamaCppAdapter (both via
# OpenAI-compatible endpoints) and TransformersAdapter (in-process, exact first-token logits).
# map_config raises UnsupportedConfigError for any setting it cannot honour; it never drops one.
# Batch (submit_batch/poll_batch): anthropic, openai, google (Gemini Developer API only).

# providers/openai_compat.py
def resolve_effort_map(adapter: Adapter, default: dict[str, Any]) -> dict[str, Any]
    # default, updated by provider extra.effort_map, then by model extra.effort_map

# thinking_defaults.py   (which models reason when no effort is set; sourced from vendor docs)
KNOWN_KINDS: frozenset[str]
def normalize_model_name(name: str) -> str          # strips vendor prefixes, lowercases
def thinks_by_default(provider_kind: str | None, model_name: str) -> bool
def always_thinks(provider_kind: str | None, model_name: str) -> bool   # reasoning cannot be turned off

# ratelimit.py
class ProviderLimiter:
    def __init__(self, concurrency: int, rpm: int | None, tpm: int | None)
    def slot(self, est_tokens: int) -> AsyncContextManager[None]
    def on_rate_limited(self, retry_after_s: float | None) -> None   # halve concurrency
    def on_success(self) -> None                                     # slow additive recovery
class CircuitBreaker:
    def __init__(self, threshold: int = 20, window_s: float = 60.0)
    def record(self, ok: bool) -> None
    @property
    def open(self) -> bool
async def call_with_retries(fn: Callable[[], Awaitable[T]], *, max_attempts: int = 5,
                            limiter: ProviderLimiter | None = None) -> T

# pricing.py   (defaults in blindearth/data/pricing.yaml, user override file supported)
def price_for(model: ModelSpec, provider: ProviderSpec) -> Price | None   # USD per 1M tokens
def cost_usd(model: ModelSpec, provider: ProviderSpec, usage: Usage, *, batch: bool = False) -> float | None
```

## B. Eval definition and extraction (`blindearth/evalspec/*`, `blindearth/extract.py`)

```python
# evalspec/grid.py
def make_grid(grid: GridSpec) -> list[Point]                    # applies subset_frac with grid.seed
def stratified_order(points: list[Point], seed: int) -> list[Point]  # coarse-to-fine, spread out
def cell_bounds(point: Point, step_deg: float, placement: Placement) -> tuple[float,float,float,float]

# evalspec/coords.py
def format_coord(lat: float, lon: float, fmt: CoordFormat) -> str   # "12° S, 45° W" etc.

# evalspec/prompts.py
def render_prompt(prompt: PromptSpec, lat: float, lon: float, fmt: CoordFormat) -> str

# evalspec/masks.py
@dataclass
class Mask:
    data: np.ndarray        # bool (H, W)
    source: str             # human label printed in reports
    hash: str               # sha256 of packed bits + shape
def load_mask(spec: MaskSpec, cache_dir: Path | None = None) -> Mask
def prepare_upload(path: str | Path, *, projection: str, invert: bool,
                   threshold: float | None) -> tuple[Mask, np.ndarray, float]
    # -> (mask, preview_rgb_uint8, threshold_used)

# evalspec/truth.py
def cell_truth(mask: Mask, points: list[Point], grid: GridSpec, rule: TruthRule) -> np.ndarray  # int8

# evalspec/load.py
@dataclass
class MatrixEntry:
    model_ref: str
    configs: list[RunConfig]
@dataclass
class EvalFile:
    eval: EvalSpec
    extraction: ExtractionSpec
    matrix: list[MatrixEntry]
    budget_usd: float | None
    name: str | None
def load_eval_file(path: str | Path) -> EvalFile
def spec_hash(spec: EvalSpec, mask_hash: str) -> str          # stable sha256 of canonical JSON

# extract.py
def extract(result: ClassifyResult, mode: ExtractionMode) -> Extracted
def parse_answer(text: str) -> Literal["land", "water"] | None
def p_land_from_logprobs(first_token_logprobs: dict[str, float]) -> tuple[float | None, float]
    # -> (p_land, validity_mass)
```

## C. Store, runner, CLI (`blindearth/store/*`, `blindearth/runner/*`, `blindearth/cli.py`)

```python
# store/db.py
class Store:
    def __init__(self, path: str | Path)                    # creates schema if missing
    def upsert_eval_spec(self, spec: EvalSpec, mask_hash: str, mask_source: str) -> str  # id = spec_hash
    def get_eval_spec(self, spec_id: str) -> tuple[EvalSpec, str, str]   # spec, mask_hash, mask_source
    def upsert_model(self, model: ModelSpec, provider: ProviderSpec) -> str
    def upsert_config(self, config: RunConfig, native: dict) -> str     # id = hash of normalized+native
    def create_run(self, run: RunRecord) -> None
    def find_complete_run(self, run_hash: str) -> RunRecord | None
    def get_run(self, run_id: str) -> RunRecord
    def list_runs(self, *, status: RunStatus | None = None, model_id: str | None = None,
                  spec_id: str | None = None) -> list[RunRecord]
    def update_run(self, run_id: str, **fields) -> None
    def write_points(self, points: list[PointResult]) -> None             # upsert by (run_id, idx)
    def done_indices(self, run_id: str) -> set[int]                       # points without error
    def failed_indices(self, run_id: str) -> set[int]
    def load_points(self, run_id: str) -> pd.DataFrame
    def save_metrics(self, run_id: str, mask_hash: str, threshold: float, metrics: dict) -> None
    def load_metrics(self, run_id: str, mask_hash: str | None = None,
                     threshold: float | None = None) -> dict | None
    def create_comparison(self, name: str, run_ids: list[str], filters: dict | None = None,
                          ordering: list[str] | None = None) -> str
    def get_comparison(self, comparison_id_or_name: str) -> dict   # {id,name,run_ids,filters,ordering,created_at}
    def list_comparisons(self) -> list[dict]
    def save_capabilities(self, provider_id: str, model_name: str, caps: Capabilities) -> None
    def load_capabilities(self, provider_id: str, model_name: str) -> Capabilities | None

# store/export.py
def export_points(store: Store, run_id: str, path: Path, fmt: Literal["parquet","csv"]) -> Path
def export_metrics(store: Store, run_id: str, path: Path) -> Path

# runner/hashing.py
def run_hash(spec_id: str, model: ModelSpec, resolved_version: str | None, config: RunConfig,
             extraction: ExtractionSpec, mode: ExtractionMode, repeat_idx: int) -> str
def effective_config(config: RunConfig, extraction: ExtractionSpec, mode: ExtractionMode, *,
                     thinking: bool, top_logprobs_cap: int | None = None) -> RunConfig
def is_thinking(config: RunConfig, model: ModelSpec, provider_kind: str | None = None) -> bool
def model_thinks_by_default(model: ModelSpec, provider_kind: str | None = None) -> bool
    # model extra.thinks_by_default wins over thinking_defaults
def model_always_thinks(model: ModelSpec, provider_kind: str | None = None) -> bool
    # model.forced_thinking, else thinking_defaults.always_thinks (unless thinks_by_default: false)
def infer_provider_kind(model: ModelSpec) -> str | None

# runner/planner.py
@dataclass
class PlanCell:
    model: ModelSpec; provider: ProviderSpec; config: RunConfig; variant: str
    mode: ExtractionMode; repeat_idx: int; run_hash: str
    n_points: int; n_calls: int; est_usage: Usage; est_cost_usd: float | None
    cached_run_id: str | None; refused: str | None; native_params: dict
    # with defaults: thinking: bool; forced_thinking: bool (= model_always_thinks; starred);
    # resolved_version: str | None; resume_run_id: str | None; use_batch: bool
@dataclass
class Plan:
    eval_file: EvalFile; spec_id: str; mask: Mask; points: list[Point]; truth: np.ndarray
    cells: list[PlanCell]; total_cost_usd: float | None; budget_usd: float | None
async def build_plan(eval_file: EvalFile, registry: Registry, store: Store, *,
                     pilot_points: int = 50, run_pilot: bool = True) -> Plan
def check_comparison(store: Store, run_ids: list[str]) -> dict[str, Any]
    # raises ValueError on mixed eval specs; else
    # {spec_id, mixed_extraction_modes, rank_on, forced_thinking_runs, warnings}

# runner/executor.py
async def execute(plan: Plan, store: Store, registry: Registry, *,
                  on_progress: Callable[[str, list[PointResult]], None] | None = None,
                  budget_usd: float | None = None, stop_event: asyncio.Event | None = None
                  ) -> list[str]                             # run ids
async def resume(run_id: str, store: Store, registry: Registry, **kw) -> None

# cli.py  — Typer app named `app`
# blindearth run SPEC [--registry providers.yaml] [--db blindearth.db] [--estimate] [--budget USD]
# blindearth probe MODEL_REF ; blindearth runs ; blindearth resume RUN_ID
# blindearth compare RUN... --name NAME ; blindearth score RUN... [--mask ...] [--threshold 0.5]
# blindearth report COMPARISON --html out.html --png grid.png
# blindearth export RUN --format parquet|csv --out PATH ; blindearth serve
```

## D. Scoring (`blindearth/scoring/*`)

```python
# scoring/cell_metrics.py
def cell_metrics(df: pd.DataFrame, threshold: float = 0.5, probabilistic: bool = True, *,
                 step_deg: float | None = None, placement: Placement | None = None) -> dict
    # keys: acc_area, acc_unweighted, acc_valid_only, baseline_area, skill, precision, recall,
    # f1, iou, brier, log_loss, ece, invalid_rate, coastline_error_deg, n_points,
    # input_tokens, output_tokens, thinking_tokens, latency_p50, latency_p95

# scoring/regions.py
# Natural Earth 110m region/marine polygons; lat/lon boxes as fallback (env BLINDEARTH_REGIONS)
class RegionAccuracy(dict):     # dict[str, float] with .method: "natural-earth-110m" | "lat-lon-boxes" | None
def region_labels(lat: np.ndarray, lon: np.ndarray, truth: np.ndarray) -> np.ndarray  # str labels
def region_labels_with_method(lat: np.ndarray, lon: np.ndarray, truth: np.ndarray, *,
                              cache_dir: str | Path | None = None) -> tuple[np.ndarray, str]
def region_accuracy(df: pd.DataFrame, threshold: float = 0.5) -> RegionAccuracy

# scoring/bootstrap.py
def block_bootstrap_ci(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], *,
                       block_deg: float = 10.0, n_boot: int = 1000, seed: int = 0,
                       alpha: float = 0.05) -> tuple[float, float, float]     # (est, lo, hi)
def paired_difference(df_a: pd.DataFrame, df_b: pd.DataFrame, *, threshold: float = 0.5,
                      block_deg: float = 10.0, n_boot: int = 1000, seed: int = 0) -> dict
    # {diff, lo, hi, significant, n_paired}
def repeat_spread(dfs: list[pd.DataFrame], threshold: float = 0.5) -> dict

# scoring/render.py
def points_to_grid(df: pd.DataFrame, step_deg: float, placement: Placement,
                   value: Literal["p_land","binary","error"], threshold: float = 0.5) -> np.ndarray
def upsample_to(grid: np.ndarray, shape: tuple[int,int], *, binary: bool) -> np.ndarray
def error_overlay(pred_binary: np.ndarray, mask: np.ndarray) -> np.ndarray   # RGB uint8: red=false land, blue=false water

# scoring/image_metrics.py
def image_metrics(df: pd.DataFrame, mask: Mask, step_deg: float, placement: Placement,
                  threshold: float = 0.5, tolerances_deg: tuple[float, ...] = (2.0, 4.0),
                  work_shape: tuple[int,int] = (1800, 3600), *, truth_rule: TruthRule = "cell_center",
                  probabilistic: bool = True, truth_grid: np.ndarray | None = None,
                  min_landmass_km2: float = 5.0e5) -> dict
    # pixel_iou, dice, ssim, boundary_f@2, boundary_f@4, contour_mean_deg/km, contour_p95_deg/km,
    # n_components_pred, n_components_true, spurious_specks, per_landmass_iou{}, prob_rmse,
    # ceiling{...same keys for the perfect-at-this-grid map}, share_of_ceiling{...}

# scoring/score.py
def score_run(store: Store, run_id: str, mask: Mask | None = None, threshold: float = 0.5,
              with_images: bool = True) -> dict      # computes + store.save_metrics
    # keys: cell_metrics keys, regions, regions_method, acc_area_ci, image, plus run context
    # (run_id, variant, extraction_mode, mask_hash, mask_source, partial, thinking, forced_thinking, ...)
```

## E. Reports and UI (`blindearth/report/*`, `blindearth/ui/*`)

```python
# report/data.py
@dataclass
class RunView:  # everything a report needs for one run, loaded from the store
    run: RunRecord; model: ModelSpec; df: pd.DataFrame; metrics: dict; regions: dict
    acc_ci: tuple[float, float, float]
def load_comparison(store: Store, comparison: str, threshold: float = 0.5) -> tuple[dict, list[RunView], Mask]

# report/mapgrid.py
def render_map_grid(views: list[RunView], *, mode: Literal["binary","probability","error"] = "binary",
                    step_deg: float, placement: Placement, mask: Mask | None = None,
                    coastline: bool = False, title: str | None = None, ncols: int = 4) -> bytes  # PNG

# report/html.py
def build_report(store: Store, comparison: str, out_html: Path, *, threshold: float = 0.5) -> Path
    # self-contained HTML: header (spec, mask, dates, runner version, spend), map grid (3 toggles),
    # error map, leaderboard, effort curve, cost vs accuracy (Pareto), lineage, diff view,
    # region table, calibration, run health, validity caveats

# ui/app.py
def create_checked_comparison(store: Store, name: str, run_ids: list[str]) -> tuple[bool, str]
    # same check_comparison as the CLI; -> (created, markdown message with refusal or warnings)
def build_app(db_path: str, registry_path: str) -> "gradio.Blocks"
def main(db_path: str = "blindearth.db", registry_path: str = "providers.yaml", port: int = 7860) -> None
```
