"""Plan step: expand the matrix into cells, resolve modes, map configs, find cache hits,
run a small pilot per uncached cell, and estimate tokens and cost.

Nothing here writes points. The only spend is the pilot (`run_pilot=False` makes it zero).
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from blindearth.evalspec.grid import make_grid, stratified_order
from blindearth.evalspec.masks import load_mask
from blindearth.evalspec.prompts import render_prompt
from blindearth.evalspec.truth import cell_truth
from blindearth.pricing import cost_usd
from blindearth.providers.base import Adapter, UnsupportedConfigError
from blindearth.providers.registry import build_adapter
from blindearth.ratelimit import ProviderLimiter, call_with_retries
from blindearth.runner.hashing import effective_config, is_thinking, model_always_thinks, run_hash
from blindearth.types import (
    CallParams,
    Capabilities,
    ClassifyResult,
    EvalSpec,
    ExtractionMode,
    ModelSpec,
    Point,
    ProviderSpec,
    RunConfig,
    RunStatus,
    Usage,
)

if TYPE_CHECKING:
    import numpy as np

    from blindearth.evalspec.load import EvalFile
    from blindearth.evalspec.masks import Mask
    from blindearth.providers.registry import Registry
    from blindearth.store.db import Store

log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8
PILOT_MAX_CONCURRENCY = 8
# Heuristic thinking tokens per sample when no pilot or stored data is available.
THINKING_TOKENS_GUESS = {"low": 400, "medium": 1500, "high": 4000, "max": 10000}
DEFAULT_THINKING_TOKENS_GUESS = 1500
# Stored points needed on a resumable run before we trust them over a pilot.
MIN_STORED_FOR_ESTIMATE = 10


@dataclass
class PlanCell:
    model: ModelSpec
    provider: ProviderSpec
    config: RunConfig  # effective config (all output-affecting fields resolved)
    variant: str
    mode: ExtractionMode
    repeat_idx: int
    run_hash: str
    n_points: int  # points still to call for this cell (0 when cached)
    n_calls: int
    est_usage: Usage
    est_cost_usd: float | None
    cached_run_id: str | None
    refused: str | None
    native_params: dict
    # Extensions (defaults only; not in INTERFACES.md):
    thinking: bool = False
    forced_thinking: bool = False
    resolved_version: str | None = None  # version used in run_hash (None = model name)
    resume_run_id: str | None = None  # unfinished run with the same hash, continued by execute()
    use_batch: bool = False
    est_source: str = "heuristic"  # pilot | stored | heuristic
    usage_per_point: Usage | None = None
    pilot_results: dict[int, ClassifyResult] = field(default_factory=dict)
    pilot_cost_usd: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Variant name, starred when the model forces thinking (as in the original chart)."""
        rep = f" #{self.repeat_idx + 1}" if self.repeat_idx else ""
        return self.variant + ("*" if self.forced_thinking else "") + rep

    @property
    def active(self) -> bool:
        return self.refused is None and self.cached_run_id is None


@dataclass
class Plan:
    eval_file: EvalFile
    spec_id: str
    mask: Mask
    points: list[Point]
    truth: np.ndarray
    cells: list[PlanCell]
    total_cost_usd: float | None
    budget_usd: float | None
    # Extensions:
    warnings: list[str] = field(default_factory=list)
    mixed_modes: bool = False
    pilot_cost_usd: float | None = None


# ----------------------------------------------------------------------------- helpers


def resolve_mode(
    requested: ExtractionMode, caps: Capabilities | None, thinking: bool
) -> tuple[ExtractionMode, str | None, str | None]:
    """-> (mode, note, refusal)."""
    requested = ExtractionMode(requested)
    if thinking:
        if requested == ExtractionMode.LOGPROBS:
            return ExtractionMode.SAMPLE, "thinking run: logprobs replaced by sampling", None
        if requested == ExtractionMode.AUTO:
            return ExtractionMode.SAMPLE, None, None
        return requested, None, None
    if requested == ExtractionMode.AUTO:
        return (ExtractionMode.LOGPROBS if caps is not None and caps.logprobs else ExtractionMode.SAMPLE), None, None
    if requested == ExtractionMode.LOGPROBS and caps is not None and not caps.logprobs:
        return requested, None, "logprobs requested but the capability probe found none"
    return requested, None, None


def effective_system_prompt(spec: EvalSpec, config: RunConfig) -> str | None:
    return config.system_prompt if config.system_prompt is not None else spec.prompt.system_prompt


def order_seed(spec: EvalSpec, config: RunConfig) -> int:
    return int(config.seed) if config.seed is not None else int(spec.grid.seed)


def store_thinking_flag(model: ModelSpec, provider: ProviderSpec) -> bool:
    return bool(model.extra.get("store_thinking", provider.extra.get("store_thinking", False)))


def call_params(config: RunConfig, mode: ExtractionMode, *, store_thinking: bool = False) -> CallParams:
    """Effective config -> per-call parameters handed to the adapter."""
    mode = ExtractionMode(mode)
    logprobs = mode == ExtractionMode.LOGPROBS
    return CallParams(
        temperature=config.temperature,
        max_output_tokens=int(config.max_output_tokens or 16),
        effort=config.effort,
        logprobs=logprobs,
        top_logprobs=config.top_logprobs if logprobs else None,
        n=int(config.n_samples or 1) if mode == ExtractionMode.SAMPLE else 1,
        seed=config.seed,
        store_thinking=store_thinking,
    )


def scale_usage(u: Usage, k: float) -> Usage:
    return Usage(
        int(round(u.input_tokens * k)),
        int(round(u.output_tokens * k)),
        int(round(u.thinking_tokens * k)),
    )


def heuristic_usage_per_point(spec: EvalSpec, config: RunConfig, mode: ExtractionMode, thinking: bool) -> Usage:
    try:
        prompt = render_prompt(spec.prompt, 45.0, -45.0, spec.coord_format)
    except Exception:  # pragma: no cover - defensive; B may validate templates
        prompt = spec.prompt.template
    sys_prompt = effective_system_prompt(spec, config) or ""
    in_tok = math.ceil((len(prompt) + len(sys_prompt)) / 4) + 8
    n = int(config.n_samples or 1) if mode == ExtractionMode.SAMPLE else 1
    think = THINKING_TOKENS_GUESS.get((config.effort or "").lower(), DEFAULT_THINKING_TOKENS_GUESS) if thinking else 0
    # Conservative: assume input is billed once per sample (adapters without `n` loop).
    return Usage(in_tok * n, (2 + think) * n, think * n)


def _safe_cost(model: ModelSpec, provider: ProviderSpec, usage: Usage, batch: bool) -> float | None:
    try:
        return cost_usd(model, provider, usage, batch=batch)
    except Exception as e:  # pricing must never break planning
        log.warning("pricing failed for %s: %s", model.ref, e)
        return None


def _limiter_settings(cells: list[PlanCell], caps_by_provider: dict[str, Capabilities | None]) -> dict[str, tuple[int, int | None, int | None]]:
    """provider id -> (concurrency, rpm, tpm) from configs, probe and provider.extra."""
    out: dict[str, tuple[int, int | None, int | None]] = {}
    by_provider: dict[str, list[PlanCell]] = {}
    for c in cells:
        by_provider.setdefault(c.provider.id, []).append(c)
    for pid, cs in by_provider.items():
        prov = cs[0].provider
        conc_vals = [c.config.concurrency for c in cs if c.config.concurrency]
        caps = caps_by_provider.get(pid)
        if conc_vals:
            conc = max(conc_vals)
        elif prov.extra.get("concurrency"):
            conc = int(prov.extra["concurrency"])
        elif caps is not None and caps.max_concurrency:
            conc = int(caps.max_concurrency)
        else:
            conc = DEFAULT_CONCURRENCY
        rpms = [c.config.rpm for c in cs if c.config.rpm] + ([int(prov.extra["rpm"])] if prov.extra.get("rpm") else [])
        tpms = [c.config.tpm for c in cs if c.config.tpm] + ([int(prov.extra["tpm"])] if prov.extra.get("tpm") else [])
        out[pid] = (conc, min(rpms) if rpms else None, min(tpms) if tpms else None)
    return out


def make_limiters(cells: list[PlanCell], store: Store) -> dict[str, ProviderLimiter]:
    caps_by_provider: dict[str, Capabilities | None] = {}
    for c in cells:
        if c.provider.id not in caps_by_provider:
            caps_by_provider[c.provider.id] = store.load_capabilities(c.provider.id, c.model.name)
    return {
        pid: ProviderLimiter(concurrency=conc, rpm=rpm, tpm=tpm)
        for pid, (conc, rpm, tpm) in _limiter_settings(cells, caps_by_provider).items()
    }


# ----------------------------------------------------------------------------- pilot


async def _run_pilot(
    cell: PlanCell,
    adapter: Adapter,
    spec: EvalSpec,
    pilot_points: list[Point],
    limiter: ProviderLimiter,
) -> None:
    params = call_params(cell.config, cell.mode, store_thinking=store_thinking_flag(cell.model, cell.provider))
    sys_prompt = effective_system_prompt(spec, cell.config)
    guess = heuristic_usage_per_point(spec, cell.config, cell.mode, cell.thinking)
    est_tokens = max(1, guess.input_tokens + guess.output_tokens)

    async def one(p: Point) -> tuple[int, ClassifyResult | None, str | None]:
        prompt = render_prompt(spec.prompt, p.lat, p.lon, spec.coord_format)

        async def attempt() -> ClassifyResult:
            # call_with_retries only reports to the limiter; each attempt takes its own slot.
            async with limiter.slot(est_tokens):
                return await adapter.classify(prompt, sys_prompt, params)

        try:
            res = await call_with_retries(attempt, limiter=limiter)
        except Exception as e:  # noqa: BLE001 - pilot failures become notes
            return p.idx, None, f"{type(e).__name__}: {e}"
        if res.error:
            return p.idx, None, res.error
        return p.idx, res, None

    results = await asyncio.gather(*(one(p) for p in pilot_points))
    ok = [(i, r) for i, r, err in results if r is not None]
    errors = [err for _, r, err in results if r is None]
    if not ok:
        cell.notes.append(f"pilot failed ({len(errors)} errors): {errors[0] if errors else 'no results'}")
        return
    total = Usage()
    for _, r in ok:
        total = total + r.usage
    cell.pilot_results = {i: r for i, r in ok}
    cell.usage_per_point = scale_usage(total, 1.0 / len(ok))
    cell.est_source = "pilot"
    cell.pilot_cost_usd = _safe_cost(cell.model, cell.provider, total, False)
    if errors:
        cell.notes.append(f"pilot: {len(errors)}/{len(pilot_points)} calls failed")
    versions = [r.resolved_model for _, r in ok if r.resolved_model]
    if versions:
        cell.notes.append(f"resolved version: {versions[-1]}")


def _stored_usage_per_point(store: Store, run_id: str) -> Usage | None:
    df = store.load_points(run_id)
    df = df[df["error"].isna()]
    if len(df) < MIN_STORED_FOR_ESTIMATE:
        return None
    return Usage(
        int(round(df["input_tokens"].mean())),
        int(round(df["output_tokens"].mean())),
        int(round(df["thinking_tokens"].mean())),
    )


# ----------------------------------------------------------------------------- build_plan


async def build_plan(
    eval_file: EvalFile,
    registry: Registry,
    store: Store,
    *,
    pilot_points: int = 50,
    run_pilot: bool = True,
) -> Plan:
    spec: EvalSpec = eval_file.eval
    extraction = eval_file.extraction
    mask = load_mask(spec.mask)
    spec_id = store.upsert_eval_spec(spec, mask.hash, mask.source)
    points = make_grid(spec.grid)
    truth = cell_truth(mask, points, spec.grid, spec.mask.truth_rule)
    n_total = len(points)

    cells: list[PlanCell] = []
    adapters: dict[int, Adapter] = {}  # id(cell) -> adapter, for the pilot
    warnings: list[str] = []
    seen_hashes: dict[str, str] = {}

    try:
        for entry in eval_file.matrix:
            model = registry.model(entry.model_ref)
            provider = registry.provider_of(model)
            caps = store.load_capabilities(provider.id, model.name)
            adapter = build_adapter(provider, model)
            adapter.capabilities = caps if caps is not None else adapter.default_capabilities()
            resolved = store.known_resolved_version(model, provider)
            configs = list(entry.configs) or [RunConfig()]
            for raw_cfg in configs:
                thinking = is_thinking(raw_cfg, model)
                mode, mode_note, refusal = resolve_mode(extraction.mode, caps, thinking)
                cap = caps.top_logprobs_max if caps is not None else None
                cfg = effective_config(raw_cfg, extraction, mode, thinking=thinking, top_logprobs_cap=cap)
                notes: list[str] = []
                if mode_note:
                    notes.append(mode_note)
                if thinking and (raw_cfg.max_output_tokens or 0) < (cfg.max_output_tokens or 0):
                    notes.append(f"max_output_tokens raised to {cfg.max_output_tokens} for thinking")
                native: dict[str, Any] = {}
                if refusal is None:
                    try:
                        native = dict(adapter.map_config(cfg, mode) or {})
                    except UnsupportedConfigError as e:
                        refusal = str(e) or "unsupported configuration"
                # Name variants by what the user set, not by the filled-in defaults.
                variant = raw_cfg.variant_name(model.id)
                use_batch = bool(provider.use_batch and getattr(adapter, "supports_batch", False))
                if provider.use_batch and not use_batch:
                    notes.append("batch requested but adapter has no batch API; using live calls")
                for r in range(max(1, int(raw_cfg.repeats or 1))):
                    h = run_hash(spec_id, model, resolved, cfg, extraction, mode, r)
                    cell = PlanCell(
                        model=model,
                        provider=provider,
                        config=cfg,
                        # Plain RunConfig.variant_name (types.RunRecord.variant); repeats are told
                        # apart by repeat_idx, which reports/UI append to the label themselves.
                        variant=variant,
                        mode=mode,
                        repeat_idx=r,
                        run_hash=h,
                        n_points=0,
                        n_calls=0,
                        est_usage=Usage(),
                        est_cost_usd=None,
                        cached_run_id=None,
                        refused=refusal,
                        native_params=native,
                        thinking=thinking,
                        # Star models that cannot stop thinking, flagged or not.
                        forced_thinking=model_always_thinks(model, provider.kind),
                        resolved_version=resolved,
                        use_batch=use_batch,
                        notes=list(notes),
                    )
                    if cell.refused is None and h in seen_hashes:
                        cell.refused = f"duplicate of cell {seen_hashes[h]}"
                    seen_hashes.setdefault(h, cell.label)
                    if cell.refused is None:
                        hit = store.find_complete_run(h)
                        if hit is not None:
                            cell.cached_run_id = hit.id
                        else:
                            unfinished = store.find_resumable_run(h)
                            remaining = n_total
                            if unfinished is not None:
                                cell.resume_run_id = unfinished.id
                                remaining = n_total - len(store.done_indices(unfinished.id))
                                stored = _stored_usage_per_point(store, unfinished.id)
                                if stored is not None:
                                    cell.usage_per_point = stored
                                    cell.est_source = "stored"
                                cell.notes.append(
                                    f"resumes {unfinished.status.value} run {unfinished.id[:12]} "
                                    f"({n_total - remaining}/{n_total} done)"
                                )
                            cell.n_points = remaining
                            per_point_calls = cfg.n_samples if mode == ExtractionMode.SAMPLE else 1
                            cell.n_calls = remaining * int(per_point_calls or 1)
                    cells.append(cell)
                    adapters[id(cell)] = adapter

        # ---- pilot (uncached, unrefused cells without stored estimates)
        pilot_cells = [
            c for c in cells if c.active and c.n_points > 0 and c.est_source == "heuristic"
        ]
        if run_pilot and pilot_points > 0 and pilot_cells:
            limiters = {
                pid: ProviderLimiter(concurrency=min(conc, PILOT_MAX_CONCURRENCY), rpm=rpm, tpm=tpm)
                for pid, (conc, rpm, tpm) in _limiter_settings(
                    pilot_cells, {c.provider.id: store.load_capabilities(c.provider.id, c.model.name) for c in pilot_cells}
                ).items()
            }

            async def pilot(c: PlanCell) -> None:
                done: set[int] = store.done_indices(c.resume_run_id) if c.resume_run_id else set()
                order = stratified_order(points, order_seed(spec, c.config))
                chosen = [p for p in order if p.idx not in done][: min(pilot_points, c.n_points)]
                await _run_pilot(c, adapters[id(c)], spec, chosen, limiters[c.provider.id])

            await asyncio.gather(*(pilot(c) for c in pilot_cells))

        # ---- estimates
        for c in cells:
            if not c.active or c.n_points == 0:
                continue
            if c.usage_per_point is None:
                c.usage_per_point = heuristic_usage_per_point(spec, c.config, c.mode, c.thinking)
            c.est_usage = scale_usage(c.usage_per_point, c.n_points)
            c.est_cost_usd = _safe_cost(c.model, c.provider, c.est_usage, c.use_batch)
            if c.est_cost_usd is None:
                c.notes.append("no price known; cost not estimated")
    finally:
        closed: set[int] = set()
        for a in adapters.values():
            if id(a) in closed:
                continue
            closed.add(id(a))
            try:
                await a.aclose()
            except Exception:  # pragma: no cover
                pass

    # ---- fairness and totals
    active = [c for c in cells if c.active]
    usable = [c for c in cells if c.refused is None]
    modes = {c.mode for c in usable}
    mixed = len(modes) > 1
    if mixed:
        warnings.append(
            "mixed extraction modes (" + ", ".join(sorted(m.value for m in modes)) + "): "
            "leaderboard ranks on thresholded accuracy only"
        )
    forced = [c.label for c in usable if c.forced_thinking]
    if forced:
        warnings.append("forced-thinking models (starred): " + ", ".join(forced))
    if any(c.thinking for c in usable) and any(not c.thinking for c in usable):
        warnings.append("thinking and non-thinking cells are not like for like")
    if any(c.model.quant for c in usable):
        warnings.append("quantized local models are not the full-precision model")
    for c in cells:
        if c.refused:
            warnings.append(f"refused {c.label}: {c.refused}")

    known = [c.est_cost_usd for c in active if c.est_cost_usd is not None]
    if not active:
        total: float | None = 0.0
    elif known:
        total = float(sum(known))
        if len(known) < len(active):
            warnings.append("some cells have no price; total cost is a lower bound")
    else:
        total = None
    budget = getattr(eval_file, "budget_usd", None)
    if budget is not None and total is not None and total > budget:
        warnings.append(f"estimated cost ${total:.2f} exceeds budget ${budget:.2f}; runs will pause at the cap")
    pilot_costs = [c.pilot_cost_usd for c in cells if c.pilot_cost_usd is not None]

    return Plan(
        eval_file=eval_file,
        spec_id=spec_id,
        mask=mask,
        points=points,
        truth=truth,
        cells=cells,
        total_cost_usd=total,
        budget_usd=budget,
        warnings=warnings,
        mixed_modes=mixed,
        pilot_cost_usd=float(sum(pilot_costs)) if pilot_costs else None,
    )


# ----------------------------------------------------------------------------- comparisons


def check_comparison(store: Store, run_ids: list[str]) -> dict[str, Any]:
    """Fairness rules for a comparison of saved runs.

    Raises ValueError when runs use different eval specs. Returns filters to store with the
    comparison: {spec_id, mixed_extraction_modes, rank_on, forced_thinking_runs, warnings}.
    """
    runs = [store.get_run(r) for r in run_ids]
    specs = {r.spec_id for r in runs}
    if len(specs) > 1:
        raise ValueError(
            "runs use different eval specs (mask, grid, prompt or coordinate format differ): "
            + ", ".join(sorted(s[:12] for s in specs))
        )
    warnings: list[str] = []
    modes = {r.extraction_mode for r in runs}
    mixed = len(modes) > 1
    if mixed:
        warnings.append("mixed extraction modes: ranking on thresholded accuracy only")
    unfinished = [r.id[:12] for r in runs if r.status != RunStatus.COMPLETE]
    if unfinished:
        warnings.append("partial runs scored on points done so far: " + ", ".join(unfinished))
    forced = [r.id for r in runs if r.forced_thinking]
    if forced:
        warnings.append(f"{len(forced)} forced-thinking run(s) marked")
    if any(r.thinking for r in runs) and any(not r.thinking for r in runs):
        warnings.append("thinking and non-thinking runs are not like for like")
    return {
        "spec_id": next(iter(specs)) if specs else None,
        "mixed_extraction_modes": mixed,
        "rank_on": "acc_area" if mixed else None,
        "forced_thinking_runs": forced,
        "warnings": warnings,
    }
