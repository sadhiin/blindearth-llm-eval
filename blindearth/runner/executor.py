"""Execute a plan: one asyncio task per cell, per-provider rate limiting, resumable writes.

Lifecycle of a run: queued -> running -> complete | paused | failed.
- paused: stop_event set (cancel), budget cap reached, or the cell's circuit breaker opened.
- failed: unexpected error, or too many points still failing after the end-of-run retry.
- complete: every point stored and failures within tolerance; the run is then immutable and
  scored with scoring.score.score_run.
Points are written in batches of 100 (or every few seconds for slow thinking runs).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from blindearth import __version__
from blindearth.evalspec.grid import make_grid, stratified_order
from blindearth.evalspec.masks import load_mask
from blindearth.evalspec.prompts import render_prompt
from blindearth.evalspec.truth import cell_truth
from blindearth.extract import extract
from blindearth.pricing import cost_usd
from blindearth.providers.base import Adapter
from blindearth.providers.registry import build_adapter
from blindearth.ratelimit import CircuitBreaker, ProviderLimiter, call_with_retries
from blindearth.runner.hashing import is_thinking, run_hash
from blindearth.runner.planner import (
    Plan,
    PlanCell,
    call_params,
    effective_system_prompt,
    heuristic_usage_per_point,
    make_limiters,
    order_seed,
    store_thinking_flag,
)
from blindearth.scoring.score import score_run
from blindearth.store.db import utcnow
from blindearth.types import (
    ClassifyResult,
    ExtractionMode,
    ExtractionSpec,
    Point,
    PointResult,
    RunRecord,
    RunStatus,
    Usage,
)

if TYPE_CHECKING:
    from blindearth.providers.registry import Registry
    from blindearth.store.db import Store

log = logging.getLogger(__name__)

WRITE_BATCH = 100
FLUSH_INTERVAL_S = 5.0
BATCH_API_CHUNK = 10_000
DEFAULT_BATCH_POLL_S = 30.0
# Share of points allowed to remain errored after the end-of-run retry for a run to complete.
MAX_FAILED_FRAC = 0.01

ProgressCb = Callable[[str, list[PointResult]], None]


class _Budget:
    """Hard spend cap shared by every cell of one execute() call."""

    def __init__(self, limit: float | None, spent: float = 0.0):
        self.limit = limit
        self.spent = spent
        self.reserved = 0.0  # submitted-but-unbilled batch spend
        self.unknown = False

    def add(self, usd: float | None) -> None:
        if usd is None:
            self.unknown = True
        else:
            self.spent += usd

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.spent >= self.limit

    def room(self) -> float | None:
        if self.limit is None:
            return None
        return max(0.0, self.limit - self.spent - self.reserved)


class _CellJob:
    def __init__(
        self,
        *,
        plan: Plan,
        cell: PlanCell,
        run_id: str,
        store: Store,
        limiter: ProviderLimiter,
        budget: _Budget,
        stop_event: asyncio.Event,
        on_progress: ProgressCb | None,
        truth_by_idx: dict[int, int | None],
        extraction: ExtractionSpec,
    ):
        self.plan = plan
        self.cell = cell
        self.run_id = run_id
        self.store = store
        self.limiter = limiter
        self.budget = budget
        self.stop_event = stop_event
        self.on_progress = on_progress
        self.truth_by_idx = truth_by_idx
        self.extraction = extraction
        self.spec = plan.eval_file.eval
        self.points_by_idx = {p.idx: p for p in plan.points}
        self.order = stratified_order(plan.points, order_seed(self.spec, cell.config))
        self.params = call_params(
            cell.config, cell.mode, store_thinking=store_thinking_flag(cell.model, cell.provider)
        )
        self.system_prompt = effective_system_prompt(self.spec, cell.config)
        self.breaker = CircuitBreaker(
            threshold=int(cell.provider.extra.get("breaker_threshold", 20)),
            window_s=float(cell.provider.extra.get("breaker_window_s", 60.0)),
        )
        self.n_workers = max(1, int(cell.config.concurrency or cell.provider.extra.get("concurrency") or 16))
        self.poll_s = float(cell.provider.extra.get("batch_poll_s", DEFAULT_BATCH_POLL_S))
        upp = cell.usage_per_point or heuristic_usage_per_point(self.spec, cell.config, cell.mode, cell.thinking)
        self.est_tokens = max(1, int(upp.input_tokens + upp.output_tokens))
        rec = store.get_run(run_id)
        self.totals: Usage = rec.totals
        self.cost: float | None = rec.cost_usd
        self.observed_version: str | None = rec.resolved_model_version
        self.buf: list[PointResult] = []
        self.last_flush = time.monotonic()
        self.adapter: Adapter | None = None

    # ------------------------------------------------------------------ bookkeeping

    def _cost(self, usage: Usage, batch: bool) -> float | None:
        try:
            return cost_usd(self.cell.model, self.cell.provider, usage, batch=batch)
        except Exception as e:  # noqa: BLE001
            log.warning("pricing failed: %s", e)
            return None

    def _account(self, pr: PointResult, batch: bool = False) -> None:
        self.totals = self.totals + pr.usage
        c = self._cost(pr.usage, batch)
        if c is not None:
            self.cost = (self.cost or 0.0) + c
        self.budget.add(c)

    def _flush(self) -> None:
        if not self.buf:
            return
        batch, self.buf = self.buf, []
        for i in range(0, len(batch), WRITE_BATCH):
            self.store.write_points(batch[i : i + WRITE_BATCH])
        done, _ = self.store.count_points(self.run_id)
        self.store.update_run(
            self.run_id,
            totals=self.totals,
            cost_usd=self.cost,
            n_points_done=done,
            resolved_model_version=self.observed_version,
        )
        self.last_flush = time.monotonic()
        if self.on_progress is not None:
            try:
                self.on_progress(self.run_id, batch)
            except Exception:  # noqa: BLE001 - a UI callback must not kill the run
                log.exception("on_progress callback failed")

    def _push(self, pr: PointResult) -> None:
        self.buf.append(pr)
        if len(self.buf) >= WRITE_BATCH or time.monotonic() - self.last_flush >= FLUSH_INTERVAL_S:
            self._flush()

    def _stop_reason(self) -> str | None:
        if self.stop_event.is_set():
            return "cancelled"
        if self.budget.exceeded:
            return "budget cap reached"
        if self.breaker.open:
            return "circuit breaker open (burst of failures)"
        return None

    def _make_point(self, p: Point, res: ClassifyResult | None, err: str | None) -> PointResult:
        truth = self.truth_by_idx.get(p.idx)
        if res is not None and res.resolved_model:
            self.observed_version = res.resolved_model
        if err is None and res is not None and res.error:
            err = res.error
        if err is not None or res is None:
            return PointResult(
                run_id=self.run_id,
                idx=p.idx,
                lat=p.lat,
                lon=p.lon,
                truth=truth,
                p_land=None,
                n_valid=0,
                n_samples=0,
                validity_mass=None,
                answer_text="",
                finish_reason=(res.finish_reasons[0] if res is not None and res.finish_reasons else None),
                latency_s=float(res.latency_s) if res is not None else 0.0,
                usage=res.usage if res is not None else Usage(),
                error=(err or "unknown error")[:500],
            )
        ex = extract(res, self.cell.mode)
        thinking_text = None
        if self.params.store_thinking and res.thinking_texts:
            thinking_text = res.thinking_texts[0]
        return PointResult(
            run_id=self.run_id,
            idx=p.idx,
            lat=p.lat,
            lon=p.lon,
            truth=truth,
            p_land=ex.p_land,
            n_valid=ex.n_valid,
            n_samples=ex.n_samples,
            validity_mass=ex.validity_mass,
            answer_text=ex.answer_text,
            finish_reason=res.finish_reasons[0] if res.finish_reasons else None,
            latency_s=float(res.latency_s),
            usage=res.usage,
            thinking_text=thinking_text,
            error=None,
        )

    # ------------------------------------------------------------------ live path

    async def _classify_one(self, p: Point) -> PointResult:
        assert self.adapter is not None
        adapter = self.adapter
        prompt = render_prompt(self.spec.prompt, p.lat, p.lon, self.spec.coord_format)

        async def attempt() -> ClassifyResult:
            # call_with_retries only reports to the limiter; each attempt takes its own slot.
            async with self.limiter.slot(self.est_tokens):
                return await adapter.classify(prompt, self.system_prompt, self.params)

        try:
            res = await call_with_retries(attempt, limiter=self.limiter)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - stored as a failed point, retried at the end
            self.breaker.record(False)
            return self._make_point(p, None, f"{type(e).__name__}: {e}")
        pr = self._make_point(p, res, None)
        self.breaker.record(pr.error is None)
        return pr

    async def _run_points(self, todo: list[Point]) -> str | None:
        queue = list(reversed(todo))  # pop() from the end keeps stratified order

        async def worker() -> None:
            while queue:
                if self._stop_reason():
                    return
                p = queue.pop()
                pr = await self._classify_one(p)
                self._account(pr)
                self._push(pr)

        n = min(self.n_workers, len(todo))
        if n:
            await asyncio.gather(*(worker() for _ in range(n)))
        self._flush()
        # Only pause when work is actually left; a cap hit on the last point still completes.
        return (self._stop_reason() or "stopped") if queue else None

    # ------------------------------------------------------------------ batch path

    async def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _run_batch(self, todo: list[Point]) -> str | None:
        assert self.adapter is not None
        state = self.store.run_meta(self.run_id).get("batch_state") or {}
        pending: dict[str, list[int]] = {k: list(v) for k, v in (state.get("pending") or {}).items()}
        in_flight = {i for idxs in pending.values() for i in idxs}
        to_submit = [p for p in todo if p.idx not in in_flight]
        est_pp = (
            self.cell.est_cost_usd / self.cell.n_points
            if self.cell.est_cost_usd is not None and self.cell.n_points
            else None
        )
        reserved: dict[str, float] = {}
        stop: str | None = None

        def save_state() -> None:
            self.store.update_run(self.run_id, batch_state={"pending": pending})

        # submit
        for i in range(0, len(to_submit), BATCH_API_CHUNK):
            if self._stop_reason():
                stop = self._stop_reason()
                break
            chunk = to_submit[i : i + BATCH_API_CHUNK]
            trimmed = False
            room = self.budget.room()
            if room is not None and est_pp:
                fit = int(room // est_pp)
                if fit <= 0:
                    stop = "budget cap reached"
                    break
                trimmed = fit < len(chunk)
                chunk = chunk[:fit]
            items = [
                (
                    f"p{p.idx}",
                    render_prompt(self.spec.prompt, p.lat, p.lon, self.spec.coord_format),
                    self.system_prompt,
                    self.params,
                )
                for p in chunk
            ]
            try:
                # No automatic retry: a retried submit could create a duplicate (billed) batch.
                batch_id = await self.adapter.submit_batch(items)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("batch submit failed for %s: %s", self.run_id, e)
                self.breaker.record(False)
                stop = self._stop_reason() or f"batch submit failed: {e}"
                break
            pending[batch_id] = [p.idx for p in chunk]
            if est_pp:
                reserved[batch_id] = est_pp * len(chunk)
                self.budget.reserved += reserved[batch_id]
            save_state()
            if trimmed:
                stop = "budget cap reached"  # trimmed to fit the budget; stop submitting
                break

        # poll (batches keep running at the provider while we are paused; resume re-polls)
        while pending:
            if self.stop_event.is_set():
                self._flush()
                return "cancelled"
            for batch_id in list(pending):
                try:
                    results = await self.adapter.poll_batch(batch_id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    log.warning("batch poll failed for %s: %s", batch_id, e)
                    self.breaker.record(False)
                    if self.breaker.open:
                        self._flush()
                        return "circuit breaker open (batch polling failing)"
                    continue
                if results is None:
                    continue
                self.breaker.record(True)
                for idx in pending[batch_id]:
                    p = self.points_by_idx[idx]
                    res = results.get(f"p{idx}")
                    pr = self._make_point(p, res, None if res is not None else "missing from batch output")
                    self._account(pr, batch=True)
                    self._push(pr)
                self._flush()
                self.budget.reserved -= reserved.pop(batch_id, 0.0)
                del pending[batch_id]
                save_state()
            if pending:
                await self._wait(self.poll_s)
        self._flush()
        return stop

    # ------------------------------------------------------------------ lifecycle

    def _pause(self, reason: str) -> None:
        self._flush()
        self.store.update_run(self.run_id, status=RunStatus.PAUSED, note=reason)

    async def _finalize(self) -> None:
        done, failed = self.store.count_points(self.run_id)
        total = len(self.plan.points)
        missing = total - done - failed
        if missing > 0 or failed > MAX_FAILED_FRAC * total:
            self.store.update_run(
                self.run_id,
                status=RunStatus.FAILED,
                ended_at=utcnow(),
                note=f"{failed} failed and {missing} missing points after retry; resume to retry",
                n_points_done=done,
            )
            return
        fields: dict[str, Any] = dict(
            status=RunStatus.COMPLETE,
            ended_at=utcnow(),
            totals=self.totals,
            cost_usd=self.cost,
            n_points_done=done,
            resolved_model_version=self.observed_version,
            note=(f"{failed} points errored after retry" if failed else None),
            batch_state=None,
        )
        if self.observed_version:
            # Re-key the run under the version the API actually reported, so later plans
            # (which know that version) hit the cache. plan_hash keeps the planned hash.
            fields["run_hash"] = run_hash(
                self.plan.spec_id,
                self.cell.model,
                self.observed_version,
                self.cell.config,
                self.extraction,
                self.cell.mode,
                self.cell.repeat_idx,
            )
        self.store.update_run(self.run_id, **fields)
        if self.observed_version:
            rec = self.store.get_run(self.run_id)
            self.store.set_model_resolved_version(rec.model_id, self.observed_version)
        try:
            await asyncio.to_thread(score_run, self.store, self.run_id, self.plan.mask)
        except Exception:  # noqa: BLE001 - scoring can be re-run with `blindearth score`
            log.exception("scoring failed for run %s", self.run_id)

    async def run(self) -> None:
        try:
            rec = self.store.get_run(self.run_id)
            self.store.update_run(
                self.run_id, status=RunStatus.RUNNING, started_at=rec.started_at or utcnow(), note=None
            )
            if self._stop_reason():
                self._pause(self._stop_reason() or "paused")
                return
            self.adapter = build_adapter(self.cell.provider, self.cell.model)
            caps = self.store.load_capabilities(self.cell.provider.id, self.cell.model.name)
            self.adapter.capabilities = caps if caps is not None else self.adapter.default_capabilities()

            done = self.store.done_indices(self.run_id)
            # Reuse pilot answers: they were real calls for the first points of this order.
            for idx, res in list(self.cell.pilot_results.items()):
                if idx not in done and idx in self.points_by_idx and not res.error:
                    pr = self._make_point(self.points_by_idx[idx], res, None)
                    self._account(pr)
                    self._push(pr)
                    done.add(idx)
            self.cell.pilot_results = {}
            self._flush()

            todo = [p for p in self.order if p.idx not in done]
            if self.cell.use_batch:
                reason = await self._run_batch(todo)
            else:
                reason = await self._run_points(todo)
            if reason:
                self._pause(reason)
                return

            failed = self.store.failed_indices(self.run_id)
            if failed:
                retry = [p for p in self.order if p.idx in failed]
                reason = await self._run_points(retry)
                if reason:
                    self._pause(reason)
                    return
            await self._finalize()
        except asyncio.CancelledError:
            try:
                self._pause("cancelled")
            except Exception:  # noqa: BLE001
                log.exception("could not pause run %s", self.run_id)
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("run %s failed", self.run_id)
            try:
                self._flush()
                self.store.update_run(
                    self.run_id, status=RunStatus.FAILED, ended_at=utcnow(), note=f"{type(e).__name__}: {e}"[:500]
                )
            except Exception:  # noqa: BLE001
                log.exception("could not mark run %s failed", self.run_id)
        finally:
            if self.adapter is not None:
                try:
                    await self.adapter.aclose()
                except Exception:  # noqa: BLE001
                    pass


# ----------------------------------------------------------------------------- public API


def _truth_by_idx(plan: Plan) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for p, t in zip(plan.points, list(plan.truth)):
        try:
            v = int(t)
        except (TypeError, ValueError):
            v = -1
        out[p.idx] = v if v in (0, 1) else None
    return out


def _extraction_dict(extraction: ExtractionSpec) -> dict[str, Any]:
    d = dataclasses.asdict(extraction)
    d["mode"] = ExtractionMode(extraction.mode).value
    return d


def _prepare_run(plan: Plan, cell: PlanCell, store: Store) -> str:
    if cell.resume_run_id:
        rec = store.get_run(cell.resume_run_id)
        if rec.status != RunStatus.COMPLETE:
            return rec.id
    spec = plan.eval_file.eval
    model_id = store.upsert_model(cell.model, cell.provider)
    config_id = store.upsert_config(cell.config, cell.native_params)
    run = RunRecord(
        id=uuid.uuid4().hex,
        run_hash=cell.run_hash,
        spec_id=plan.spec_id,
        model_id=model_id,
        config_id=config_id,
        variant=cell.variant,
        extraction_mode=cell.mode,
        status=RunStatus.QUEUED,
        runner_version=__version__,
        seed=order_seed(spec, cell.config),
        repeat_idx=cell.repeat_idx,
        forced_thinking=cell.forced_thinking,
        thinking=cell.thinking or is_thinking(cell.config, cell.model),
        n_points_total=len(plan.points),
    )
    store.create_run(run)
    store.update_run(run.id, extraction=_extraction_dict(plan.eval_file.extraction))
    return run.id


async def execute(
    plan: Plan,
    store: Store,
    registry: Registry,
    *,
    on_progress: Callable[[str, list[PointResult]], None] | None = None,
    budget_usd: float | None = None,
    stop_event: asyncio.Event | None = None,
) -> list[str]:
    """Run every active cell; returns run ids (cached runs included, refused cells skipped)."""
    stop_event = stop_event if stop_event is not None else asyncio.Event()
    limit = budget_usd if budget_usd is not None else plan.budget_usd
    budget = _Budget(limit)
    truth = _truth_by_idx(plan)
    run_ids: list[str] = []
    jobs: list[_CellJob] = []
    active = [c for c in plan.cells if c.refused is None and c.cached_run_id is None]
    limiters = make_limiters(active, store) if active else {}

    for cell in plan.cells:
        if cell.refused is not None:
            continue
        if cell.cached_run_id is not None:
            run_ids.append(cell.cached_run_id)
            continue
        run_id = _prepare_run(plan, cell, store)
        rec = store.get_run(run_id)
        if rec.status == RunStatus.COMPLETE:
            run_ids.append(run_id)
            continue
        if rec.cost_usd:
            budget.spent += rec.cost_usd  # earlier spend of a resumed run counts toward the cap
        run_ids.append(run_id)
        jobs.append(
            _CellJob(
                plan=plan,
                cell=cell,
                run_id=run_id,
                store=store,
                limiter=limiters[cell.provider.id],
                budget=budget,
                stop_event=stop_event,
                on_progress=on_progress,
                truth_by_idx=truth,
                extraction=plan.eval_file.extraction,
            )
        )

    if jobs:
        await asyncio.gather(*(j.run() for j in jobs))
    if budget.unknown and limit is not None:
        log.warning("some calls had no known price; the budget cap could not count them")
    return run_ids


async def resume(run_id: str, store: Store, registry: Registry, **kw: Any) -> None:
    """Continue an unfinished run: missing points first, then failed points."""
    from blindearth.evalspec.load import EvalFile

    rec = store.get_run(run_id)
    if rec.status == RunStatus.COMPLETE:
        log.info("run %s is already complete", rec.id)
        return
    spec, mask_hash, _ = store.get_eval_spec(rec.spec_id)
    mask = load_mask(spec.mask)
    if mask.hash != mask_hash:
        raise RuntimeError(
            f"mask for spec {rec.spec_id[:12]} changed since the run started "
            f"({mask_hash[:12]} -> {mask.hash[:12]}); start a new run instead"
        )
    model = store.get_model(rec.model_id)
    if model.provider not in registry.providers:
        raise KeyError(f"provider '{model.provider}' is not in the registry; add it to resume")
    provider = registry.providers[model.provider]
    config, native = store.get_config(rec.config_id)
    meta = store.run_meta(rec.id)
    ex = dict(meta.get("extraction") or {})
    extraction = ExtractionSpec(
        mode=ExtractionMode(ex.get("mode", rec.extraction_mode.value)),
        n_samples=int(ex.get("n_samples", 4)),
        temperature=float(ex.get("temperature", 1.0)),
    )
    points = make_grid(spec.grid)
    truth = cell_truth(mask, points, spec.grid, spec.mask.truth_rule)
    adapter_kind_batch = bool(provider.use_batch)
    if adapter_kind_batch:
        probe_adapter = build_adapter(provider, model)
        adapter_kind_batch = bool(getattr(probe_adapter, "supports_batch", False))
        await probe_adapter.aclose()
    done = store.done_indices(rec.id)
    cell = PlanCell(
        model=model,
        provider=provider,
        config=config,
        variant=rec.variant,
        mode=rec.extraction_mode,
        repeat_idx=rec.repeat_idx,
        run_hash=rec.run_hash,
        n_points=len(points) - len(done),
        n_calls=0,
        est_usage=Usage(),
        est_cost_usd=None,
        cached_run_id=None,
        refused=None,
        native_params=native,
        thinking=rec.thinking,
        forced_thinking=rec.forced_thinking,
        resolved_version=rec.resolved_model_version,
        resume_run_id=rec.id,
        use_batch=adapter_kind_batch,
    )
    eval_file = EvalFile(eval=spec, extraction=extraction, matrix=[], budget_usd=None, name=None)
    plan = Plan(
        eval_file=eval_file,
        spec_id=rec.spec_id,
        mask=mask,
        points=points,
        truth=truth,
        cells=[cell],
        total_cost_usd=None,
        budget_usd=kw.pop("budget_usd", None),
    )
    await execute(plan, store, registry, budget_usd=plan.budget_usd, **kw)
