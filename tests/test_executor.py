"""Executor tests with fake adapters, limiter, breaker and extractor. No network."""

from __future__ import annotations

import contextlib
import asyncio
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from blindearth.providers.base import Adapter, ProviderError
from blindearth.runner import executor, planner
from blindearth.runner.hashing import effective_config, run_hash
from blindearth.runner.planner import Plan, PlanCell
from blindearth.store import db as dbmod
from blindearth.store.db import Store
from blindearth.types import (
    ClassifyResult,
    EvalSpec,
    Extracted,
    ExtractionMode,
    ExtractionSpec,
    GridSpec,
    ModelSpec,
    Point,
    ProviderSpec,
    RunConfig,
    RunStatus,
    Usage,
)

N = 250
SAMPLE = ExtractionMode.SAMPLE


class ExecAdapter(Adapter):
    kind = "fake"

    def __init__(self, provider=None, model=None, *, fail_once=(), always_fail=False, version=None,
                 batch=False, on_call=None):
        super().__init__(provider, model)
        self.fail_once = set(fail_once)
        self.always_fail = always_fail
        self.version = version
        self.supports_batch = batch
        self.on_call = on_call
        self.prompts: list[str] = []
        self.submitted: list[list] = []
        self.polls = 0

    def _result(self, idx: int, n: int) -> ClassifyResult:
        return ClassifyResult(
            texts=["Land" if idx % 2 else "Water"] * n,
            usage=Usage(50 * n, 2 * n, 0),
            latency_s=0.01,
            finish_reasons=["stop"] * n,
            resolved_model=self.version,
        )

    async def classify(self, prompt, system_prompt, params):
        self.prompts.append(prompt)
        if self.on_call:
            self.on_call(self)
        idx = int(prompt)
        if self.always_fail:
            raise ProviderError("invalid api key", status=401)
        if idx in self.fail_once:
            self.fail_once.discard(idx)
            raise ProviderError("server error", status=500, retryable=True)
        return self._result(idx, params.n)

    def map_config(self, config, mode):
        return {}

    async def submit_batch(self, items):
        self.submitted.append(items)
        return f"batch{len(self.submitted)}"

    async def poll_batch(self, batch_id):
        self.polls += 1
        if self.polls == 1:
            return None  # still running
        items = self.submitted[int(batch_id.removeprefix("batch")) - 1]
        return {cid: self._result(int(prompt), params.n) for cid, prompt, _sys, params in items}


class FakeBreaker:
    def __init__(self, threshold=20, window_s=60.0):
        self.threshold = threshold
        self.fails = 0

    def record(self, ok):
        self.fails = 0 if ok else self.fails + 1

    @property
    def open(self):
        return self.fails >= self.threshold


class FakeLimiter:
    def __init__(self, concurrency, rpm=None, tpm=None):
        self.concurrency = concurrency

    def slot(self, est_tokens):
        return contextlib.nullcontext()


async def fake_cwr(fn, *, max_attempts=5, limiter=None):
    return await fn()


def fake_extract(res, mode):
    t = res.texts[0] if res.texts else ""
    p = {"Land": 1.0, "Water": 0.0}.get(t)
    return Extracted(p_land=p, n_valid=0 if p is None else len(res.texts), n_samples=len(res.texts),
                     validity_mass=None, answer_text=t, invalid=p is None)


POINTS = [Point(i, 0.0, float(i), 1.0) for i in range(N)]
TRUTH = np.array([i % 2 for i in range(N)], dtype=np.int8)
MASK = SimpleNamespace(hash="mh", source="fake", data=None)


@pytest.fixture
def env(monkeypatch, tmp_path):
    state = SimpleNamespace(adapters={}, scored=[], costs=None)

    def build_adapter(provider, model):
        return state.adapters[model.id]

    def cost(model, provider, usage, batch=False):
        return state.costs if state.costs is not None else usage.input_tokens * 1e-6

    monkeypatch.setattr(executor, "build_adapter", build_adapter)
    monkeypatch.setattr(executor, "render_prompt", lambda prompt, lat, lon, fmt: str(int(lon)))
    monkeypatch.setattr(executor, "extract", fake_extract)
    monkeypatch.setattr(executor, "stratified_order", lambda pts, seed: list(reversed(pts)))
    monkeypatch.setattr(executor, "cost_usd", cost)
    monkeypatch.setattr(executor, "CircuitBreaker", FakeBreaker)
    monkeypatch.setattr(executor, "call_with_retries", fake_cwr)
    monkeypatch.setattr(executor, "score_run", lambda store, run_id, mask=None, **kw: state.scored.append(run_id) or {})
    monkeypatch.setattr(executor, "load_mask", lambda spec, cache_dir=None: MASK)
    monkeypatch.setattr(executor, "make_grid", lambda grid: list(POINTS))
    monkeypatch.setattr(executor, "cell_truth", lambda mask, pts, grid, rule: TRUTH.copy())
    monkeypatch.setattr(planner, "ProviderLimiter", FakeLimiter)
    monkeypatch.setattr(
        dbmod, "_spec_hash", lambda spec, mh: hashlib.sha256(json.dumps([spec.to_dict(), mh], sort_keys=True).encode()).hexdigest()
    )
    state.store = Store(tmp_path / "e.db")
    yield state
    state.store.close()


def make_plan(store, model_ids=("m1",), *, use_batch=False, extra=None):
    spec = EvalSpec(grid=GridSpec(step_deg=4.0))
    spec_id = store.upsert_eval_spec(spec, "mh", "fake")
    provider = ProviderSpec(
        id="prov", kind="openai_compatible", use_batch=use_batch,
        extra={"breaker_threshold": 3, "batch_poll_s": 0, "concurrency": 4, **(extra or {})},
    )
    extraction = ExtractionSpec()
    cells = []
    for mid in model_ids:
        model = ModelSpec(id=mid, provider="prov", name=f"{mid}-name")
        cfg = effective_config(RunConfig(), extraction, SAMPLE, thinking=False)
        cells.append(
            PlanCell(
                model=model, provider=provider, config=cfg, variant=mid, mode=SAMPLE, repeat_idx=0,
                run_hash=run_hash(spec_id, model, None, cfg, extraction, SAMPLE, 0),
                n_points=N, n_calls=N * 4, est_usage=Usage(), est_cost_usd=N * 200e-6,
                cached_run_id=None, refused=None, native_params={}, use_batch=use_batch,
            )
        )
    ef = SimpleNamespace(eval=spec, extraction=extraction, matrix=[], budget_usd=None, name="t")
    return Plan(eval_file=ef, spec_id=spec_id, mask=MASK, points=list(POINTS), truth=TRUTH.copy(),
                cells=cells, total_cost_usd=None, budget_usd=None)


class FakeRegistry:
    def __init__(self, plan):
        self.providers = {c.provider.id: c.provider for c in plan.cells}
        self.models = {c.model.id: c.model for c in plan.cells}


async def test_full_run_completes_in_batches(env, monkeypatch):
    store = env.store
    plan = make_plan(store)
    env.adapters["m1"] = ExecAdapter()
    sizes, progress = [], []
    orig = store.write_points
    monkeypatch.setattr(store, "write_points", lambda pts: (sizes.append(len(pts)), orig(pts))[1])

    ids = await executor.execute(plan, store, FakeRegistry(plan), on_progress=lambda rid, b: progress.append(len(b)))
    assert len(ids) == 1
    rec = store.get_run(ids[0])
    assert rec.status == RunStatus.COMPLETE and rec.n_points_done == N and rec.ended_at
    assert max(sizes) <= 100 and sum(sizes) == N
    assert sum(progress) == N
    assert env.adapters["m1"].prompts[0] == str(N - 1)  # stratified order (reversed in the fake)
    assert rec.totals.input_tokens == N * 200
    assert rec.cost_usd == pytest.approx(N * 200e-6)
    df = store.load_points(rec.id)
    assert len(df) == N and (df["p_land"] == df["truth"]).all()
    assert env.scored == [rec.id]


async def test_cancel_then_resume_continues_missing_points(env):
    store = env.store
    plan = make_plan(store)
    stop = asyncio.Event()

    def stop_after_30(a):
        if len(a.prompts) >= 30:
            stop.set()

    env.adapters["m1"] = first = ExecAdapter(on_call=stop_after_30)
    [rid] = await executor.execute(plan, store, FakeRegistry(plan), stop_event=stop)
    rec = store.get_run(rid)
    assert rec.status == RunStatus.PAUSED
    done = store.done_indices(rid)
    assert len(done) == 30 and env.scored == []

    env.adapters["m1"] = second = ExecAdapter()
    await executor.resume(rid, store, FakeRegistry(plan))
    assert store.get_run(rid).status == RunStatus.COMPLETE
    assert len(second.prompts) == N - 30
    assert not ({int(p) for p in second.prompts} & done)
    assert len(first.prompts) == 30


async def test_failed_points_retried_at_end(env):
    store = env.store
    plan = make_plan(store)
    env.adapters["m1"] = a = ExecAdapter(fail_once={5, 17})
    [rid] = await executor.execute(plan, store, FakeRegistry(plan))
    assert store.get_run(rid).status == RunStatus.COMPLETE
    assert store.failed_indices(rid) == set()
    assert a.prompts[-2:] == ["17", "5"]
    assert len(a.prompts) == N + 2


async def test_circuit_breaker_pauses_run(env):
    store = env.store
    plan = make_plan(store)
    env.adapters["m1"] = a = ExecAdapter(always_fail=True)
    [rid] = await executor.execute(plan, store, FakeRegistry(plan))
    rec = store.get_run(rid)
    assert rec.status == RunStatus.PAUSED
    assert "circuit breaker" in store.run_meta(rid)["note"]
    assert len(a.prompts) == 3
    assert store.failed_indices(rid) == {N - 1, N - 2, N - 3}


async def test_budget_cap_pauses_all_runs(env):
    store = env.store
    plan = make_plan(store, ("m1", "m2"))
    env.adapters["m1"] = a1 = ExecAdapter()
    env.adapters["m2"] = a2 = ExecAdapter()
    env.costs = 1.0
    ids = await executor.execute(plan, store, FakeRegistry(plan), budget_usd=10.0)
    assert [store.get_run(r).status for r in ids] == [RunStatus.PAUSED, RunStatus.PAUSED]
    assert len(a1.prompts) + len(a2.prompts) == 10
    assert all("budget" in store.run_meta(r)["note"] for r in ids)


async def test_stop_before_start_spends_nothing(env):
    store = env.store
    plan = make_plan(store)
    env.adapters["m1"] = a = ExecAdapter()
    stop = asyncio.Event()
    stop.set()
    [rid] = await executor.execute(plan, store, FakeRegistry(plan), stop_event=stop)
    assert store.get_run(rid).status == RunStatus.PAUSED
    assert a.prompts == []


async def test_batch_api_path(env):
    store = env.store
    plan = make_plan(store, use_batch=True)
    env.adapters["m1"] = a = ExecAdapter(batch=True)
    [rid] = await executor.execute(plan, store, FakeRegistry(plan))
    rec = store.get_run(rid)
    assert rec.status == RunStatus.COMPLETE and rec.n_points_done == N
    assert a.prompts == [] and len(a.submitted) == 1 and len(a.submitted[0]) == N
    assert store.run_meta(rid)["batch_state"] is None
    assert env.scored == [rid]


async def test_resolved_version_rekeys_run_hash(env):
    store = env.store
    plan = make_plan(store)
    cell = plan.cells[0]
    env.adapters["m1"] = ExecAdapter(version="m1-2026-10-01")
    [rid] = await executor.execute(plan, store, FakeRegistry(plan))
    rec = store.get_run(rid)
    expected = run_hash(plan.spec_id, cell.model, "m1-2026-10-01", cell.config, plan.eval_file.extraction, SAMPLE, 0)
    assert rec.run_hash == expected != cell.run_hash
    assert store.run_meta(rid)["plan_hash"] == cell.run_hash
    assert rec.resolved_model_version == "m1-2026-10-01"
    assert store.known_resolved_version(cell.model, cell.provider) == "m1-2026-10-01"
    assert store.find_complete_run(expected).id == rid


async def test_cached_refused_and_pilot_reuse(env):
    store = env.store
    plan = make_plan(store, ("m1", "m2", "m3"))
    plan.cells[0].cached_run_id = "old-run"
    plan.cells[1].refused = "effort max unsupported"
    pilot = {i: ExecAdapter()._result(i, 4) for i in range(N - 5, N)}
    plan.cells[2].pilot_results = dict(pilot)
    env.adapters["m3"] = a = ExecAdapter()
    ids = await executor.execute(plan, store, FakeRegistry(plan))
    assert ids[0] == "old-run" and len(ids) == 2
    assert store.get_run(ids[1]).status == RunStatus.COMPLETE
    assert len(a.prompts) == N - 5
    assert not ({int(p) for p in a.prompts} & set(pilot))
