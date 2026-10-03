"""Planner tests with a fake registry and fake adapters. No network, no real masks."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from blindearth.providers.base import Adapter, UnsupportedConfigError
from blindearth.runner import planner
from blindearth.runner.hashing import THINKING_MIN_OUTPUT_TOKENS
from blindearth.store import db as dbmod
from blindearth.store.db import Store
from blindearth.types import (
    Capabilities,
    ClassifyResult,
    EvalSpec,
    ExtractionMode,
    ExtractionSpec,
    GridSpec,
    ModelSpec,
    Point,
    ProviderSpec,
    RunConfig,
    RunRecord,
    RunStatus,
    Usage,
)

N_POINTS = 120


class FakeAdapter(Adapter):
    kind = "fake"

    def __init__(self, provider, model, *, refuse=("max",), version=None):
        super().__init__(provider, model)
        self.refuse = set(refuse)
        self.version = version
        self.calls = 0
        self.closed = False

    async def classify(self, prompt, system_prompt, params):
        self.calls += 1
        return ClassifyResult(
            texts=["Land"] * params.n,
            usage=Usage(50 * params.n, 3 * params.n, 0),
            latency_s=0.01,
            finish_reasons=["stop"] * params.n,
            resolved_model=self.version,
        )

    def map_config(self, config, mode):
        if config.effort in self.refuse:
            raise UnsupportedConfigError(f"effort {config.effort} unsupported")
        return {"effort": config.effort, "mode": mode.value}

    async def aclose(self):
        self.closed = True


class FakeRegistry:
    def __init__(self, providers, models):
        self.providers = {p.id: p for p in providers}
        self.models = {m.id: m for m in models}

    def model(self, ref):
        return self.models[ref.split("/")[-1]]

    def provider_of(self, model):
        return self.providers[model.provider]


class FakeLimiter:
    def __init__(self, concurrency, rpm=None, tpm=None):
        self.concurrency = concurrency

    def slot(self, est_tokens):
        return contextlib.nullcontext()


async def fake_cwr(fn, *, max_attempts=5, limiter=None):
    return await fn()


PROV = ProviderSpec(id="prov", kind="openai_compatible", base_url="http://x")
M_LP = ModelSpec(id="lp", provider="prov", name="lp-name")
M_PLAIN = ModelSpec(id="plain", provider="prov", name="plain-name")
M_FORCED = ModelSpec(id="forced", provider="prov", name="forced-name", forced_thinking=True)


@pytest.fixture
def env(monkeypatch, tmp_path):
    adapters: dict[str, FakeAdapter] = {}

    def build_adapter(provider, model):
        a = FakeAdapter(provider, model)
        adapters[model.id] = a
        return a

    points = [Point(i, 80.0 - i, float(i), math.cos(math.radians(80.0 - i))) for i in range(N_POINTS)]
    monkeypatch.setattr(planner, "build_adapter", build_adapter)
    monkeypatch.setattr(planner, "load_mask", lambda spec, cache_dir=None: SimpleNamespace(data=None, source="fake", hash="mh"))
    monkeypatch.setattr(planner, "make_grid", lambda grid: list(points))
    monkeypatch.setattr(planner, "cell_truth", lambda mask, pts, grid, rule: np.zeros(len(pts), dtype=np.int8))
    monkeypatch.setattr(planner, "render_prompt", lambda prompt, lat, lon, fmt: f"{lat},{lon}")
    monkeypatch.setattr(planner, "stratified_order", lambda pts, seed: list(reversed(pts)))
    monkeypatch.setattr(planner, "call_with_retries", fake_cwr)
    monkeypatch.setattr(planner, "ProviderLimiter", FakeLimiter)
    monkeypatch.setattr(
        planner, "cost_usd", lambda model, provider, usage, batch=False: usage.input_tokens * 1e-6 + usage.output_tokens * 5e-6
    )
    monkeypatch.setattr(
        dbmod, "_spec_hash", lambda spec, mh: hashlib.sha256(json.dumps([spec.to_dict(), mh], sort_keys=True).encode()).hexdigest()
    )
    store = Store(tmp_path / "p.db")
    registry = FakeRegistry([PROV], [M_LP, M_PLAIN, M_FORCED])
    store.save_capabilities("prov", "lp-name", Capabilities(logprobs=True, top_logprobs_max=5))
    yield SimpleNamespace(store=store, registry=registry, adapters=adapters)
    store.close()


def eval_file(matrix, *, mode=ExtractionMode.AUTO, budget=None, n_samples=4):
    return SimpleNamespace(
        eval=EvalSpec(grid=GridSpec(step_deg=4.0)),
        extraction=ExtractionSpec(mode=mode, n_samples=n_samples),
        matrix=[SimpleNamespace(model_ref=ref, configs=cfgs) for ref, cfgs in matrix],
        budget_usd=budget,
        name="t",
    )


async def test_auto_mode_resolution(env):
    ef = eval_file([("prov/lp", [RunConfig(), RunConfig(effort="low")]), ("plain", [RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    modes = [(c.variant, c.mode) for c in plan.cells]
    assert modes == [
        ("lp", ExtractionMode.LOGPROBS),
        ("lp@effort=low", ExtractionMode.SAMPLE),  # thinking runs always sample
        ("plain", ExtractionMode.SAMPLE),  # no stored logprobs capability
    ]
    assert plan.cells[0].config.top_logprobs == 5  # capped by the probe
    assert plan.mixed_modes and any("mixed extraction modes" in w for w in plan.warnings)
    assert all(a.calls == 0 for a in env.adapters.values())
    assert all(a.closed for a in env.adapters.values())


async def test_refusal_is_recorded_not_dropped(env):
    ef = eval_file([("plain", [RunConfig(effort="max"), RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    assert plan.cells[0].refused == "effort max unsupported"
    assert plan.cells[0].n_calls == 0 and plan.cells[0].est_cost_usd is None
    assert plan.cells[1].refused is None
    assert any("refused" in w for w in plan.warnings)


async def test_pilot_measures_tokens_per_point(env):
    ef = eval_file([("plain", [RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, pilot_points=50)
    cell = plan.cells[0]
    assert env.adapters["plain"].calls == 50
    assert cell.est_source == "pilot"
    assert cell.usage_per_point == Usage(200, 12, 0)
    assert cell.n_points == N_POINTS and cell.n_calls == N_POINTS * 4
    assert cell.est_usage == Usage(200 * N_POINTS, 12 * N_POINTS, 0)
    assert cell.est_cost_usd == pytest.approx(200 * N_POINTS * 1e-6 + 12 * N_POINTS * 5e-6)
    assert plan.total_cost_usd == pytest.approx(cell.est_cost_usd)
    assert len(cell.pilot_results) == 50
    # Pilot points are the first points of the run order (reversed in this fake).
    assert set(cell.pilot_results) == set(range(N_POINTS - 50, N_POINTS))


async def test_no_pilot_spends_nothing(env):
    ef = eval_file([("plain", [RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    assert env.adapters["plain"].calls == 0
    assert plan.cells[0].est_source == "heuristic"
    assert plan.cells[0].est_usage.input_tokens > 0


async def test_thinking_raises_max_output_tokens(env):
    ef = eval_file([("plain", [RunConfig(effort="low", max_output_tokens=10)])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    assert plan.cells[0].config.max_output_tokens == THINKING_MIN_OUTPUT_TOKENS["low"]
    assert plan.cells[0].thinking


async def test_forced_thinking_marked(env):
    ef = eval_file([("forced", [RunConfig()]), ("plain", [RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    forced = plan.cells[0]
    assert forced.forced_thinking and forced.label == "forced*" and forced.mode == ExtractionMode.SAMPLE
    assert any("forced-thinking" in w for w in plan.warnings)


async def test_repeats_get_distinct_hashes(env):
    ef = eval_file([("plain", [RunConfig(temperature=1.0, repeats=3)])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    assert [c.repeat_idx for c in plan.cells] == [0, 1, 2]
    assert len({c.run_hash for c in plan.cells}) == 3


async def test_duplicate_cells_refused(env):
    ef = eval_file([("plain", [RunConfig()]), ("plain", [RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    assert plan.cells[1].refused and "duplicate" in plan.cells[1].refused


def _complete_run(store, plan, cell, status=RunStatus.COMPLETE, rid="cached-run"):
    rec = RunRecord(
        id=rid,
        run_hash=cell.run_hash,
        spec_id=plan.spec_id,
        model_id=store.upsert_model(cell.model, cell.provider),
        config_id=store.upsert_config(cell.config, cell.native_params),
        variant=cell.variant,
        extraction_mode=cell.mode,
        status=status,
        n_points_total=N_POINTS,
    )
    store.create_run(rec)
    return rec


async def test_cache_hit_skips_pilot_and_cost(env):
    ef = eval_file([("plain", [RunConfig()])])
    first = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    _complete_run(env.store, first, first.cells[0])
    plan = await planner.build_plan(ef, env.registry, env.store, pilot_points=50)
    cell = plan.cells[0]
    assert cell.cached_run_id == "cached-run"
    assert cell.n_calls == 0 and cell.est_cost_usd is None
    assert env.adapters["plain"].calls == 0
    assert plan.total_cost_usd == 0.0


async def test_unfinished_run_is_resumed(env):
    ef = eval_file([("plain", [RunConfig()])])
    first = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    _complete_run(env.store, first, first.cells[0], status=RunStatus.PAUSED, rid="paused-run")
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    assert plan.cells[0].resume_run_id == "paused-run"
    assert plan.cells[0].cached_run_id is None


async def test_check_comparison_rejects_mixed_specs(env):
    ef = eval_file([("plain", [RunConfig()])])
    plan = await planner.build_plan(ef, env.registry, env.store, run_pilot=False)
    a = _complete_run(env.store, plan, plan.cells[0], rid="a")
    other_spec = env.store.upsert_eval_spec(EvalSpec(coord_format="dms"), "mh", "fake")
    b = RunRecord(
        id="b", run_hash="zz", spec_id=other_spec, model_id=a.model_id, config_id=a.config_id,
        variant="plain", extraction_mode=ExtractionMode.LOGPROBS, status=RunStatus.COMPLETE,
    )
    env.store.create_run(b)
    with pytest.raises(ValueError):
        planner.check_comparison(env.store, ["a", "b"])
    info = planner.check_comparison(env.store, ["a"])
    assert info["spec_id"] == plan.spec_id and not info["mixed_extraction_modes"]
