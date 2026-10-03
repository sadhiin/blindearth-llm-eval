from __future__ import annotations

import hashlib
import json
import math
import sqlite3

import pytest

from blindearth.store import db as dbmod
from blindearth.store.db import RunImmutableError, Store
from blindearth.store.export import export_metrics, export_points
from blindearth.types import (
    Capabilities,
    EvalSpec,
    ExtractionMode,
    GridSpec,
    ModelSpec,
    PointResult,
    ProviderSpec,
    RunConfig,
    RunRecord,
    RunStatus,
    Usage,
)


@pytest.fixture(autouse=True)
def _fake_spec_hash(monkeypatch):
    def fake(spec, mask_hash):
        return hashlib.sha256(json.dumps([spec.to_dict(), mask_hash], sort_keys=True).encode()).hexdigest()

    monkeypatch.setattr(dbmod, "_spec_hash", fake)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


PROVIDER = ProviderSpec(id="anthropic", kind="anthropic", api_key_env="ANTHROPIC_API_KEY")
MODEL = ModelSpec(id="opus-5-5", provider="anthropic", name="claude-opus-5-5", extra={"api_key": "sk-secret-123", "note": "ok"})


def make_run(store: Store, *, run_hash="h1", status=RunStatus.QUEUED, rid=None) -> RunRecord:
    spec_id = store.upsert_eval_spec(EvalSpec(grid=GridSpec(step_deg=4.0)), "maskhash", "Natural Earth")
    model_id = store.upsert_model(MODEL, PROVIDER)
    config_id = store.upsert_config(RunConfig(effort="low"), {"thinking": {"type": "enabled"}, "api_key": "sk-secret-123"})
    rec = RunRecord(
        id=rid or f"run{run_hash}{status.value}",
        run_hash=run_hash,
        spec_id=spec_id,
        model_id=model_id,
        config_id=config_id,
        variant="opus-5-5@effort=low",
        extraction_mode=ExtractionMode.SAMPLE,
        status=status,
        n_points_total=3,
    )
    store.create_run(rec)
    return rec


def pt(run_id, idx, *, error=None, p=1.0, lat=10.0):
    return PointResult(
        run_id=run_id, idx=idx, lat=lat, lon=20.0, truth=1, p_land=None if error else p,
        n_valid=0 if error else 4, n_samples=4, validity_mass=None, answer_text="Land",
        finish_reason="stop", latency_s=0.5, usage=Usage(50, 2, 0), error=error,
    )


def test_pragmas_and_tables(store):
    c = store._conn
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("eval_specs", "models", "configs", "runs", "points", "metrics", "comparisons", "capabilities"):
        assert t in tables
    idx = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "points_run_idx" in idx


def test_eval_spec_roundtrip(store):
    spec = EvalSpec(grid=GridSpec(step_deg=4.0, seed=3), coord_format="signed_decimal")
    sid = store.upsert_eval_spec(spec, "mh", "src")
    assert store.upsert_eval_spec(spec, "mh", "src") == sid
    got, mh, src = store.get_eval_spec(sid)
    assert got == spec and mh == "mh" and src == "src"


def test_model_and_config_never_store_keys(store, tmp_path):
    make_run(store)
    model = store.get_model(store.upsert_model(MODEL, PROVIDER))
    assert model.name == "claude-opus-5-5" and "api_key" not in model.extra and model.extra["note"] == "ok"
    cfg, native = store.get_config(store.upsert_config(RunConfig(effort="low"), {"api_key": "sk-secret-123", "x": 1}))
    assert cfg.effort == "low" and native == {"x": 1}
    store.close()
    raw = (tmp_path / "t.db").read_bytes()
    for f in tmp_path.glob("t.db*"):
        raw += f.read_bytes()
    assert b"sk-secret-123" not in raw


def test_run_roundtrip_prefix_and_listing(store):
    rec = make_run(store, rid="abcdef123456")
    got = store.get_run("abcdef")
    assert got.id == rec.id and got.extraction_mode == ExtractionMode.SAMPLE and got.status == RunStatus.QUEUED
    assert [r.id for r in store.list_runs(model_id="anthropic/opus-5-5")] == [rec.id]
    assert store.list_runs(status=RunStatus.COMPLETE) == []
    store.update_run(rec.id, status=RunStatus.RUNNING, totals=Usage(10, 2, 1), cost_usd=0.5, note="x", extraction={"mode": "sample"})
    got = store.get_run(rec.id)
    assert got.status == RunStatus.RUNNING and got.totals == Usage(10, 2, 1) and got.cost_usd == 0.5
    assert store.run_meta(rec.id)["extraction"] == {"mode": "sample"}
    with pytest.raises(ValueError):
        store.update_run(rec.id, nonsense=1)


def test_points_upsert_done_failed_and_load(store):
    rec = make_run(store)
    store.write_points([pt(rec.id, 0), pt(rec.id, 1, error="boom"), pt(rec.id, 2, lat=60.0)])
    assert store.done_indices(rec.id) == {0, 2}
    assert store.failed_indices(rec.id) == {1}
    store.write_points([pt(rec.id, 1)])  # retry succeeded
    assert store.failed_indices(rec.id) == set()
    df = store.load_points(rec.id)
    assert list(df.columns) == dbmod.POINT_COLUMNS
    assert list(df["idx"]) == [0, 1, 2]
    assert df.loc[df["idx"] == 2, "weight"].iloc[0] == pytest.approx(math.cos(math.radians(60.0)))


def test_complete_runs_are_immutable(store):
    rec = make_run(store)
    store.write_points([pt(rec.id, 0)])
    store.update_run(rec.id, status=RunStatus.COMPLETE, ended_at="2026-10-03T00:00:00+00:00")
    with pytest.raises(RunImmutableError):
        store.update_run(rec.id, note="changed")
    with pytest.raises(RunImmutableError):
        store.write_points([pt(rec.id, 1)])
    # The database refuses too, even if the Python checks were bypassed.
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("UPDATE runs SET note='x' WHERE id=?", (rec.id,))
    assert store.find_complete_run("h1").id == rec.id


def test_find_resumable(store):
    rec = make_run(store, run_hash="hx", status=RunStatus.PAUSED)
    assert store.find_complete_run("hx") is None
    assert store.find_resumable_run("hx").id == rec.id


def test_metrics(store):
    rec = make_run(store)
    store.save_metrics(rec.id, "mh", 0.5, {"acc_area": 0.9, "nan": float("nan")})
    store.save_metrics(rec.id, "mh", 0.7, {"acc_area": 0.8})
    assert store.load_metrics(rec.id, "mh", 0.5)["acc_area"] == 0.9
    assert store.load_metrics(rec.id, threshold=0.7)["acc_area"] == 0.8
    assert store.load_metrics(rec.id, "other") is None
    store.save_metrics(rec.id, "mh", 0.5, {"acc_area": 0.95})
    assert store.load_metrics(rec.id, "mh", 0.5)["acc_area"] == 0.95


def test_comparisons(store):
    a = make_run(store, run_hash="a")
    b = make_run(store, run_hash="b")
    cid = store.create_comparison("cmp", [a.id, b.id], filters={"x": 1}, ordering=["acc_area"])
    got = store.get_comparison("cmp")
    assert got["id"] == cid and got["run_ids"] == [a.id, b.id] and got["filters"] == {"x": 1}
    assert store.get_comparison(cid)["name"] == "cmp"
    assert store.create_comparison("cmp", [b.id]) == cid
    assert store.get_comparison("cmp")["run_ids"] == [b.id]
    assert len(store.list_comparisons()) == 1


def test_capabilities(store):
    assert store.load_capabilities("p", "m") is None
    caps = Capabilities(logprobs=True, top_logprobs_max=5, supported_efforts=["low"], probed_at="2026-10-03T00:00:00+00:00")
    store.save_capabilities("p", "m", caps)
    assert store.load_capabilities("p", "m") == caps


def test_known_resolved_version(store):
    make_run(store)
    assert store.known_resolved_version(MODEL, PROVIDER) is None
    store.set_model_resolved_version(store.upsert_model(MODEL, PROVIDER), "claude-opus-5-5-20261001")
    assert store.known_resolved_version(MODEL, PROVIDER) == "claude-opus-5-5-20261001"


def test_export(store, tmp_path):
    rec = make_run(store)
    store.write_points([pt(rec.id, 0), pt(rec.id, 1)])
    store.save_metrics(rec.id, "mh", 0.5, {"acc_area": 1.0})
    out = export_points(store, rec.id, tmp_path / "out.csv", "csv")
    assert out.read_text().splitlines()[0].startswith("run_id,idx,lat")
    mj = json.loads(export_metrics(store, rec.id, tmp_path / "m.json").read_text())
    assert mj["metrics"][0]["metrics"]["acc_area"] == 1.0
