"""SQLite results store.

One file, WAL mode, foreign keys on. Runs are immutable once complete (enforced by triggers
in schema.sql and by checks here). API keys are never written: only provider id/kind/base_url
are stored, and model `extra` / native params are scrubbed of secret-looking keys.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from blindearth.types import (
    Capabilities,
    EvalSpec,
    ExtractionMode,
    GridSpec,
    MaskSpec,
    ModelSpec,
    PointResult,
    PromptSpec,
    ProviderSpec,
    RunConfig,
    RunRecord,
    RunStatus,
    Usage,
)

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_VERSION = "1"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

POINT_COLUMNS = [
    "idx",
    "lat",
    "lon",
    "weight",
    "truth",
    "p_land",
    "n_valid",
    "n_samples",
    "validity_mass",
    "answer_text",
    "finish_reason",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "error",
]
_NUMERIC_POINT_COLUMNS = [
    "lat",
    "lon",
    "weight",
    "truth",
    "p_land",
    "n_valid",
    "n_samples",
    "validity_mass",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
]

# RunRecord fields stored 1:1 in runs columns.
_RUN_SIMPLE_FIELDS = {
    "run_hash",
    "spec_id",
    "model_id",
    "config_id",
    "variant",
    "extraction_mode",
    "status",
    "started_at",
    "ended_at",
    "runner_version",
    "seed",
    "repeat_idx",
    "resolved_model_version",
    "forced_thinking",
    "thinking",
    "cost_usd",
    "n_points_total",
    "n_points_done",
}
# Extra run columns (not on RunRecord) that update_run accepts. JSON ones are serialized.
_RUN_EXTRA_FIELDS = {"plan_hash", "note"}
_RUN_EXTRA_JSON_FIELDS = {"extraction": "extraction_json", "batch_state": "batch_state_json"}

_SECRET_MARKERS = ("api_key", "apikey", "secret", "password", "authorization", "token_value", "bearer")


class RunImmutableError(RuntimeError):
    """Raised when something tries to modify a complete run."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(obj: Any) -> str:
    """JSON with dataclass/enum/numpy support; NaN is allowed (round-trips through json.loads)."""
    return json.dumps(obj, default=_json_default, sort_keys=True)


def scrub_secrets(obj: Any) -> Any:
    """Drop dict keys that look like credentials, recursively."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(m in lk for m in _SECRET_MARKERS):
                continue
            out[k] = scrub_secrets(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [scrub_secrets(v) for v in obj]
    return obj


def _spec_hash(spec: EvalSpec, mask_hash: str) -> str:
    # Indirection so tests can patch it without package B.
    from blindearth.evalspec.load import spec_hash

    return spec_hash(spec, mask_hash)


def eval_spec_from_dict(d: dict[str, Any]) -> EvalSpec:
    try:
        from blindearth.evalspec.load import evalspec_from_dict
    except ImportError:  # pragma: no cover - package B always ships it
        pass
    else:
        return evalspec_from_dict(d)
    return EvalSpec(
        mask=MaskSpec(**d.get("mask", {})),
        grid=GridSpec(**d.get("grid", {})),
        coord_format=d.get("coord_format", "hemisphere"),
        prompt=PromptSpec(**d.get("prompt", {})),
    )


def model_row_id(model: ModelSpec, provider: ProviderSpec) -> str:
    key = json.dumps([provider.id, model.id, model.name, model.quant])
    return f"{provider.id}/{model.id}#{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def config_row_id(config: RunConfig, native: dict) -> str:
    payload = dumps({"normalized": config.normalized(), "native": scrub_secrets(native)})
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def run_config_from_dict(d: dict[str, Any]) -> RunConfig:
    names = {f.name for f in dataclasses.fields(RunConfig)}
    return RunConfig(**{k: v for k, v in d.items() if k in names})


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:" and not self.path.startswith("file:"):
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    # ------------------------------------------------------------------ plumbing

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ eval specs

    def upsert_eval_spec(self, spec: EvalSpec, mask_hash: str, mask_source: str) -> str:
        spec_id = _spec_hash(spec, mask_hash)
        with self._tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO eval_specs
                   (id, mask_hash, mask_source, mask_id, truth_rule, step_deg, placement,
                    subset_frac, grid_seed, coord_format, prompt_id, prompt_template,
                    system_prompt, spec_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    spec_id,
                    mask_hash,
                    mask_source,
                    spec.mask.id,
                    spec.mask.truth_rule,
                    float(spec.grid.step_deg),
                    spec.grid.placement,
                    spec.grid.subset_frac,
                    int(spec.grid.seed),
                    spec.coord_format,
                    spec.prompt.id,
                    spec.prompt.template,
                    spec.prompt.system_prompt,
                    dumps(spec.to_dict()),
                    utcnow(),
                ),
            )
        return spec_id

    def get_eval_spec(self, spec_id: str) -> tuple[EvalSpec, str, str]:
        r = self._one("SELECT spec_json, mask_hash, mask_source FROM eval_specs WHERE id=?", (spec_id,))
        if r is None:
            raise KeyError(f"eval spec not found: {spec_id}")
        return eval_spec_from_dict(json.loads(r["spec_json"])), r["mask_hash"], r["mask_source"]

    # ------------------------------------------------------------------ models / configs

    def upsert_model(self, model: ModelSpec, provider: ProviderSpec) -> str:
        mid = model_row_id(model, provider)
        now = utcnow()
        with self._tx() as c:
            c.execute(
                """INSERT INTO models (id, provider_id, provider_kind, registry_id, name, quant,
                       vendor, release_date, forced_thinking, base_url, extra_json,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       provider_kind=excluded.provider_kind, vendor=excluded.vendor,
                       release_date=excluded.release_date,
                       forced_thinking=excluded.forced_thinking, base_url=excluded.base_url,
                       extra_json=excluded.extra_json, updated_at=excluded.updated_at""",
                (
                    mid,
                    provider.id,
                    provider.kind,
                    model.id,
                    model.name,
                    model.quant,
                    model.vendor,
                    model.release_date,
                    int(bool(model.forced_thinking)),
                    provider.base_url,
                    dumps(scrub_secrets(model.extra or {})),
                    now,
                    now,
                ),
            )
        return mid

    def get_model(self, model_id: str) -> ModelSpec:
        r = self._one("SELECT * FROM models WHERE id=?", (model_id,))
        if r is None:
            raise KeyError(f"model not found: {model_id}")
        return ModelSpec(
            id=r["registry_id"],
            provider=r["provider_id"],
            name=r["name"],
            quant=r["quant"],
            vendor=r["vendor"],
            release_date=r["release_date"],
            forced_thinking=bool(r["forced_thinking"]),
            extra=json.loads(r["extra_json"] or "{}"),
        )

    def model_row(self, model_id: str) -> dict[str, Any]:
        r = self._one("SELECT * FROM models WHERE id=?", (model_id,))
        if r is None:
            raise KeyError(f"model not found: {model_id}")
        return dict(r)

    def set_model_resolved_version(self, model_id: str, version: str | None) -> None:
        if not version:
            return
        with self._tx() as c:
            c.execute(
                "UPDATE models SET resolved_version=?, updated_at=? WHERE id=?",
                (version, utcnow(), model_id),
            )

    def known_resolved_version(self, model: ModelSpec, provider: ProviderSpec) -> str | None:
        """Last version the API reported for this model, from the models row or its runs."""
        mid = model_row_id(model, provider)
        r = self._one("SELECT resolved_version FROM models WHERE id=?", (mid,))
        if r is not None and r["resolved_version"]:
            return r["resolved_version"]
        r = self._one(
            """SELECT resolved_model_version FROM runs
               WHERE model_id=? AND resolved_model_version IS NOT NULL
               ORDER BY COALESCE(ended_at, started_at, created_at) DESC LIMIT 1""",
            (mid,),
        )
        return r["resolved_model_version"] if r is not None else None

    def upsert_config(self, config: RunConfig, native: dict) -> str:
        native = scrub_secrets(native or {})
        cid = config_row_id(config, native)
        with self._tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO configs (id, normalized_json, native_json, config_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (cid, dumps(config.normalized()), dumps(native), dumps(dataclasses.asdict(config)), utcnow()),
            )
        return cid

    def get_config(self, config_id: str) -> tuple[RunConfig, dict]:
        r = self._one("SELECT config_json, native_json FROM configs WHERE id=?", (config_id,))
        if r is None:
            raise KeyError(f"config not found: {config_id}")
        return run_config_from_dict(json.loads(r["config_json"])), json.loads(r["native_json"])

    # ------------------------------------------------------------------ runs

    @staticmethod
    def _row_to_run(r: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=r["id"],
            run_hash=r["run_hash"],
            spec_id=r["spec_id"],
            model_id=r["model_id"],
            config_id=r["config_id"],
            variant=r["variant"],
            extraction_mode=ExtractionMode(r["extraction_mode"]),
            status=RunStatus(r["status"]),
            started_at=r["started_at"],
            ended_at=r["ended_at"],
            runner_version=r["runner_version"] or "",
            seed=int(r["seed"]),
            repeat_idx=int(r["repeat_idx"]),
            resolved_model_version=r["resolved_model_version"],
            forced_thinking=bool(r["forced_thinking"]),
            thinking=bool(r["thinking"]),
            totals=Usage(int(r["input_tokens"]), int(r["output_tokens"]), int(r["thinking_tokens"])),
            cost_usd=r["cost_usd"],
            n_points_total=int(r["n_points_total"]),
            n_points_done=int(r["n_points_done"]),
        )

    def create_run(self, run: RunRecord) -> None:
        if run.extraction_mode == ExtractionMode.AUTO:
            raise ValueError("run.extraction_mode must be resolved (not AUTO)")
        with self._tx() as c:
            c.execute(
                """INSERT INTO runs (id, run_hash, plan_hash, spec_id, model_id, config_id, variant,
                       extraction_mode, status, created_at, started_at, ended_at, runner_version,
                       seed, repeat_idx, resolved_model_version, forced_thinking, thinking,
                       input_tokens, output_tokens, thinking_tokens, cost_usd,
                       n_points_total, n_points_done)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run.id,
                    run.run_hash,
                    run.run_hash,
                    run.spec_id,
                    run.model_id,
                    run.config_id,
                    run.variant,
                    ExtractionMode(run.extraction_mode).value,
                    RunStatus(run.status).value,
                    utcnow(),
                    run.started_at,
                    run.ended_at,
                    run.runner_version,
                    int(run.seed),
                    int(run.repeat_idx),
                    run.resolved_model_version,
                    int(bool(run.forced_thinking)),
                    int(bool(run.thinking)),
                    run.totals.input_tokens,
                    run.totals.output_tokens,
                    run.totals.thinking_tokens,
                    run.cost_usd,
                    run.n_points_total,
                    run.n_points_done,
                ),
            )

    def resolve_run_id(self, id_or_prefix: str) -> str:
        r = self._one("SELECT id FROM runs WHERE id=?", (id_or_prefix,))
        if r is not None:
            return r["id"]
        if len(id_or_prefix) >= 4:
            rows = self._all("SELECT id FROM runs WHERE id LIKE ? LIMIT 2", (id_or_prefix + "%",))
            if len(rows) == 1:
                return rows[0]["id"]
            if len(rows) > 1:
                raise KeyError(f"run id prefix is ambiguous: {id_or_prefix}")
        raise KeyError(f"run not found: {id_or_prefix}")

    def get_run(self, run_id: str) -> RunRecord:
        rid = self.resolve_run_id(run_id)
        r = self._one("SELECT * FROM runs WHERE id=?", (rid,))
        assert r is not None
        return self._row_to_run(r)

    def run_meta(self, run_id: str) -> dict[str, Any]:
        """Run columns that are not on RunRecord: plan_hash, note, extraction, batch_state, created_at."""
        rid = self.resolve_run_id(run_id)
        r = self._one(
            "SELECT plan_hash, note, extraction_json, batch_state_json, created_at FROM runs WHERE id=?",
            (rid,),
        )
        assert r is not None
        return {
            "plan_hash": r["plan_hash"],
            "note": r["note"],
            "extraction": json.loads(r["extraction_json"] or "{}"),
            "batch_state": json.loads(r["batch_state_json"]) if r["batch_state_json"] else None,
            "created_at": r["created_at"],
        }

    def find_complete_run(self, run_hash: str) -> RunRecord | None:
        r = self._one(
            """SELECT * FROM runs WHERE run_hash=? AND status='complete'
               ORDER BY ended_at DESC, created_at DESC LIMIT 1""",
            (run_hash,),
        )
        return self._row_to_run(r) if r is not None else None

    def find_resumable_run(self, run_hash: str) -> RunRecord | None:
        """Latest unfinished run (queued/running/paused/failed) planned under this hash."""
        r = self._one(
            """SELECT * FROM runs WHERE (run_hash=? OR plan_hash=?) AND status!='complete'
               ORDER BY created_at DESC LIMIT 1""",
            (run_hash, run_hash),
        )
        return self._row_to_run(r) if r is not None else None

    def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        model_id: str | None = None,
        spec_id: str | None = None,
    ) -> list[RunRecord]:
        sql = "SELECT runs.* FROM runs JOIN models ON models.id = runs.model_id WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            sql += " AND runs.status=?"
            params.append(RunStatus(status).value)
        if model_id is not None:
            # Accept the store id, "provider/model_id" or the bare registry id.
            sql += (
                " AND (runs.model_id=? OR (models.provider_id || '/' || models.registry_id)=?"
                " OR models.registry_id=?)"
            )
            params += [model_id, model_id, model_id]
        if spec_id is not None:
            sql += " AND runs.spec_id=?"
            params.append(spec_id)
        sql += " ORDER BY runs.created_at DESC, runs.id"
        return [self._row_to_run(r) for r in self._all(sql, params)]

    def _assert_mutable(self, c: sqlite3.Connection, run_id: str) -> None:
        r = c.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            raise KeyError(f"run not found: {run_id}")
        if r["status"] == RunStatus.COMPLETE.value:
            raise RunImmutableError(f"run {run_id} is complete and immutable")

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets: list[str] = []
        params: list[Any] = []
        for k, v in fields.items():
            if k == "totals":
                u: Usage = v
                sets += ["input_tokens=?", "output_tokens=?", "thinking_tokens=?"]
                params += [u.input_tokens, u.output_tokens, u.thinking_tokens]
            elif k in _RUN_SIMPLE_FIELDS or k in _RUN_EXTRA_FIELDS:
                if isinstance(v, Enum):
                    v = v.value
                elif isinstance(v, bool):
                    v = int(v)
                sets.append(f"{k}=?")
                params.append(v)
            elif k in _RUN_EXTRA_JSON_FIELDS:
                sets.append(f"{_RUN_EXTRA_JSON_FIELDS[k]}=?")
                params.append(None if v is None else dumps(v))
            else:
                raise ValueError(f"unknown run field: {k}")
        if fields.get("extraction_mode") in (ExtractionMode.AUTO, "auto"):
            raise ValueError("extraction_mode must be resolved (not AUTO)")
        with self._tx() as c:
            self._assert_mutable(c, run_id)
            c.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", (*params, run_id))

    # ------------------------------------------------------------------ points

    def write_points(self, points: list[PointResult]) -> None:
        if not points:
            return
        now = utcnow()
        rows = []
        for p in points:
            rows.append(
                (
                    p.run_id,
                    int(p.idx),
                    float(p.lat),
                    float(p.lon),
                    math.cos(math.radians(float(p.lat))),
                    None if p.truth is None else int(p.truth),
                    None if p.p_land is None else float(p.p_land),
                    int(p.n_valid),
                    int(p.n_samples),
                    None if p.validity_mass is None else float(p.validity_mass),
                    p.answer_text or "",
                    p.finish_reason,
                    float(p.latency_s or 0.0),
                    int(p.usage.input_tokens),
                    int(p.usage.output_tokens),
                    int(p.usage.thinking_tokens),
                    p.thinking_text,
                    p.error,
                    now,
                )
            )
        with self._tx() as c:
            for rid in {p.run_id for p in points}:
                self._assert_mutable(c, rid)
            c.executemany(
                """INSERT INTO points (run_id, idx, lat, lon, weight, truth, p_land, n_valid,
                       n_samples, validity_mass, answer_text, finish_reason, latency_s,
                       input_tokens, output_tokens, thinking_tokens, thinking_text, error,
                       updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id, idx) DO UPDATE SET
                       lat=excluded.lat, lon=excluded.lon, weight=excluded.weight,
                       truth=excluded.truth, p_land=excluded.p_land, n_valid=excluded.n_valid,
                       n_samples=excluded.n_samples, validity_mass=excluded.validity_mass,
                       answer_text=excluded.answer_text, finish_reason=excluded.finish_reason,
                       latency_s=excluded.latency_s, input_tokens=excluded.input_tokens,
                       output_tokens=excluded.output_tokens,
                       thinking_tokens=excluded.thinking_tokens,
                       thinking_text=excluded.thinking_text, error=excluded.error,
                       attempts=points.attempts + 1, updated_at=excluded.updated_at""",
                rows,
            )

    def done_indices(self, run_id: str) -> set[int]:
        rows = self._all("SELECT idx FROM points WHERE run_id=? AND error IS NULL", (run_id,))
        return {int(r["idx"]) for r in rows}

    def failed_indices(self, run_id: str) -> set[int]:
        rows = self._all("SELECT idx FROM points WHERE run_id=? AND error IS NOT NULL", (run_id,))
        return {int(r["idx"]) for r in rows}

    def count_points(self, run_id: str) -> tuple[int, int]:
        """(done, failed)."""
        r = self._one(
            """SELECT SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END) AS done,
                      SUM(CASE WHEN error IS NULL THEN 0 ELSE 1 END) AS failed
               FROM points WHERE run_id=?""",
            (run_id,),
        )
        return int(r["done"] or 0), int(r["failed"] or 0)

    def load_points(self, run_id: str) -> pd.DataFrame:
        import pandas as pd

        rid = self.resolve_run_id(run_id)
        with self._lock:
            df = pd.read_sql_query(
                f"SELECT {', '.join(POINT_COLUMNS)} FROM points WHERE run_id=? ORDER BY idx",
                self._conn,
                params=(rid,),
            )
        for col in _NUMERIC_POINT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["idx"] = df["idx"].astype("int64")
        return df

    def load_thinking_texts(self, run_id: str) -> dict[int, str]:
        rows = self._all(
            "SELECT idx, thinking_text FROM points WHERE run_id=? AND thinking_text IS NOT NULL",
            (run_id,),
        )
        return {int(r["idx"]): r["thinking_text"] for r in rows}

    # ------------------------------------------------------------------ metrics

    def save_metrics(self, run_id: str, mask_hash: str, threshold: float, metrics: dict) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO metrics (run_id, mask_hash, threshold, metrics_json, computed_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(run_id, mask_hash, threshold) DO UPDATE SET
                       metrics_json=excluded.metrics_json, computed_at=excluded.computed_at""",
                (run_id, mask_hash, float(threshold), dumps(metrics), utcnow()),
            )

    def load_metrics(
        self, run_id: str, mask_hash: str | None = None, threshold: float | None = None
    ) -> dict | None:
        sql = "SELECT metrics_json FROM metrics WHERE run_id=?"
        params: list[Any] = [run_id]
        if mask_hash is not None:
            sql += " AND mask_hash=?"
            params.append(mask_hash)
        if threshold is not None:
            sql += " AND abs(threshold - ?) < 1e-9"
            params.append(float(threshold))
        sql += " ORDER BY computed_at DESC, id DESC LIMIT 1"
        r = self._one(sql, params)
        return json.loads(r["metrics_json"]) if r is not None else None

    def list_metrics(self, run_id: str) -> list[dict]:
        rows = self._all(
            "SELECT mask_hash, threshold, metrics_json, computed_at FROM metrics WHERE run_id=? ORDER BY computed_at",
            (run_id,),
        )
        return [
            {
                "mask_hash": r["mask_hash"],
                "threshold": r["threshold"],
                "computed_at": r["computed_at"],
                "metrics": json.loads(r["metrics_json"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ comparisons

    def create_comparison(
        self,
        name: str,
        run_ids: list[str],
        filters: dict | None = None,
        ordering: list[str] | None = None,
    ) -> str:
        """Create a named comparison. Re-using a name replaces that comparison's run set."""
        ids = [self.resolve_run_id(r) for r in run_ids]
        now = utcnow()
        with self._tx() as c:
            existing = c.execute("SELECT id FROM comparisons WHERE name=?", (name,)).fetchone()
            if existing is not None:
                cid = existing["id"]
                c.execute(
                    """UPDATE comparisons SET run_ids_json=?, filters_json=?, ordering_json=?,
                       updated_at=? WHERE id=?""",
                    (dumps(ids), dumps(filters or {}), dumps(ordering or []), now, cid),
                )
                c.execute("DELETE FROM comparison_runs WHERE comparison_id=?", (cid,))
            else:
                import uuid

                cid = uuid.uuid4().hex
                c.execute(
                    """INSERT INTO comparisons (id, name, run_ids_json, filters_json, ordering_json,
                       created_at, updated_at) VALUES (?,?,?,?,?,?,?)""",
                    (cid, name, dumps(ids), dumps(filters or {}), dumps(ordering or []), now, now),
                )
            c.executemany(
                "INSERT OR IGNORE INTO comparison_runs (comparison_id, run_id, position) VALUES (?,?,?)",
                [(cid, rid, i) for i, rid in enumerate(ids)],
            )
        return cid

    @staticmethod
    def _row_to_comparison(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "name": r["name"],
            "run_ids": json.loads(r["run_ids_json"]),
            "filters": json.loads(r["filters_json"] or "{}"),
            "ordering": json.loads(r["ordering_json"] or "[]"),
            "created_at": r["created_at"],
        }

    def get_comparison(self, comparison_id_or_name: str) -> dict:
        r = self._one("SELECT * FROM comparisons WHERE id=?", (comparison_id_or_name,))
        if r is None:
            r = self._one("SELECT * FROM comparisons WHERE name=?", (comparison_id_or_name,))
        if r is None:
            raise KeyError(f"comparison not found: {comparison_id_or_name}")
        return self._row_to_comparison(r)

    def list_comparisons(self) -> list[dict]:
        return [self._row_to_comparison(r) for r in self._all("SELECT * FROM comparisons ORDER BY created_at DESC")]

    # ------------------------------------------------------------------ capabilities

    def save_capabilities(self, provider_id: str, model_name: str, caps: Capabilities) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO capabilities (provider_id, model_name, caps_json, probed_at, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(provider_id, model_name) DO UPDATE SET
                       caps_json=excluded.caps_json, probed_at=excluded.probed_at,
                       updated_at=excluded.updated_at""",
                (provider_id, model_name, dumps(dataclasses.asdict(caps)), caps.probed_at, utcnow()),
            )

    def load_capabilities(self, provider_id: str, model_name: str) -> Capabilities | None:
        r = self._one(
            "SELECT caps_json FROM capabilities WHERE provider_id=? AND model_name=?",
            (provider_id, model_name),
        )
        if r is None:
            return None
        d = json.loads(r["caps_json"])
        names = {f.name for f in dataclasses.fields(Capabilities)}
        return Capabilities(**{k: v for k, v in d.items() if k in names})
