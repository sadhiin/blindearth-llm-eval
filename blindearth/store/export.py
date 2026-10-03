"""Export stored runs: points as Parquet/CSV, metrics as JSON."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

from blindearth.store.db import Store, dumps


def _target(path: Path, run_id: str, ext: str) -> Path:
    path = Path(path).expanduser()
    if path.is_dir() or (not path.suffix and not path.exists()):
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{run_id}.{ext}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_points(store: Store, run_id: str, path: Path, fmt: Literal["parquet", "csv"]) -> Path:
    run_id = store.resolve_run_id(run_id)
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"unsupported points format: {fmt}")
    df = store.load_points(run_id)
    df.insert(0, "run_id", run_id)
    out = _target(path, run_id, fmt)
    if fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out


def export_metrics(store: Store, run_id: str, path: Path) -> Path:
    """JSON with the run record, its provenance and every stored metrics row."""
    run_id = store.resolve_run_id(run_id)
    run = store.get_run(run_id)
    spec, mask_hash, mask_source = store.get_eval_spec(run.spec_id)
    model = store.model_row(run.model_id)
    payload = {
        "run": dataclasses.asdict(run),
        "run_meta": store.run_meta(run_id),
        "model": {k: model[k] for k in ("provider_id", "registry_id", "name", "resolved_version", "quant")},
        "eval_spec": {"id": run.spec_id, "spec": spec.to_dict(), "mask_hash": mask_hash, "mask_source": mask_source},
        "metrics": store.list_metrics(run_id),
    }
    out = _target(path, run_id, "json")
    out.write_text(dumps(payload), encoding="utf-8")
    return out
