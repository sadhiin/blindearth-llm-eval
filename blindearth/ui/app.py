"""Gradio web UI following the spec's five-step flow, plus saved comparisons.

Tabs: 1 Eval spec · 2 Scope & models · 3 Config variants · 4 Plan · 5 Run (live map) ·
Saved comparisons. Gradio is imported lazily inside `build_app`, so importing this module never
needs the `ui` extra.

This is a single-user local tool: per-session state lives in one `_AppState` held by the app.
SQLite connections are opened per thread (Gradio runs sync handlers in a thread pool).
"""

from __future__ import annotations

import asyncio
import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from blindearth.types import (
    EvalSpec,
    ExtractionMode,
    ExtractionSpec,
    GridSpec,
    MaskSpec,
    PointResult,
    PromptSpec,
    RunConfig,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover
    import gradio

    from blindearth.runner.planner import Plan

BUILTIN_MASKS = ["natural-earth-land", "gshhg", "modis-mod44w", "upload"]
SCOPES = ["Within a vendor", "Across vendors", "Mixed matrix", "Config sweep", "From saved runs"]
EFFORTS = ["default", "off", "low", "medium", "high", "max"]
VARIANT_COLS = ["model", "effort", "temperature", "n_samples", "top_logprobs",
                "max_output_tokens", "seed", "repeats"]
LIVE_REFRESH_S = 3.0


@dataclass
class _AppState:
    eval_spec: EvalSpec = field(default_factory=EvalSpec)
    extraction: ExtractionSpec = field(default_factory=ExtractionSpec)
    budget_usd: float | None = None
    name: str | None = None
    selected_models: list[str] = field(default_factory=list)
    matrix: list[tuple[str, RunConfig]] = field(default_factory=list)
    plan: "Plan | None" = None
    task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None
    run_ids: list[str] = field(default_factory=list)
    live: dict[str, dict[int, PointResult]] = field(default_factory=dict)
    base_loaded: set[str] = field(default_factory=set)
    run_state: str = "idle"
    run_error: str | None = None
    started: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


# --------------------------------------------------------------------------- pure helpers


def _none_if_blank(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, str) and x.strip() in ("", "default", "None", "nan"):
        return None
    return x


def _num(x: Any, cast=float) -> Any:
    x = _none_if_blank(x)
    if x is None:
        return None
    try:
        return cast(float(x)) if cast is int else cast(x)
    except (TypeError, ValueError):
        return None


def variants_from_table(table: Any) -> list[tuple[str, RunConfig]]:
    """Editable variants table (DataFrame or list of rows) -> [(model_ref, RunConfig)]."""
    if table is None:
        return []
    df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table, columns=VARIANT_COLS)
    out = []
    for row in df.to_dict("records"):
        ref = _none_if_blank(row.get("model"))
        if not ref:
            continue
        effort = _none_if_blank(row.get("effort"))
        cfg = RunConfig(
            effort=str(effort) if effort is not None else None,
            temperature=_num(row.get("temperature")),
            n_samples=_num(row.get("n_samples"), int),
            top_logprobs=_num(row.get("top_logprobs"), int),
            max_output_tokens=_num(row.get("max_output_tokens"), int),
            seed=_num(row.get("seed"), int),
            repeats=_num(row.get("repeats"), int) or 1,
        )
        out.append((str(ref).strip(), cfg))
    return out


def apply_to_all(models: list[str], efforts: list[str], temperature: Any, n_samples: Any,
                 max_output_tokens: Any, repeats: Any) -> list[list[Any]]:
    """One row per model × effort, the same other settings for all."""
    efforts = efforts or ["default"]
    rows = []
    for m in models:
        for e in efforts:
            rows.append([m, "" if e == "default" else e, _num(temperature), _num(n_samples, int),
                         None, _num(max_output_tokens, int), None, _num(repeats, int) or 1])
    return rows


def _model_id(ref: str) -> str:
    return ref.split("/", 1)[1] if "/" in ref else ref


def _fmt_usd(x: Any) -> str:
    return "—" if x is None else f"${x:,.2f}"


def create_checked_comparison(store: Any, name: str, run_ids: list[str]) -> tuple[bool, str]:
    """Create a comparison after the same fairness check the CLI's ``compare`` runs.

    Calls ``check_comparison`` and stores its filters with the comparison. Returns
    ``(created, markdown)``: on refusal (runs from different eval specs) nothing is created and
    the message explains why; otherwise the message lists any fairness warnings.
    """
    from blindearth.runner.planner import check_comparison

    ids = list(run_ids)
    try:
        filters = check_comparison(store, ids)
    except ValueError as e:
        return False, f"**Not created (refused):** {e}"
    cid = store.create_comparison(name, ids, filters=filters, ordering=None)
    msg = f"Comparison **{name}** ({str(cid)[:12]}) created with {len(ids)} runs. Nothing is re-called."
    warnings = filters.get("warnings") or []
    if warnings:
        msg += "\n\n**Fairness warnings:**\n" + "\n".join(f"- {w}" for w in warnings)
    return True, msg


# --------------------------------------------------------------------------- app


def build_app(db_path: str, registry_path: str) -> "gradio.Blocks":
    import gradio as gr

    from blindearth.evalspec.grid import make_grid
    from blindearth.providers.registry import Registry, load_registry
    from blindearth.report import mapgrid
    from blindearth.store.db import Store

    db_dir = Path(db_path).expanduser().resolve().parent
    st = _AppState()
    local = threading.local()

    def store() -> Store:
        s = getattr(local, "store", None)
        if s is None:
            s = Store(db_path)
            local.store = s
        return s

    try:
        registry = load_registry(registry_path)
        reg_msg = f"Registry: {len(registry.models)} models from `{registry_path}`."
    except Exception as exc:  # noqa: BLE001 - UI still usable for saved comparisons
        registry = Registry(providers={}, models={})
        reg_msg = f"Could not load registry `{registry_path}`: {exc}"

    def vendor_of(m) -> str:
        if m.vendor:
            return m.vendor
        try:
            return registry.provider_of(m).kind
        except Exception:  # noqa: BLE001
            return m.provider

    def model_choices(vendors: list[str] | None = None) -> list[tuple[str, str]]:
        out = []
        for m in registry.models.values():
            if vendors and vendor_of(m) not in vendors:
                continue
            star = " ★" if m.forced_thinking else ""
            quant = f" ({m.quant})" if m.quant else ""
            out.append((f"{m.id}{star} · {vendor_of(m)} · {m.name}{quant}", m.ref))
        return sorted(out)

    all_vendors = sorted({vendor_of(m) for m in registry.models.values()})

    def run_choices() -> list[tuple[str, str]]:
        out = []
        for r in store().list_runs():
            status = r.status.value if hasattr(r.status, "value") else str(r.status)
            date = (r.ended_at or r.started_at or "")[:16].replace("T", " ")
            out.append((f"{r.variant} · {status} · {date} · spec {r.spec_id[:8]} · {r.id[:8]}", r.id))
        return out

    def comparison_choices() -> list[str]:
        return [c["name"] for c in store().list_comparisons()]

    # ------------------------------------------------------------------ step 1: eval spec

    def on_preview(path, projection, invert, auto_thr, thr):
        if not path:
            return None, "Upload an equirectangular PNG or GeoTIFF first (white = land)."
        from blindearth.evalspec.masks import prepare_upload

        try:
            mask, preview, used = prepare_upload(
                path, projection=projection, invert=bool(invert),
                threshold=None if auto_thr else float(thr))
        except Exception as exc:  # noqa: BLE001
            return None, f"**Upload rejected:** {exc}"
        h, w = mask.data.shape
        land = float(np.asarray(mask.data).mean())
        return preview, (f"Binarized at threshold **{used:.3f}** ({'Otsu' if auto_thr else 'manual'}) · "
                         f"{w}×{h} px · land share {land*100:.1f}% (unweighted) · hash `{mask.hash[:12]}`. "
                         "Confirm the preview shows land white before saving.")

    def on_save_spec(mask_id, upload_path, projection, invert, auto_thr, thr, truth_rule, step,
                     custom_step, placement, subset_pct, seed, coord_fmt, template, system_prompt,
                     ext_mode, n_samples, temperature, budget, name):
        try:
            step_deg = float(custom_step) if step == "custom" else float(step)
            if step_deg <= 0 or (180 % step_deg) or (360 % step_deg):
                raise ValueError(f"grid step {step_deg}° must divide 180 and 360")
            mask_path = None
            if mask_id == "upload":
                if not upload_path:
                    raise ValueError("pick a mask file to upload")
                dest_dir = db_dir / "masks"
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / Path(upload_path).name
                if Path(upload_path).resolve() != dest.resolve():
                    shutil.copyfile(upload_path, dest)
                mask_path = str(dest)
            if not any(tok in template for tok in ("{coord}", "{lat}", "{lon}")):
                raise ValueError("prompt template needs {coord}, or {lat} and {lon}")
            default_prompt = PromptSpec()
            prompt = PromptSpec(
                id=default_prompt.id if template.strip() == default_prompt.template else "custom",
                template=template,
                system_prompt=(system_prompt or None),
            )
            subset = None if not subset_pct or float(subset_pct) >= 100 else float(subset_pct) / 100
            st.eval_spec = EvalSpec(
                mask=MaskSpec(id=mask_id, path=mask_path, truth_rule=truth_rule,
                              projection=projection, invert=bool(invert),
                              threshold=None if auto_thr else float(thr)),
                grid=GridSpec(step_deg=step_deg, placement=placement, subset_frac=subset,
                              seed=int(seed or 0)),
                coord_format=coord_fmt,
                prompt=prompt,
            )
            st.extraction = ExtractionSpec(mode=ExtractionMode(ext_mode), n_samples=int(n_samples),
                                           temperature=float(temperature))
            st.budget_usd = float(budget) if budget else None
            st.name = name or None
            n_points = len(make_grid(st.eval_spec.grid))
        except Exception as exc:  # noqa: BLE001
            return f"**Not saved:** {exc}"
        calls = n_points * (st.extraction.n_samples if st.extraction.mode == ExtractionMode.SAMPLE else 1)
        return (f"Eval spec saved: mask **{mask_id}** ({truth_rule.replace('_', ' ')}), "
                f"grid **{step_deg:g}°** {placement.replace('_', ' ')}, **{n_points:,} points**"
                f"{f' ({subset*100:g}% subset)' if subset else ''}, coords **{coord_fmt}**, "
                f"prompt **{prompt.id}**, extraction **{ext_mode}** "
                f"(≈{calls:,} calls per cell in sample mode). "
                f"Budget cap: {_fmt_usd(st.budget_usd)}. Next: pick models.")

    def on_load_yaml(path):
        from blindearth.evalspec.load import load_eval_file

        if not path:
            return [gr.update()] * 16 + ["Pick a YAML eval file.", gr.update()]
        try:
            ef = load_eval_file(path)
        except Exception as exc:  # noqa: BLE001
            return [gr.update()] * 16 + [f"**Could not load:** {exc}", gr.update()]
        st.eval_spec, st.extraction = ef.eval, ef.extraction
        st.budget_usd, st.name = ef.budget_usd, ef.name
        st.selected_models = [e.model_ref for e in ef.matrix]
        rows = []
        for e in ef.matrix:
            for c in e.configs or [RunConfig()]:
                rows.append([e.model_ref, c.effort or "", c.temperature, c.n_samples, c.top_logprobs,
                             c.max_output_tokens, c.seed, c.repeats])
        s = ef.eval
        step = f"{s.grid.step_deg:g}"
        step_known = step in ("1", "2", "4", "5")
        return [
            gr.update(value=s.mask.id if s.mask.id in BUILTIN_MASKS else "upload"),
            gr.update(value=s.mask.truth_rule),
            gr.update(value=step if step_known else "custom"),
            gr.update(value=s.grid.step_deg),
            gr.update(value=s.grid.placement),
            gr.update(value=(s.grid.subset_frac or 1.0) * 100),
            gr.update(value=s.grid.seed),
            gr.update(value=s.coord_format),
            gr.update(value=s.prompt.template),
            gr.update(value=s.prompt.system_prompt or ""),
            gr.update(value=ef.extraction.mode.value),
            gr.update(value=ef.extraction.n_samples),
            gr.update(value=ef.extraction.temperature),
            gr.update(value=ef.budget_usd),
            gr.update(value=ef.name or ""),
            gr.update(value=s.mask.projection),
            (f"Loaded `{Path(path).name}`: {len(ef.matrix)} models, {len(rows)} variants. "
             "Fields above now show the file; the variants table (step 3) is filled. "
             "Click **Save eval spec** if you change anything."),
            gr.update(value=rows or None),
        ]

    # ------------------------------------------------------------------ step 2: scope

    def on_scope(scope):
        saved = scope == "From saved runs"
        return (
            gr.update(visible=scope == "Within a vendor", value=None),
            gr.update(visible=scope == "Across vendors", value=[]),
            gr.update(visible=not saved, choices=model_choices(), value=[]),
            gr.update(visible=saved),
            gr.update(choices=run_choices(), value=[]),
        )

    def on_vendors(vendors):
        if isinstance(vendors, str):
            vendors = [vendors]
        return gr.update(choices=model_choices(vendors or None), value=[])

    def on_use_models(scope, models):
        models = list(models or [])
        try:
            if not models:
                raise ValueError("pick at least one model")
            picked = [registry.model(r) for r in models]
            vs = {vendor_of(m) for m in picked}
            if scope == "Within a vendor" and len(vs) != 1:
                raise ValueError("within-vendor scope needs models from one vendor")
            if scope == "Across vendors" and len(vs) < 2:
                raise ValueError("across-vendor scope needs models from two or more vendors")
            if scope == "Config sweep" and len(models) != 1:
                raise ValueError("a config sweep is one model with several configurations")
        except Exception as exc:  # noqa: BLE001
            return f"**Selection not used:** {exc}", gr.update()
        st.selected_models = models
        rows = [[m, "", None, None, None, None, None, 1] for m in models]
        if scope == "Config sweep":
            rows = apply_to_all(models, ["low", "medium", "high"], None, None, None, 1)
        return (f"{len(models)} model(s) selected ({', '.join(sorted(vs))}). "
                "Next: set configuration variants (step 3)."), gr.update(value=rows)

    async def on_probe(models):
        from blindearth.providers.probe import probe
        from blindearth.providers.registry import build_adapter

        if not models:
            return "Pick models to probe."
        lines = ["| model | logprobs | top-N | effort | efforts | max conc. | batch |",
                 "|---|---|---|---|---|---|---|"]
        for ref in models:
            try:
                m = registry.model(ref)
                p = registry.provider_of(m)
                adapter = build_adapter(p, m)
                try:
                    caps = await probe(adapter)
                finally:
                    await adapter.aclose()
                store().save_capabilities(p.id, m.name, caps)
                lines.append(f"| {m.id} | {caps.logprobs} | {caps.top_logprobs_max or '—'} | "
                             f"{caps.effort_param} | {', '.join(caps.supported_efforts) or '—'} | "
                             f"{caps.max_concurrency or '—'} | {caps.supports_batch} |")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"| {ref} | probe failed: {str(exc)[:80]} | | | | | |")
        return "\n".join(lines)

    def on_make_comparison(run_ids, name):
        if not run_ids:
            return "Pick saved runs.", gr.update()
        if not name:
            return "Name the comparison.", gr.update()
        ok, msg = create_checked_comparison(store(), name, list(run_ids))
        if not ok:
            return msg, gr.update()
        return (msg + "\n\nOpen the Saved comparisons tab to build the report."), \
            gr.update(choices=comparison_choices(), value=name)

    # ------------------------------------------------------------------ step 3: variants

    def on_apply_all(efforts, temperature, n_samples, max_tok, repeats):
        if not st.selected_models:
            return gr.update(), "Select models in step 2 first."
        rows = apply_to_all(st.selected_models, efforts, temperature, n_samples, max_tok, repeats)
        return gr.update(value=rows), f"{len(rows)} variants: {len(st.selected_models)} models × {len(efforts or ['default'])} settings."

    def on_check_variants(table):
        try:
            matrix = variants_from_table(table)
        except Exception as exc:  # noqa: BLE001
            return f"**Invalid table:** {exc}"
        if not matrix:
            return "No variants yet."
        st.matrix = matrix
        names = [cfg.variant_name(_model_id(ref)) for ref, cfg in matrix]
        dup = sorted({n for n in names if names.count(n) > 1})
        reps = sum(cfg.repeats for _, cfg in matrix)
        msg = f"{len(matrix)} variants ({reps} cells incl. repeats): " + ", ".join(f"`{n}`" for n in names)
        if dup:
            msg += f"\n\n**Duplicates** (same model and settings): {', '.join(dup)}. Use `repeats` instead."
        return msg + "\n\nNext: review the plan (step 4)."

    # ------------------------------------------------------------------ step 4: plan

    async def on_plan(table, run_pilot, pilot_points):
        from blindearth.evalspec.load import EvalFile, MatrixEntry
        from blindearth.runner.planner import build_plan

        matrix = variants_from_table(table)
        if not matrix:
            return None, "Set variants in step 3 first."
        grouped: dict[str, list[RunConfig]] = {}
        for ref, cfg in matrix:
            grouped.setdefault(ref, []).append(cfg)
        ef = EvalFile(eval=st.eval_spec, extraction=st.extraction,
                      matrix=[MatrixEntry(model_ref=r, configs=c) for r, c in grouped.items()],
                      budget_usd=st.budget_usd, name=st.name)
        try:
            plan = await build_plan(ef, registry, store(), pilot_points=int(pilot_points or 50),
                                    run_pilot=bool(run_pilot))
        except Exception as exc:  # noqa: BLE001
            return None, f"**Plan failed:** {exc}"
        st.plan = plan
        rows = []
        for c in plan.cells:
            u = c.est_usage
            rows.append({
                "variant": c.variant + (f" #{c.repeat_idx + 1}" if c.repeat_idx else ""),
                "provider": c.provider.id, "mode": c.mode.value, "points": c.n_points,
                "calls": 0 if c.cached_run_id else c.n_calls,
                "input tok": u.input_tokens, "output tok": u.output_tokens,
                "thinking tok": u.thinking_tokens,
                "est. cost": _fmt_usd(c.est_cost_usd),
                "cache": f"reuses {c.cached_run_id[:8]}" if c.cached_run_id else "",
                "refused": c.refused or "",
            })
        live = [c for c in plan.cells if not c.cached_run_id and not c.refused]
        calls = sum(c.n_calls for c in live)
        cached = sum(1 for c in plan.cells if c.cached_run_id)
        refused = [c for c in plan.cells if c.refused]
        md = [f"**{len(plan.cells)} cells** · {calls:,} calls · estimated **{_fmt_usd(plan.total_cost_usd)}** · "
              f"{cached} cache hit(s) · {len(refused)} refused · budget cap {_fmt_usd(plan.budget_usd)} · "
              f"{len(plan.points):,} points per cell"]
        if plan.budget_usd is not None and plan.total_cost_usd is not None and plan.total_cost_usd > plan.budget_usd:
            md.append(f"\n**Over budget:** the run pauses when spend reaches {_fmt_usd(plan.budget_usd)}.")
        if any(c.est_cost_usd is None for c in live):
            md.append("\nSome cells have no price; their cost is not in the total.")
        for c in refused:
            md.append(f"\n- Refused `{c.variant}`: {c.refused}")
        if not run_pilot:
            md.append("\nNo pilot ran; thinking-run costs may be far off. Enable the pilot for a measured estimate.")
        return pd.DataFrame(rows), "".join(md)

    # ------------------------------------------------------------------ step 5: run

    def on_progress(run_id: str, points: list[PointResult]) -> None:
        with st.lock:
            bucket = st.live.setdefault(run_id, {})
            for p in points:
                bucket[p.idx] = p
            if run_id not in st.run_ids:
                st.run_ids.append(run_id)

    async def _execute(plan, stop_event):
        from blindearth.runner.executor import execute

        try:
            st.run_state = "running"
            ids = await execute(plan, store(), registry, on_progress=on_progress,
                                budget_usd=plan.budget_usd, stop_event=stop_event)
            with st.lock:
                for rid in ids:
                    if rid not in st.run_ids:
                        st.run_ids.append(rid)
            st.run_state = "paused" if stop_event.is_set() else "finished"
        except asyncio.CancelledError:
            st.run_state = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            st.run_state, st.run_error = "failed", str(exc)

    async def _resume_all(run_ids, stop_event):
        from blindearth.runner.executor import resume

        try:
            st.run_state = "running"
            for rid in run_ids:
                if stop_event.is_set():
                    break
                r = store().get_run(rid)
                if r.status == RunStatus.COMPLETE:
                    continue
                await resume(rid, store(), registry, on_progress=on_progress, stop_event=stop_event)
            st.run_state = "paused" if stop_event.is_set() else "finished"
        except asyncio.CancelledError:
            st.run_state = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            st.run_state, st.run_error = "failed", str(exc)

    def _busy() -> bool:
        return st.task is not None and not st.task.done()

    async def on_start():
        if st.plan is None:
            return "Build a plan in step 4 first."
        if _busy():
            return "A run is already in progress."
        runnable = [c for c in st.plan.cells if not c.refused]
        if not runnable:
            return "Every cell was refused; nothing to run."
        st.stop_event = asyncio.Event()
        with st.lock:
            st.live.clear()
            st.base_loaded.clear()
            st.run_ids = [c.cached_run_id for c in st.plan.cells if c.cached_run_id]
        st.run_error, st.started = None, time.time()
        st.task = asyncio.create_task(_execute(st.plan, st.stop_event))
        return (f"Started {len(runnable)} cells. The map sharpens as points arrive "
                "(stratified order); refreshes every few seconds.")

    async def on_pause():
        if not _busy() or st.stop_event is None:
            return "Nothing is running."
        st.stop_event.set()
        return "Pausing: in-flight calls finish and are saved; press Resume to continue at the first missing point."

    async def on_resume():
        if _busy():
            return "Already running."
        ids = list(st.run_ids)
        if not ids:
            return "No runs to resume; start a run first."
        st.stop_event = asyncio.Event()
        st.run_error = None
        st.task = asyncio.create_task(_resume_all(ids, st.stop_event))
        return f"Resuming {len(ids)} run(s); finished runs are skipped."

    async def on_cancel():
        if not _busy():
            return "Nothing is running."
        if st.stop_event is not None:
            st.stop_event.set()
        done, _ = await asyncio.wait({st.task}, timeout=10)
        if not done:
            st.task.cancel()
        st.run_state = "cancelled"
        return ("Cancelled. Points already answered stay in the store; partial runs can be resumed "
                "later (`blindearth resume RUN_ID`) or compared as they are.")

    def _live_df(run_id: str) -> pd.DataFrame:
        with st.lock:
            pts = list(st.live.get(run_id, {}).values())
            need_base = run_id not in st.base_loaded
        rows = [{"idx": p.idx, "lat": p.lat, "lon": p.lon, "truth": p.truth, "p_land": p.p_land,
                 "weight": math.cos(math.radians(p.lat)), "latency_s": p.latency_s,
                 "error": p.error} for p in pts]
        df = pd.DataFrame(rows, columns=["idx", "lat", "lon", "truth", "p_land", "weight",
                                         "latency_s", "error"])
        if need_base:
            try:
                base = store().load_points(run_id)
            except Exception:  # noqa: BLE001
                base = None
            if base is not None and len(base):
                with st.lock:
                    bucket = st.live.setdefault(run_id, {})
                    for r in base.itertuples(index=False):
                        if int(r.idx) not in bucket:
                            bucket[int(r.idx)] = PointResult(
                                run_id=run_id, idx=int(r.idx), lat=float(r.lat), lon=float(r.lon),
                                truth=None if pd.isna(r.truth) else int(r.truth),
                                p_land=None if pd.isna(r.p_land) else float(r.p_land),
                                n_valid=0, n_samples=0, validity_mass=None, answer_text="",
                                finish_reason=None, latency_s=float(getattr(r, "latency_s", 0) or 0),
                                usage=_zero_usage(), error=getattr(r, "error", None))
                    st.base_loaded.add(run_id)
                return _live_df(run_id)
            with st.lock:
                st.base_loaded.add(run_id)
        return df

    def _variant(run_id: str) -> str:
        try:
            return store().get_run(run_id).variant
        except Exception:  # noqa: BLE001
            return run_id[:8]

    def on_tick(selected, live_mode):
        from blindearth.scoring.render import points_to_grid

        with st.lock:
            ids = list(st.run_ids)
        choices = [(f"{_variant(r)} · {r[:8]}", r) for r in ids]
        if not selected and ids:
            selected = ids[0]
        img = None
        if selected and st.eval_spec is not None:
            df = _live_df(selected)
            g = st.plan.eval_file.eval.grid if st.plan is not None else st.eval_spec.grid
            if len(df):
                if live_mode == "Errors":
                    rgb = mapgrid.grid_rgb(df, g.step_deg, g.placement, "error")
                    img = mapgrid.upscale(rgb, max(1, int(720 // rgb.shape[1])))
                else:
                    grid = points_to_grid(df, g.step_deg, g.placement, "p_land")
                    scale = max(1, int(720 // np.asarray(grid).shape[1]))
                    img = mapgrid.live_rgb(grid, binary=(live_mode == "Binary"), scale=scale)
        # progress table
        lines = ["| run | status | points | invalid | cost |", "|---|---|---|---|---|"]
        spent = 0.0
        for rid in ids:
            try:
                r = store().get_run(rid)
            except Exception:  # noqa: BLE001
                continue
            with st.lock:
                pts = list(st.live.get(rid, {}).values())
            n_inv = sum(1 for p in pts if p.p_land is None)
            done = max(r.n_points_done, len(pts))
            total = r.n_points_total or len(st.plan.points if st.plan else [])
            pct = f" ({done / total * 100:.0f}%)" if total else ""
            status = r.status.value if hasattr(r.status, "value") else str(r.status)
            spent += r.cost_usd or 0.0
            lines.append(f"| {r.variant} | {status} | {done:,} / {total:,}{pct} | {n_inv:,} | {_fmt_usd(r.cost_usd)} |")
        elapsed = f" · {time.time() - st.started:,.0f} s elapsed" if st.started else ""
        head = f"**State:** {st.run_state}{elapsed} · spend so far {_fmt_usd(spent)}"
        if st.plan is not None and st.plan.budget_usd is not None:
            head += f" of {_fmt_usd(st.plan.budget_usd)} cap"
        if st.run_error:
            head += f"\n\n**Error:** {st.run_error}"
        md = head + ("\n\n" + "\n".join(lines) if ids else "\n\nNo runs yet.")
        return img, md, gr.update(choices=choices, value=selected)

    def on_save_run_comparison(name):
        with st.lock:
            ids = list(st.run_ids)
        if not ids:
            return "No runs to save yet.", gr.update()
        if not name:
            return "Name the comparison.", gr.update()
        ok, msg = create_checked_comparison(store(), name, ids)
        if not ok:
            return msg, gr.update()
        return msg, gr.update(choices=comparison_choices(), value=name)

    # ------------------------------------------------------------------ saved comparisons

    def on_refresh():
        return gr.update(choices=comparison_choices()), gr.update(choices=run_choices())

    def on_summary(name, threshold):
        if not name:
            return None, "Pick a comparison."
        from blindearth.report.data import load_comparison

        try:
            info, views, _ = load_comparison(store(), name, float(threshold), registry=registry)
        except Exception as exc:  # noqa: BLE001
            return None, f"**Could not load:** {exc}"
        rows = []
        for v in sorted(views, key=lambda v: -(v.acc if v.acc == v.acc else -1)):
            m = v.metrics
            rows.append({
                "run": v.label + (" ★" if v.forced_thinking else ""), "vendor": v.vendor,
                "mode": v.mode, "acc (area)": f"{v.acc*100:.1f}%",
                "95% CI": f"[{v.acc_ci[1]*100:.1f}, {v.acc_ci[2]*100:.1f}]",
                "skill": _r(m.get("skill")), "F1": _r(m.get("f1")),
                "invalid": _p(m.get("invalid_rate")), "cost": _fmt_usd(v.run.cost_usd),
                "status": v.status,
            })
        warn = "\n\n".join(info.get("warnings") or [])
        return pd.DataFrame(rows), (f"{len(views)} runs · mask {info.get('mask_source')} · "
                                    f"modes {', '.join(info.get('modes', []))}" + (f"\n\n{warn}" if warn else ""))

    def on_build_html(name, threshold):
        if not name:
            return None, "Pick a comparison."
        from blindearth.report.html import build_report

        out = Path(tempfile.mkdtemp(prefix="blindearth-")) / f"{_safe(name)}.html"
        try:
            build_report(store(), name, out, threshold=float(threshold), registry=registry)
        except Exception as exc:  # noqa: BLE001
            return None, f"**Report failed:** {exc}"
        return str(out), f"Report written ({out.stat().st_size / 1e6:.1f} MB, self-contained, works offline)."

    def on_build_png(name, threshold, mode, coast, ncols):
        if not name:
            return None, None, "Pick a comparison."
        from blindearth.report.data import load_comparison
        from blindearth.report.mapgrid import render_map_grid

        try:
            info, views, mask = load_comparison(store(), name, float(threshold), registry=registry)
            views = sorted(views, key=lambda v: -(v.acc if v.acc == v.acc else -1))
            g = info["eval_spec"].grid
            png = render_map_grid(views, mode=mode, step_deg=g.step_deg, placement=g.placement,
                                  mask=mask, coastline=bool(coast),
                                  title=f"How {len(views)} blind models see the Earth",
                                  ncols=int(ncols), threshold=float(threshold))
        except Exception as exc:  # noqa: BLE001
            return None, None, f"**Map grid failed:** {exc}"
        out = Path(tempfile.mkdtemp(prefix="blindearth-")) / f"{_safe(name)}-{mode}.png"
        out.write_bytes(png)
        from PIL import Image

        return np.asarray(Image.open(out).convert("RGB")), str(out), "Map grid ready."

    # ------------------------------------------------------------------ layout

    with gr.Blocks(title="Blind Earth eval runner") as demo:
        gr.Markdown("# Blind Earth eval runner\nAsk text-only models *Land or Water?* for every "
                    "grid cell, map the answers and compare runs. " + reg_msg)

        with gr.Tab("1 · Eval spec"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Ground-truth mask")
                    mask_id = gr.Dropdown(BUILTIN_MASKS, value="natural-earth-land", label="Mask")
                    upload = gr.File(label="Upload equirectangular PNG / GeoTIFF (white = land)",
                                     file_types=[".png", ".tif", ".tiff"], type="filepath")
                    projection = gr.Radio(["equirectangular", "web_mercator"], value="equirectangular",
                                          label="Declared projection")
                    with gr.Row():
                        invert = gr.Checkbox(False, label="Invert (white = water)")
                        auto_thr = gr.Checkbox(True, label="Otsu threshold")
                    thr = gr.Slider(0, 1, value=0.5, step=0.01, label="Manual threshold")
                    preview_btn = gr.Button("Preview binarized upload")
                    preview_img = gr.Image(label="Binarized preview", interactive=False)
                    preview_md = gr.Markdown()
                    truth_rule = gr.Radio(["cell_center", "majority"], value="cell_center",
                                          label="Truth rule per cell")
                with gr.Column():
                    gr.Markdown("### Grid and prompt")
                    with gr.Row():
                        step = gr.Dropdown(["1", "2", "4", "5", "custom"], value="2", label="Grid step (°)")
                        custom_step = gr.Number(3, label="Custom step (°)")
                    placement = gr.Radio(["cell_center", "cell_corner"], value="cell_center",
                                         label="Point placement")
                    with gr.Row():
                        subset = gr.Slider(1, 100, value=100, step=1, label="Subset (% of points)")
                        seed = gr.Number(0, precision=0, label="Seed")
                    coord_fmt = gr.Radio(["hemisphere", "signed_decimal", "dms"], value="hemisphere",
                                         label="Coordinate format")
                    template = gr.Textbox(PromptSpec().template, lines=3,
                                          label="Prompt template ({coord}, or {lat} and {lon})")
                    system_prompt = gr.Textbox("", lines=2, label="System prompt (optional)")
                    gr.Markdown("### Extraction and budget")
                    with gr.Row():
                        ext_mode = gr.Radio([m.value for m in ExtractionMode], value="auto",
                                            label="Extraction mode")
                        n_samples = gr.Number(4, precision=0, label="Samples")
                        temperature = gr.Number(1.0, label="Temperature")
                    with gr.Row():
                        budget = gr.Number(None, label="Budget cap (USD)")
                        name = gr.Textbox("", label="Name (optional)")
            with gr.Row():
                save_spec_btn = gr.Button("Save eval spec", variant="primary")
                yaml_file = gr.File(label="…or load an eval YAML", file_types=[".yaml", ".yml"],
                                    type="filepath")
            spec_md = gr.Markdown()

        with gr.Tab("2 · Scope and models"):
            scope = gr.Radio(SCOPES, value="Within a vendor", label="Scope")
            vendor_one = gr.Dropdown(all_vendors, value=None, label="Vendor")
            vendor_many = gr.Dropdown(all_vendors, value=[], multiselect=True, label="Vendors",
                                      visible=False)
            models = gr.CheckboxGroup(model_choices(), label="Models (★ = forced thinking)")
            with gr.Group(visible=False) as saved_group:
                saved_runs = gr.CheckboxGroup(run_choices(), label="Saved runs (same eval spec)")
                with gr.Row():
                    cmp_name = gr.Textbox("", label="Comparison name")
                    cmp_btn = gr.Button("Create comparison (no API calls)", variant="primary")
            with gr.Row():
                use_btn = gr.Button("Use selection", variant="primary")
                probe_btn = gr.Button("Probe capabilities (20 calls each)")
            scope_md = gr.Markdown()
            probe_md = gr.Markdown()

        with gr.Tab("3 · Config variants"):
            gr.Markdown("One row per variant (`model@key=value`). Add rows to give a model more "
                        "variants; blank cells take the eval defaults.")
            with gr.Row():
                all_efforts = gr.CheckboxGroup(EFFORTS, value=["default"], label="Effort levels")
                all_temp = gr.Number(None, label="Temperature")
                all_n = gr.Number(None, precision=0, label="Samples")
                all_max = gr.Number(None, precision=0, label="Max output tokens")
                all_rep = gr.Number(1, precision=0, label="Repeats")
            apply_btn = gr.Button("Apply to all models")
            variants = gr.Dataframe(headers=VARIANT_COLS,
                                    datatype=["str", "str", "number", "number", "number", "number",
                                              "number", "number"],
                                    col_count=(len(VARIANT_COLS), "fixed"), row_count=(1, "dynamic"),
                                    interactive=True, label="Variants")
            check_btn = gr.Button("Check variants")
            variants_md = gr.Markdown()

        with gr.Tab("4 · Plan"):
            with gr.Row():
                run_pilot = gr.Checkbox(True, label="Run a pilot per cell (measures tokens per point; small spend)")
                pilot_points = gr.Number(50, precision=0, label="Pilot points")
            plan_btn = gr.Button("Build plan", variant="primary")
            plan_md = gr.Markdown()
            plan_table = gr.Dataframe(interactive=False, label="Cells")

        with gr.Tab("5 · Run"):
            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")
                pause_btn = gr.Button("Pause")
                resume_btn = gr.Button("Resume")
                cancel_btn = gr.Button("Cancel", variant="stop")
            run_msg = gr.Markdown()
            with gr.Row():
                live_run = gr.Dropdown([], label="Live map for run", allow_custom_value=False)
                live_mode = gr.Radio(["Probability", "Binary", "Errors"], value="Probability",
                                     label="Live map")
            live_img = gr.Image(label="Live map (grey = not yet asked)", interactive=False)
            progress_md = gr.Markdown()
            with gr.Row():
                run_cmp_name = gr.Textbox("", label="Save these runs as comparison")
                run_cmp_btn = gr.Button("Save comparison")
            run_cmp_md = gr.Markdown()

        with gr.Tab("Saved comparisons"):
            with gr.Row():
                cmp_pick = gr.Dropdown(comparison_choices(), label="Comparison")
                refresh_btn = gr.Button("Refresh")
                threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="P(Land) threshold")
            summary_btn = gr.Button("Show leaderboard")
            summary_md = gr.Markdown()
            summary_df = gr.Dataframe(interactive=False)
            with gr.Row():
                html_btn = gr.Button("Build HTML report", variant="primary")
                html_file = gr.File(label="HTML report")
            html_md = gr.Markdown()
            with gr.Row():
                png_mode = gr.Radio(["binary", "probability", "error"], value="binary", label="Map mode")
                png_coast = gr.Checkbox(False, label="True coastline")
                png_cols = gr.Slider(1, 8, value=4, step=1, label="Columns")
                png_btn = gr.Button("Build map grid PNG")
            png_img = gr.Image(label="Map grid", interactive=False)
            png_file = gr.File(label="PNG")
            png_md = gr.Markdown()

        # wiring: step 1
        preview_btn.click(on_preview, [upload, projection, invert, auto_thr, thr], [preview_img, preview_md])
        save_spec_btn.click(
            on_save_spec,
            [mask_id, upload, projection, invert, auto_thr, thr, truth_rule, step, custom_step,
             placement, subset, seed, coord_fmt, template, system_prompt, ext_mode, n_samples,
             temperature, budget, name],
            [spec_md])
        yaml_file.change(
            on_load_yaml, [yaml_file],
            [mask_id, truth_rule, step, custom_step, placement, subset, seed, coord_fmt, template,
             system_prompt, ext_mode, n_samples, temperature, budget, name, projection, spec_md,
             variants])
        # step 2
        scope.change(on_scope, [scope], [vendor_one, vendor_many, models, saved_group, saved_runs])
        vendor_one.change(on_vendors, [vendor_one], [models])
        vendor_many.change(on_vendors, [vendor_many], [models])
        use_btn.click(on_use_models, [scope, models], [scope_md, variants])
        probe_btn.click(on_probe, [models], [probe_md])
        cmp_btn.click(on_make_comparison, [saved_runs, cmp_name], [scope_md, cmp_pick])
        # step 3
        apply_btn.click(on_apply_all, [all_efforts, all_temp, all_n, all_max, all_rep], [variants, variants_md])
        check_btn.click(on_check_variants, [variants], [variants_md])
        # step 4
        plan_btn.click(on_plan, [variants, run_pilot, pilot_points], [plan_table, plan_md])
        # step 5
        start_btn.click(on_start, None, [run_msg])
        pause_btn.click(on_pause, None, [run_msg])
        resume_btn.click(on_resume, None, [run_msg])
        cancel_btn.click(on_cancel, None, [run_msg])
        run_cmp_btn.click(on_save_run_comparison, [run_cmp_name], [run_cmp_md, cmp_pick])
        tick_inputs, tick_outputs = [live_run, live_mode], [live_img, progress_md, live_run]
        live_mode.change(on_tick, tick_inputs, tick_outputs)
        if hasattr(gr, "Timer"):
            timer = gr.Timer(LIVE_REFRESH_S)
            timer.tick(on_tick, tick_inputs, tick_outputs)
        else:  # older Gradio
            demo.load(on_tick, tick_inputs, tick_outputs, every=LIVE_REFRESH_S)
        # saved comparisons
        refresh_btn.click(on_refresh, None, [cmp_pick, saved_runs])
        summary_btn.click(on_summary, [cmp_pick, threshold], [summary_df, summary_md])
        html_btn.click(on_build_html, [cmp_pick, threshold], [html_file, html_md])
        png_btn.click(on_build_png, [cmp_pick, threshold, png_mode, png_coast, png_cols],
                      [png_img, png_file, png_md])

    return demo


def _zero_usage():
    from blindearth.types import Usage

    return Usage()


def _r(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.3f}"


def _p(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x)*100:.2f}%"


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80] or "comparison"


def main(db_path: str = "blindearth.db", registry_path: str = "providers.yaml", port: int = 7860) -> None:
    app = build_app(db_path, registry_path)
    app.queue().launch(server_port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
