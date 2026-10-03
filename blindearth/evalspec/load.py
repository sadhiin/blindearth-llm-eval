"""Eval file (YAML) loading, validation and the spec hash.

Format::

    name: optional label                  # optional
    budget_usd: 25.0                      # optional hard cap for the comparison
    eval:
      mask: natural-earth-land            # id, or a mapping (below)
      # mask: { id: upload, path: truth.png, projection: equirectangular, invert: false,
      #         threshold: null, resolution_km: 1.0, truth_rule: majority }
      truth_rule: cell_center             # shortcut for mask.truth_rule
      grid: { step_deg: 2, placement: cell_center, subset_frac: 0.1, seed: 0 }
      coord_format: hemisphere            # hemisphere | signed_decimal | dms
      prompt: default-land-water          # built-in id, or { id, template, system_prompt }
      system_prompt: null                 # shortcut for prompt.system_prompt
    extraction:
      mode: auto                          # auto | logprobs | sample | greedy
      n_samples: 4
      temperature: 1.0
    matrix:
      - model: anthropic/opus-5-5         # registry ref
        repeats: 1                        # optional default for this entry's configs
        configs: [ { effort: low }, { effort: high, repeats: 3 } ]

Every error is an :class:`EvalFileError` that names the offending key path
(``matrix[1].configs[0].effrt: unknown key ...``).

Notes:

- YAML 1.1 reads ``effort: off`` as boolean false; it is mapped back to ``"off"``.
- ``providers:`` / ``models:`` top-level keys are tolerated and ignored, so one file can carry the
  registry too (the registry loader reads them).
- A relative ``mask.path`` is resolved against the YAML file's directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from blindearth.evalspec.coords import COORD_FORMATS
from blindearth.evalspec.grid import grid_dims
from blindearth.evalspec.masks import MASK_IDS
from blindearth.evalspec.prompts import BUILTIN_PROMPTS, DEFAULT_PROMPT_ID, validate_template
from blindearth.types import (
    EvalSpec,
    ExtractionMode,
    ExtractionSpec,
    GridSpec,
    MaskSpec,
    PromptSpec,
    RunConfig,
)

SPEC_HASH_VERSION = 1


class EvalFileError(ValueError):
    """Validation error in an eval file; the message starts with the key path."""

    def __init__(self, key: str, msg: str):
        self.key = key
        super().__init__(f"{key}: {msg}" if key else msg)


@dataclass
class MatrixEntry:
    model_ref: str
    configs: list[RunConfig] = field(default_factory=lambda: [RunConfig()])


@dataclass
class EvalFile:
    eval: EvalSpec
    extraction: ExtractionSpec
    matrix: list[MatrixEntry]
    budget_usd: float | None
    name: str | None


# --------------------------------------------------------------------------- small validators


def _mapping(v: Any, key: str) -> dict:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise EvalFileError(key, f"expected a mapping, got {type(v).__name__}")
    return v


def _no_unknown(d: dict, allowed: set[str] | tuple[str, ...], key: str) -> None:
    for k in d:
        if k not in allowed:
            path = f"{key}.{k}" if key else str(k)
            raise EvalFileError(path, f"unknown key (allowed: {', '.join(sorted(allowed))})")


def _num(v: Any, key: str, *, lo: float | None = None, hi: float | None = None,
         lo_open: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EvalFileError(key, f"expected a number, got {v!r}")
    x = float(v)
    if lo is not None and (x < lo or (lo_open and x == lo)):
        raise EvalFileError(key, f"must be {'>' if lo_open else '>='} {lo:g}, got {v!r}")
    if hi is not None and x > hi:
        raise EvalFileError(key, f"must be <= {hi:g}, got {v!r}")
    return x


def _int(v: Any, key: str, *, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        else:
            raise EvalFileError(key, f"expected an integer, got {v!r}")
    if lo is not None and v < lo:
        raise EvalFileError(key, f"must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise EvalFileError(key, f"must be <= {hi}, got {v}")
    return int(v)


def _str(v: Any, key: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise EvalFileError(key, f"expected a non-empty string, got {v!r}")
    return v.strip()


def _choice(v: Any, key: str, options: tuple[str, ...] | list[str],
            aliases: dict[str, str] | None = None) -> str:
    if not isinstance(v, str):
        raise EvalFileError(key, f"expected one of {', '.join(options)}, got {v!r}")
    s = v.strip().lower()
    if s not in options and s.replace("-", "_") in options:
        s = s.replace("-", "_")
    s = (aliases or {}).get(s, s)
    if s not in options:
        raise EvalFileError(key, f"expected one of {', '.join(options)}, got {v!r}")
    return s


# --------------------------------------------------------------------------- sections


_MASK_KEYS = {f.name for f in fields(MaskSpec)}
_GRID_KEYS = {f.name for f in fields(GridSpec)}
_PROMPT_KEYS = {f.name for f in fields(PromptSpec)}
_EXTRACTION_KEYS = {f.name for f in fields(ExtractionSpec)}
_CONFIG_KEYS = {f.name for f in fields(RunConfig)}
_EVAL_KEYS = {"mask", "truth_rule", "grid", "coord_format", "prompt", "system_prompt"}
_TOP_KEYS = {"name", "budget_usd", "eval", "extraction", "matrix", "providers", "models"}
_ENTRY_KEYS = {"model", "configs", "repeats"}


def _parse_mask(v: Any, truth_rule: Any, base_dir: Path | None) -> MaskSpec:
    key = "eval.mask"
    if v is None:
        d: dict = {}
    elif isinstance(v, str):
        d = {"id": v}
    else:
        d = dict(_mapping(v, key))
        _no_unknown(d, _MASK_KEYS, key)
        if "path" in d and "id" not in d:
            d["id"] = "upload"

    mid = d.get("id", MaskSpec.id)
    mid = _choice(mid, f"{key}.id" if isinstance(v, dict) else key, MASK_IDS)
    out: dict[str, Any] = {"id": mid}

    if "path" in d and d["path"] is not None:
        p = Path(_str(d["path"], f"{key}.path")).expanduser()
        if not p.is_absolute() and base_dir is not None:
            p = (base_dir / p).resolve()
        out["path"] = str(p)
    if mid == "upload" and "path" not in out:
        raise EvalFileError(f"{key}.path", "required when mask id is 'upload'")

    if "truth_rule" in d and truth_rule is not None:
        raise EvalFileError("eval.truth_rule", "set either eval.truth_rule or eval.mask.truth_rule")
    tr = d.get("truth_rule", truth_rule)
    if tr is not None:
        tr_key = f"{key}.truth_rule" if "truth_rule" in d else "eval.truth_rule"
        out["truth_rule"] = _choice(tr, tr_key, ("cell_center", "majority"),
                                    {"center": "cell_center"})
    if "resolution_km" in d:
        out["resolution_km"] = _num(d["resolution_km"], f"{key}.resolution_km", lo=0.2)
    if "projection" in d:
        out["projection"] = _choice(d["projection"], f"{key}.projection",
                                    ("equirectangular", "web_mercator"),
                                    {"mercator": "web_mercator", "plate_carree": "equirectangular"})
    if "invert" in d:
        if not isinstance(d["invert"], bool):
            raise EvalFileError(f"{key}.invert", f"expected true or false, got {d['invert']!r}")
        out["invert"] = d["invert"]
    if "threshold" in d and d["threshold"] is not None:
        out["threshold"] = _num(d["threshold"], f"{key}.threshold", lo=0.0, hi=1.0)
    return MaskSpec(**out)


def _parse_subset(v: Any, key: str) -> float | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().endswith("%"):
        try:
            v = float(v.strip()[:-1]) / 100.0
        except ValueError:
            raise EvalFileError(key, f"expected a fraction such as 0.1 or '10%', got {v!r}") from None
    x = _num(v, key, lo=0.0, hi=1.0, lo_open=True)
    return None if x == 1.0 else x


def _parse_grid(v: Any) -> GridSpec:
    key = "eval.grid"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = {"step_deg": v}
    d = _mapping(v, key)
    _no_unknown(d, _GRID_KEYS, key)
    out: dict[str, Any] = {}
    if "step_deg" in d:
        step = _num(d["step_deg"], f"{key}.step_deg", lo=0.0, hi=180.0, lo_open=True)
        try:
            grid_dims(step)
        except ValueError as e:
            raise EvalFileError(f"{key}.step_deg", str(e)) from None
        out["step_deg"] = step
    if "placement" in d:
        out["placement"] = _choice(d["placement"], f"{key}.placement",
                                   ("cell_center", "cell_corner"),
                                   {"center": "cell_center", "corner": "cell_corner"})
    if "subset_frac" in d:
        out["subset_frac"] = _parse_subset(d["subset_frac"], f"{key}.subset_frac")
    if "seed" in d:
        out["seed"] = _int(d["seed"], f"{key}.seed", lo=0)
    return GridSpec(**out)


def _parse_prompt(v: Any, system_prompt: Any) -> PromptSpec:
    key = "eval.prompt"
    if v is None:
        base = BUILTIN_PROMPTS[DEFAULT_PROMPT_ID]
        d: dict = {}
    elif isinstance(v, str):
        pid = v.strip()
        if pid not in BUILTIN_PROMPTS:
            raise EvalFileError(
                key,
                f"unknown built-in prompt {v!r} (built-ins: {', '.join(BUILTIN_PROMPTS)}); "
                "use a mapping with 'template' for a custom prompt",
            )
        base = BUILTIN_PROMPTS[pid]
        d = {}
    else:
        d = _mapping(v, key)
        _no_unknown(d, _PROMPT_KEYS, key)
        pid = d.get("id")
        if pid is not None:
            pid = _str(pid, f"{key}.id")
        if "template" in d:
            template = _str(d["template"], f"{key}.template") if d["template"] is not None else None
            if template is None:
                raise EvalFileError(f"{key}.template", "must not be null")
            try:
                validate_template(template)
            except ValueError as e:
                raise EvalFileError(f"{key}.template", str(e)) from None
            if pid is not None and pid in BUILTIN_PROMPTS and template != BUILTIN_PROMPTS[pid].template:
                raise EvalFileError(f"{key}.id", f"{pid!r} is a built-in id; give a custom template its own id")
            base = PromptSpec(id=pid or "custom", template=template)
        elif pid is not None:
            if pid not in BUILTIN_PROMPTS:
                raise EvalFileError(f"{key}.id", f"unknown built-in prompt {pid!r} and no template given")
            base = BUILTIN_PROMPTS[pid]
        else:
            base = BUILTIN_PROMPTS[DEFAULT_PROMPT_ID]

    sp = base.system_prompt
    if "system_prompt" in d and system_prompt is not None:
        raise EvalFileError("eval.system_prompt", "set either eval.system_prompt or eval.prompt.system_prompt")
    raw_sp = d.get("system_prompt", system_prompt)
    if raw_sp is not None:
        if not isinstance(raw_sp, str):
            k = f"{key}.system_prompt" if "system_prompt" in d else "eval.system_prompt"
            raise EvalFileError(k, f"expected a string, got {raw_sp!r}")
        sp = raw_sp
    return PromptSpec(id=base.id, template=base.template, system_prompt=sp)


def _parse_eval(v: Any, base_dir: Path | None) -> EvalSpec:
    d = _mapping(v, "eval")
    _no_unknown(d, _EVAL_KEYS, "eval")
    mask = _parse_mask(d.get("mask"), d.get("truth_rule"), base_dir)
    grid = _parse_grid(d.get("grid"))
    fmt = "hemisphere"
    if d.get("coord_format") is not None:
        fmt = _choice(d["coord_format"], "eval.coord_format", COORD_FORMATS,
                      {"signed": "signed_decimal", "decimal": "signed_decimal"})
    prompt = _parse_prompt(d.get("prompt"), d.get("system_prompt"))
    return EvalSpec(mask=mask, grid=grid, coord_format=fmt, prompt=prompt)  # type: ignore[arg-type]


def _parse_extraction(v: Any) -> ExtractionSpec:
    key = "extraction"
    d = _mapping(v, key)
    _no_unknown(d, _EXTRACTION_KEYS, key)
    out = ExtractionSpec()
    if d.get("mode") is not None:
        mode = _choice(d["mode"], f"{key}.mode", tuple(m.value for m in ExtractionMode))
        out.mode = ExtractionMode(mode)
    if "n_samples" in d:
        out.n_samples = _int(d["n_samples"], f"{key}.n_samples", lo=1, hi=16)
    if "temperature" in d:
        out.temperature = _num(d["temperature"], f"{key}.temperature", lo=0.0, hi=2.0)
    return out


def _parse_effort(v: Any, key: str) -> str | None:
    if v is None:
        return None
    if v is False:  # YAML 1.1: `effort: off` -> False
        return "off"
    if v is True:
        raise EvalFileError(key, "got boolean true; use off, low, medium, high, max or a provider-native value")
    if isinstance(v, (int, float)):  # e.g. a thinking budget in tokens
        return str(int(v)) if float(v).is_integer() else str(v)
    return _str(v, key).lower()


def _parse_config(v: Any, key: str, default_repeats: int | None) -> RunConfig:
    d = _mapping(v, key)
    _no_unknown(d, _CONFIG_KEYS - {"HASHED_FIELDS"}, key)
    out: dict[str, Any] = {}
    if "effort" in d:
        out["effort"] = _parse_effort(d["effort"], f"{key}.effort")
    if d.get("temperature") is not None:
        out["temperature"] = _num(d["temperature"], f"{key}.temperature", lo=0.0, hi=2.0)
    if d.get("n_samples") is not None:
        out["n_samples"] = _int(d["n_samples"], f"{key}.n_samples", lo=1, hi=16)
    if d.get("top_logprobs") is not None:
        out["top_logprobs"] = _int(d["top_logprobs"], f"{key}.top_logprobs", lo=1)
    if d.get("max_output_tokens") is not None:
        out["max_output_tokens"] = _int(d["max_output_tokens"], f"{key}.max_output_tokens", lo=1)
    if d.get("system_prompt") is not None:
        if not isinstance(d["system_prompt"], str):
            raise EvalFileError(f"{key}.system_prompt", f"expected a string, got {d['system_prompt']!r}")
        out["system_prompt"] = d["system_prompt"]
    for k in ("concurrency", "rpm", "tpm"):
        if d.get(k) is not None:
            out[k] = _int(d[k], f"{key}.{k}", lo=1)
    if d.get("seed") is not None:
        out["seed"] = _int(d["seed"], f"{key}.seed")
    if d.get("repeats") is not None:
        out["repeats"] = _int(d["repeats"], f"{key}.repeats", lo=1, hi=100)
    elif default_repeats is not None:
        out["repeats"] = default_repeats
    return RunConfig(**out)


def _parse_matrix(v: Any) -> list[MatrixEntry]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise EvalFileError("matrix", f"expected a list, got {type(v).__name__}")
    entries: list[MatrixEntry] = []
    for i, raw in enumerate(v):
        key = f"matrix[{i}]"
        if isinstance(raw, str):  # shorthand: `- anthropic/opus-5-5`
            raw = {"model": raw}
        d = _mapping(raw, key)
        _no_unknown(d, _ENTRY_KEYS, key)
        if "model" not in d:
            raise EvalFileError(f"{key}.model", "required")
        ref = _str(d["model"], f"{key}.model")
        rep = _int(d["repeats"], f"{key}.repeats", lo=1, hi=100) if d.get("repeats") is not None else None
        cfgs_raw = d.get("configs")
        if cfgs_raw is None:
            cfgs_raw = [{}]
        if not isinstance(cfgs_raw, list) or not cfgs_raw:
            raise EvalFileError(f"{key}.configs", "expected a non-empty list of mappings (use [ {} ] for defaults)")
        configs = [_parse_config(c, f"{key}.configs[{j}]", rep) for j, c in enumerate(cfgs_raw)]
        entries.append(MatrixEntry(model_ref=ref, configs=configs))
    return entries


# --------------------------------------------------------------------------- public API


def parse_eval_dict(doc: Any, base_dir: Path | None = None) -> EvalFile:
    """Validate an already-parsed YAML/JSON document."""
    d = _mapping(doc, "")
    _no_unknown(d, _TOP_KEYS, "")
    name = None
    if d.get("name") is not None:
        name = _str(str(d["name"]), "name")
    budget = None
    if d.get("budget_usd") is not None:
        budget = _num(d["budget_usd"], "budget_usd", lo=0.0, lo_open=True)
    return EvalFile(
        eval=_parse_eval(d.get("eval"), base_dir),
        extraction=_parse_extraction(d.get("extraction")),
        matrix=_parse_matrix(d.get("matrix")),
        budget_usd=budget,
        name=name,
    )


def load_eval_file(path: str | Path) -> EvalFile:
    p = Path(path).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EvalFileError("", f"eval file not found: {p}") from None
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise EvalFileError("", f"{p}: invalid YAML: {e}") from None
    return parse_eval_dict(doc, base_dir=p.resolve().parent)


# --------------------------------------------------------------------------- hashing / round trip

_FLOAT_FIELDS = {"step_deg", "subset_frac", "resolution_km", "threshold"}


def _canon(obj: Any, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: _canon(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, ExtractionMode):
        return obj.value
    if key in _FLOAT_FIELDS and isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    return obj


def canonical_spec(spec: EvalSpec) -> dict[str, Any]:
    """The hashed form of a spec. ``mask.path`` is dropped: the mask's content hash identifies it,
    so the same upload stored in two places gives the same spec hash."""
    d = _canon(spec.to_dict())
    d["mask"].pop("path", None)
    return d


def spec_hash(spec: EvalSpec, mask_hash: str) -> str:
    payload = {"v": SPEC_HASH_VERSION, "spec": canonical_spec(spec), "mask_hash": mask_hash}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def evalspec_from_dict(d: dict[str, Any]) -> EvalSpec:
    """Inverse of ``EvalSpec.to_dict()`` (for the store)."""
    return EvalSpec(
        mask=MaskSpec(**d.get("mask", {})),
        grid=GridSpec(**d.get("grid", {})),
        coord_format=d.get("coord_format", "hemisphere"),
        prompt=PromptSpec(**d.get("prompt", {})),
    )
