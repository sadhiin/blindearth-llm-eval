import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from blindearth.evalspec.load import (
    EvalFileError,
    canonical_spec,
    evalspec_from_dict,
    load_eval_file,
    parse_eval_dict,
    spec_hash,
)
from blindearth.types import EvalSpec, ExtractionMode, GridSpec, PromptSpec, RunConfig

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _write(tmp_path, text, name="spec.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_example_spec_loads():
    ef = load_eval_file(EXAMPLES / "spec.yaml")
    assert ef.eval.mask.id == "natural-earth-land"
    assert ef.eval.grid == GridSpec(step_deg=2.0, placement="cell_center")
    assert ef.eval.coord_format == "hemisphere"
    assert ef.eval.prompt == PromptSpec()
    assert ef.extraction.mode is ExtractionMode.AUTO
    assert ef.extraction.n_samples == 4 and ef.extraction.temperature == 1.0
    assert [m.model_ref for m in ef.matrix] == ["anthropic/opus-5-5", "openai/gpt-model", "my-vllm/qwen-local"]
    assert [c.effort for c in ef.matrix[0].configs] == ["low", "high"]
    assert ef.matrix[1].configs[0].effort == "off"  # YAML `off` is boolean false
    assert ef.matrix[2].configs == [RunConfig()]


def test_smoke_example_loads():
    ef = load_eval_file(EXAMPLES / "smoke_4deg.yaml")
    assert ef.eval.grid.step_deg == 4.0
    assert ef.eval.grid.subset_frac == pytest.approx(0.1)
    assert ef.budget_usd == 2.0


def test_full_featured_file(tmp_path):
    (tmp_path / "truth.png").write_bytes(b"")
    p = _write(tmp_path, """
        name: custom
        budget_usd: 12.5
        eval:
          mask: { path: truth.png, invert: true, threshold: 0.4, projection: web-mercator,
                  truth_rule: majority }
          grid: { step_deg: 5, placement: corner, subset_frac: "10%", seed: 7 }
          coord_format: signed-decimal
          prompt:
            id: terse
            template: "Land or Water? {lat}, {lon}"
            system_prompt: Answer in one word.
        extraction: { mode: sample, n_samples: 8, temperature: 0.7 }
        matrix:
          - model: anthropic/opus-5-5
            repeats: 3
            configs:
              - { effort: low }
              - { effort: high, repeats: 1, max_output_tokens: 2048 }
          - my-vllm/qwen-local
    """)
    ef = load_eval_file(p)
    m = ef.eval.mask
    assert m.id == "upload" and Path(m.path) == (tmp_path / "truth.png").resolve()
    assert m.invert and m.threshold == 0.4 and m.projection == "web_mercator"
    assert m.truth_rule == "majority"
    assert ef.eval.grid == GridSpec(step_deg=5.0, placement="cell_corner", subset_frac=0.1, seed=7)
    assert ef.eval.coord_format == "signed_decimal"
    assert ef.eval.prompt.id == "terse" and ef.eval.prompt.system_prompt == "Answer in one word."
    assert ef.extraction.mode is ExtractionMode.SAMPLE and ef.extraction.n_samples == 8
    assert [c.repeats for c in ef.matrix[0].configs] == [3, 1]
    assert ef.matrix[0].configs[1].max_output_tokens == 2048
    assert ef.matrix[1].model_ref == "my-vllm/qwen-local" and ef.matrix[1].configs == [RunConfig()]
    assert ef.name == "custom" and ef.budget_usd == 12.5


def test_defaults_when_sections_missing():
    ef = parse_eval_dict({})
    assert ef.eval == EvalSpec()
    assert ef.matrix == [] and ef.name is None and ef.budget_usd is None


@pytest.mark.parametrize(
    "doc,key",
    [
        ({"evall": {}}, "evall"),
        ({"eval": {"grid": {"stepdeg": 2}}}, "eval.grid.stepdeg"),
        ({"eval": {"grid": {"step_deg": 7}}}, "eval.grid.step_deg"),
        ({"eval": {"grid": {"subset_frac": 0}}}, "eval.grid.subset_frac"),
        ({"eval": {"coord_format": "utm"}}, "eval.coord_format"),
        ({"eval": {"mask": "openstreetmap"}}, "eval.mask"),
        ({"eval": {"mask": {"id": "upload"}}}, "eval.mask.path"),
        ({"eval": {"mask": {"id": "gshhg", "threshold": 2}}}, "eval.mask.threshold"),
        ({"eval": {"prompt": "nope"}}, "eval.prompt"),
        ({"eval": {"prompt": {"template": "no coords"}}}, "eval.prompt.template"),
        ({"eval": {"truth_rule": "majority", "mask": {"id": "gshhg", "truth_rule": "majority"}}},
         "eval.truth_rule"),
        ({"extraction": {"mode": "vibes"}}, "extraction.mode"),
        ({"extraction": {"n_samples": 0}}, "extraction.n_samples"),
        ({"extraction": {"temperature": 3}}, "extraction.temperature"),
        ({"matrix": {"model": "x"}}, "matrix"),
        ({"matrix": [{"configs": [{}]}]}, "matrix[0].model"),
        ({"matrix": [{"model": "a/b", "configs": [{"effrt": "low"}]}]}, "matrix[0].configs[0].effrt"),
        ({"matrix": [{"model": "a/b", "configs": [{"effort": True}]}]}, "matrix[0].configs[0].effort"),
        ({"matrix": [{"model": "a/b", "configs": []}]}, "matrix[0].configs"),
        ({"matrix": [{"model": "a/b", "configs": [{"n_samples": 40}]}]}, "matrix[0].configs[0].n_samples"),
        ({"budget_usd": -1}, "budget_usd"),
    ],
)
def test_validation_errors_name_the_key(doc, key):
    with pytest.raises(EvalFileError) as ei:
        parse_eval_dict(doc)
    assert ei.value.key == key
    assert str(ei.value).startswith(key)


def test_registry_keys_tolerated():
    ef = parse_eval_dict({"providers": [], "models": [], "matrix": ["a/b"]})
    assert ef.matrix[0].model_ref == "a/b"


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(EvalFileError):
        load_eval_file(tmp_path / "missing.yaml")
    with pytest.raises(EvalFileError):
        load_eval_file(_write(tmp_path, "eval: [unclosed"))


def test_spec_hash_stable_and_canonical():
    a = EvalSpec(grid=GridSpec(step_deg=2))
    b = EvalSpec(grid=GridSpec(step_deg=2.0))
    h = spec_hash(a, "m1")
    assert len(h) == 64
    assert h == spec_hash(b, "m1")
    assert h != spec_hash(a, "m2")
    assert h != spec_hash(replace(a, coord_format="dms"), "m1")
    assert h != spec_hash(replace(a, prompt=PromptSpec(system_prompt="x")), "m1")
    # The upload path is not hashed; the mask content hash identifies the mask.
    u1 = EvalSpec(mask=replace(a.mask, id="upload", path="/a/x.png"))
    u2 = EvalSpec(mask=replace(a.mask, id="upload", path="/b/x.png"))
    assert spec_hash(u1, "m1") == spec_hash(u2, "m1")
    assert "path" not in canonical_spec(u1)["mask"]


def test_evalspec_round_trip():
    spec = EvalSpec(grid=GridSpec(step_deg=4.0, subset_frac=0.1, seed=3), coord_format="dms")
    assert evalspec_from_dict(spec.to_dict()) == spec
