"""Report data, charts and HTML tests on synthetic RunViews (no store, no network)."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blindearth.report import charts, data
from blindearth.report.data import ComparisonError, RunView
from blindearth.types import EvalSpec, ExtractionMode, GridSpec, ModelSpec, RunRecord, RunStatus

from test_mapgrid import make_df  # noqa: E402  (tests dir is on sys.path under pytest rootdir)

STEP = 10.0


def view(variant: str, df: pd.DataFrame, *, vendor="anthropic", mode=ExtractionMode.SAMPLE,
         cost=1.0, release="2025-01-01", forced=False, thinking=False, regions=None) -> RunView:
    acc = data.area_accuracy(df)
    run = RunRecord(id="run-" + variant, run_hash="h-" + variant, spec_id="spec", model_id="p/x",
                    config_id="c", variant=variant, extraction_mode=mode,
                    status=RunStatus.COMPLETE, started_at="2026-10-01T10:00:00Z",
                    ended_at="2026-10-01T11:00:00Z", runner_version="0.1.0",
                    forced_thinking=forced, thinking=thinking or forced, cost_usd=cost,
                    n_points_total=len(df), n_points_done=len(df))
    model = ModelSpec(id=variant.split("@")[0], provider="p", name=variant.split("@")[0],
                      vendor=vendor, release_date=release, forced_thinking=forced)
    metrics = {"acc_area": acc, "acc_valid_only": acc, "skill": 0.5, "f1": 0.8,
               "invalid_rate": float(df["p_land"].isna().mean()), "brier": 0.1, "ece": 0.05,
               "latency_p50": 0.3, "latency_p95": 0.5, "n_points": len(df)}
    return RunView(run=run, model=model, df=df, metrics=metrics,
                   regions=regions or {"africa": acc, "pacific_ocean": acc - 0.05, "antarctica": 0.7},
                   acc_ci=(acc, acc - 0.01, acc + 0.01))


@pytest.fixture
def views():
    return [
        view("opus@effort=low", make_df(flip_every=11), cost=2.0, release="2025-05-01"),
        view("opus@effort=high", make_df(flip_every=23), cost=8.0, release="2025-05-01",
             thinking=True),
        view("sonnet", make_df(flip_every=4, invalid_every=13), cost=0.5, release="2024-10-01"),
        view("gpt-x", make_df(flip_every=9), vendor="openai", mode=ExtractionMode.LOGPROBS,
             cost=3.0, release="2025-08-01", forced=True),
    ]


def fake_paired(df_a, df_b, *, threshold=0.5, block_deg=10.0, n_boot=1000, seed=0):
    d = data.area_accuracy(df_a, threshold) - data.area_accuracy(df_b, threshold)
    return {"diff": d, "lo": d - 0.02, "hi": d + 0.02, "significant": abs(d) > 0.02,
            "n_paired": int(min(len(df_a), len(df_b)))}


# --------------------------------------------------------------------------- data helpers


def test_parse_variant():
    assert data.parse_variant("opus-5-5@effort=low,temperature=1.0") == (
        "opus-5-5", {"effort": "low", "temperature": "1.0"})
    assert data.parse_variant("plain") == ("plain", {})


def test_area_accuracy_counts_invalid_as_wrong():
    df = make_df()
    assert data.area_accuracy(df) == pytest.approx(1.0)
    bad = df.copy()
    bad.loc[bad["idx"] % 2 == 0, "p_land"] = np.nan
    acc = data.area_accuracy(bad)
    w = bad["weight"].to_numpy()
    expected = w[bad["idx"].to_numpy() % 2 == 1].sum() / w.sum()
    assert acc == pytest.approx(expected)


def test_error_breakdown_sums_to_one():
    df = make_df(flip_every=6, invalid_every=17)
    b = data.error_breakdown(df)
    assert b["correct"] + b["false_land"] + b["false_water"] + b["invalid"] == pytest.approx(1.0)
    assert b["n_false_land"] > 0 and b["n_false_water"] > 0 and b["n_invalid"] > 0


def test_runview_properties(views):
    v = views[1]
    assert v.model_key == "opus" and v.effort == "high" and v.thinking
    assert views[3].forced_thinking and views[3].thinking
    assert views[3].probabilistic and views[0].probabilistic  # sample with n_samples=4


# --------------------------------------------------------------------------- charts


def test_pareto_frontier():
    pts = [(1.0, 0.80), (2.0, 0.85), (3.0, 0.84), (0.5, 0.70), (4.0, 0.90)]
    assert charts.pareto_frontier(pts) == [3, 0, 1, 4]


def test_reliability_bins_perfect():
    df = make_df()
    rb = charts.reliability_bins(df, 10)
    assert rb["count"].sum() == len(df)
    assert (abs(rb["mean_p"] - rb["observed"]) < 0.1).all()


def test_entity_colours_follow_model(views):
    slots = charts.entity_colours(views)
    assert slots == {"opus": 0, "sonnet": 1, "gpt-x": 2}


def test_charts_build(views):
    assert charts.effort_curve(views) is not None          # opus has two effort levels
    assert charts.effort_curve(views[2:]) is None          # nothing swept
    assert charts.cost_vs_accuracy(views) is not None
    assert charts.lineage(views) is not None
    assert charts.calibration(views) is not None
    assert charts.latency_histogram(views) is not None
    assert charts.error_split(views) is not None
    assert charts.fig_html(None, "x") == ""


# --------------------------------------------------------------------------- ranking / html


def test_rank_with_ties(monkeypatch, views):
    from blindearth.report import html

    monkeypatch.setattr(html, "paired_difference", fake_paired)
    pairs = html.PairCache(views, 0.5, 10)
    ranks = html.rank_with_ties(views, pairs)
    order = sorted(range(len(views)), key=lambda i: -views[i].acc)
    assert ranks[order[0]]["rank"] == 1
    assert sorted(r["rank"] for r in ranks.values())[0] == 1
    # flipped pair sign
    d01 = pairs.get(0, 1)
    d10 = pairs.get(1, 0)
    assert d01["diff"] == pytest.approx(-d10["diff"])
    assert d01["lo"] == pytest.approx(-d10["hi"])


def test_near_ties_share_rank(monkeypatch):
    from blindearth.report import html

    a = view("fable-5", make_df(flip_every=40))
    b = view("fable-5.1", make_df(flip_every=41))
    c = view("small", make_df(flip_every=3))
    monkeypatch.setattr(html, "paired_difference", fake_paired)
    ranks = html.rank_with_ties([a, b, c], html.PairCache([a, b, c], 0.5, 10))
    assert ranks[0]["rank"] == ranks[1]["rank"] == 1 and ranks[0]["tied"] and ranks[1]["tied"]
    assert ranks[2]["rank"] == 3 and not ranks[2]["tied"]


def _info(mixed: bool) -> dict:
    return {"id": "cmp1", "name": "Blind test", "run_ids": [], "created_at": "2026-10-02",
            "eval_spec": EvalSpec(grid=GridSpec(step_deg=STEP)), "spec_id": "spec" * 8,
            "mask_hash": "ab" * 32, "mask_hash_loaded": "ab" * 32, "mask_source": "synthetic mask",
            "modes": ["logprobs", "sample"] if mixed else ["sample"], "mixed_modes": mixed,
            "threshold": 0.5, "warnings": ["Runs use different extraction modes"] if mixed else []}


def test_render_report_self_contained(monkeypatch, views):
    from blindearth.report import html

    monkeypatch.setattr(html, "paired_difference", fake_paired)
    land = np.zeros((180, 360), bool)
    land[50:130, 160:220] = True
    mask = SimpleNamespace(data=land, source="synthetic mask", hash="ab" * 32)
    out = html.render_report(_info(mixed=True), views, mask, threshold=0.5, n_boot=10)

    for anchor in ("maps", "errors", "leaderboard", "effort", "cost", "lineage", "diff",
                   "regions", "calibration", "health", "caveats"):
        assert f'id="{anchor}"' in out
    # offline: plotly.js inlined in <head>; no external scripts, stylesheets or images
    head, body = out.split("<body>", 1)
    assert "<script>" in head and len(head) > 1_000_000
    assert not re.search(r'<script[^>]+src=', body)
    assert not re.search(r'<link[^>]+href="http', out)
    assert not re.search(r'<img[^>]+src="http', body)
    assert out.count("data:image/png;base64,") >= len(views)
    # header facts: mask, runner version, total spend 2 + 8 + 0.5 + 3
    assert "synthetic mask" in out and "0.1.0" in out and "$13.50" in out
    # mixed-mode: probability columns are not sortable and rows flagged
    assert 'class="nosort"' in out and "mixed</span>" in out
    # forced thinking star and validity caveats
    assert "★" in out and "survived compression" in out
    # all pairs precomputed for the diff view
    m = re.search(r'<script type="application/json" id="diffdata">(.*?)</script>', out, re.S)
    diff = json.loads(m.group(1))
    assert len(diff["runs"]) == 4 and len(diff["pairs"]) == 6


def test_render_report_caps_diff_pairs(monkeypatch, views):
    from blindearth.report import html

    monkeypatch.setattr(html, "paired_difference", fake_paired)
    mask = SimpleNamespace(data=np.zeros((90, 180), bool), source="m", hash="0" * 64)
    out = html.render_report(_info(mixed=False), views, mask, max_diff_runs=3, n_boot=10)
    m = re.search(r'id="diffdata">(.*?)</script>', out, re.S)
    assert len(json.loads(m.group(1))["pairs"]) == 3
    assert 'class="nosort"' not in out


# --------------------------------------------------------------------------- fairness


class FakeStore:
    def __init__(self, runs):
        self.runs = {r.id: r for r in runs}

    def get_comparison(self, name):
        return {"id": "c", "name": name, "run_ids": list(self.runs), "filters": None,
                "ordering": None, "created_at": "2026-10-01"}

    def get_run(self, rid):
        return self.runs[rid]

    def get_eval_spec(self, spec_id):  # pragma: no cover - must not be reached
        raise AssertionError("fairness check must run before loading the spec")


def test_load_comparison_rejects_mixed_specs():
    r1 = RunRecord(id="a", run_hash="h", spec_id="spec-1", model_id="m", config_id="c",
                   variant="a", extraction_mode=ExtractionMode.SAMPLE)
    r2 = RunRecord(id="b", run_hash="h", spec_id="spec-2", model_id="m", config_id="c",
                   variant="b", extraction_mode=ExtractionMode.SAMPLE)
    with pytest.raises(ComparisonError, match="eval specs"):
        data.load_comparison(FakeStore([r1, r2]), "x")


def test_load_comparison_rejects_empty():
    with pytest.raises(ComparisonError):
        data.load_comparison(FakeStore([]), "x")
