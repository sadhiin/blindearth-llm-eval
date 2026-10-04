from __future__ import annotations

from types import SimpleNamespace

from blindearth.types import ExtractionMode, RunStatus
from blindearth.ui.app import create_checked_comparison


def _run(rid, spec="s1", mode=ExtractionMode.LOGPROBS, status=RunStatus.COMPLETE,
         forced=False, thinking=False):
    return SimpleNamespace(id=rid, spec_id=spec, extraction_mode=mode, status=status,
                           forced_thinking=forced, thinking=thinking)


class FakeStore:
    def __init__(self, runs):
        self.runs = {r.id: r for r in runs}
        self.created = []

    def get_run(self, rid):
        return self.runs[rid]

    def create_comparison(self, name, run_ids, filters=None, ordering=None):
        self.created.append((name, list(run_ids), filters, ordering))
        return "c" * 32


def test_refuses_mixed_specs():
    store = FakeStore([_run("a" * 32, spec="s1"), _run("b" * 32, spec="s2")])
    ok, msg = create_checked_comparison(store, "cmp", ["a" * 32, "b" * 32])
    assert not ok
    assert "refused" in msg.lower() and "different eval specs" in msg
    assert store.created == []


def test_stores_filters_and_shows_warnings():
    store = FakeStore([
        _run("a" * 32, mode=ExtractionMode.LOGPROBS),
        _run("b" * 32, mode=ExtractionMode.SAMPLE, status=RunStatus.PAUSED, forced=True, thinking=True),
    ])
    ok, msg = create_checked_comparison(store, "cmp", ["a" * 32, "b" * 32])
    assert ok
    (name, ids, filters, ordering), = store.created
    assert (name, ids, ordering) == ("cmp", ["a" * 32, "b" * 32], None)
    assert filters["spec_id"] == "s1"
    assert filters["mixed_extraction_modes"] is True
    assert filters["rank_on"] == "acc_area"
    assert filters["forced_thinking_runs"] == ["b" * 32]
    assert "Fairness warnings" in msg
    for w in filters["warnings"]:
        assert w in msg
    assert "mixed extraction modes" in msg and "partial runs" in msg


def test_clean_comparison_has_no_warnings():
    store = FakeStore([_run("a" * 32), _run("b" * 32)])
    ok, msg = create_checked_comparison(store, "cmp", ["a" * 32, "b" * 32])
    assert ok and "warning" not in msg.lower()
    assert store.created[0][2]["warnings"] == []
