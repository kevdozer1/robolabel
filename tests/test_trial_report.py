"""Tests for the blind-trial tally, incl. the mark-failures-only protocol."""

from __future__ import annotations

from robolabel.trial_report import tally


def _unblind():
    return {
        "__dataset__": "ds",
        "A": {"episode_id": "0", "strategy": "grounded", "bands": [], "denom": {"b": 3, "p": 3, "e": 3}},
        "B": {"episode_id": "1", "strategy": "grounded", "bands": ["uniform_split"], "denom": {"b": 2, "p": 2, "e": 0}},
    }


def test_mark_failures_only_unmarked_is_pass():
    # A has one boundary failure; B has none. Boundary acceptance = 1 - 1/(3+2) = 0.8.
    grades = {"A": {"marks": {"b0": False}, "verdict": "touchup"},
              "B": {"marks": {}, "verdict": "usable"}}
    t = tally(grades, _unblind(), protocol="mark-failures-only")["grounded"]
    assert t["n_items"] == 2
    assert abs(t["boundary_acceptance"] - 0.8) < 1e-9     # 1 failure / 5 boundaries
    assert t["n_boundaries"] == 5
    assert t["phase_accuracy"] == 1.0                      # no phase marks -> all pass
    assert t["evidence_factual_accuracy"] == 1.0           # no evidence failures over 3 slots
    assert t["n_evidence"] == 3                            # only A had evidence slots
    assert t["failure_band_rate"] == 0.5                   # B is in a band
    assert t["verdicts"] == {"usable": 1, "touchup": 1, "garbage": 0}


def test_mark_all_protocol_is_mean_of_marks():
    grades = {"A": {"marks": {"b0": True, "b1": False, "p0": True}, "verdict": "usable"}}
    t = tally(grades, _unblind(), protocol="mark-all")["grounded"]
    assert t["boundary_acceptance"] == 0.5                 # 1 of 2 boundary marks true
    assert t["phase_accuracy"] == 1.0                      # 1 of 1


def test_evidence_na_when_no_evidence_slots():
    # A strategy with no evidence (e.g. S0) -> evidence accuracy is None, not 1.0.
    ub = {"X": {"episode_id": "0", "strategy": "S0", "bands": [], "denom": {"b": 3, "p": 3, "e": 0}}}
    t = tally({"X": {"marks": {}, "verdict": "garbage"}}, ub, protocol="mark-failures-only")["S0"]
    assert t["evidence_factual_accuracy"] is None
