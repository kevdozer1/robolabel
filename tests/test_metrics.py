"""Tests for the shared boundary-quality metrics."""

from __future__ import annotations

from robolabel.metrics import boundaries, boundary_pr_mae, episode_iou, temporal_iou


def test_temporal_iou_exact_and_disjoint():
    assert temporal_iou({"start": 0, "end": 9}, {"start": 0, "end": 9}) == 1.0
    assert temporal_iou({"start": 0, "end": 4}, {"start": 10, "end": 14}) == 0.0
    # half overlap: [0,9] vs [5,14] -> inter [5,9]=5, union [0,14]=15
    assert abs(temporal_iou({"start": 0, "end": 9}, {"start": 5, "end": 14}) - 5 / 15) < 1e-9


def test_episode_iou_index_aligned():
    auto = [{"start": 0, "end": 10}, {"start": 11, "end": 20}]
    gold = [{"start": 0, "end": 10}, {"start": 11, "end": 20}]
    assert episode_iou(auto, gold) == 1.0


def test_boundaries_are_internal_transitions():
    segs = [{"start": 0, "end": 10}, {"start": 11, "end": 20}, {"start": 21, "end": 30}]
    assert boundaries(segs) == [10, 20]          # last segment's end (30) is not a boundary
    assert boundaries([{"start": 0, "end": 30}]) == []  # single segment -> no boundaries


def test_boundary_pr_mae_greedy_within_tol():
    # gold boundaries 10, 50; pred 12 (matches 10, err 2), 90 (no match)
    r = boundary_pr_mae([12, 90], [10, 50], tol=5)
    assert r["matched"] == 1
    assert r["recall"] == 0.5          # 1 of 2 gold matched
    assert r["precision"] == 0.5       # 1 of 2 pred matched
    assert r["mae"] == 2.0


def test_boundary_pr_mae_no_gold_boundaries():
    # single-segment gold (ep7 shape): any predicted boundary is a false positive.
    r = boundary_pr_mae([40, 120], [], tol=5)
    assert r["matched"] == 0
    assert r["recall"] is None         # undefined with no gold boundaries
    assert r["precision"] == 0.0       # all predicted are false positives
