"""Tests for the temporal (T1 to T6), cost (K1 to K3) and component (X1 to X3) metrics.

Appendix F items 1 to 5 and 9 of MEASUREMENT_SPEC are checked with the hand-computed numbers.
"""

from __future__ import annotations

import itertools
import json
import logging
import random

import pytest

from robolabel.eval import component, cost_metrics, temporal
from robolabel.eval.temporal import (
    boundaries,
    failed_spans_from_segments,
    greedy_match_count,
    match_boundaries,
    match_spans,
    missing_output_segments,
    t1_episode,
    t1_micro,
    t2,
    t2_summary,
    t3_episode,
    t3_legacy_episode,
    t3_legacy_flat_mean,
    t3_legacy_ious,
    t3_pairs,
    t3_summary,
    t4_episode,
    t4_rate,
    t5_episode,
    t5_summary,
    t6_episode,
)


def _segs(*ranges: tuple[int, int], **extra) -> list[dict]:
    return [{"start_frame": s, "end_frame": e, **extra} for s, e in ranges]


def _segs_len(lengths: list[int]) -> list[dict]:
    out, start = [], 0
    for n in lengths:
        out.append({"start_frame": start, "end_frame": start + n - 1})
        start += n
    return out


# --------------------------------------------------------------------------- #
# Appendix F
# --------------------------------------------------------------------------- #
def test_appendix_f1_t1_and_t2_mae():
    r = t1_episode([10, 50, 90], [12, 48, 70], tau=5)
    assert r["matched"] == 2
    assert r["n_pred"] == 3 and r["n_gold"] == 3
    assert r["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["pairs"] == [[0, 0], [1, 1]]
    m = t2([10, 50, 90], [12, 48, 70], fps=30)
    assert m["tau"] == 10
    assert m["abs_errors"] == [2, 2]  # pairs (10, 12) and (50, 48); 90 and 70 differ by 20
    assert m["n_pairs"] == 2
    assert m["mae_frames"] == 2.0
    assert m["mae_seconds"] == pytest.approx(2 / 30, abs=1e-6)


def test_appendix_f2_tie_rule():
    assert match_boundaries([20, 24], [22], tau=5) == [(0, 0)]  # the pair (20, 22)


def test_appendix_f3_optimal_beats_greedy_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="robolabel.eval.temporal"):
        r = t1_episode([10, 14], [13, 18], tau=4, episode_key="F1/0")
    assert r["matched"] == 2
    assert r["pairs"] == [[0, 0], [1, 1]]  # (10, 13) and (14, 18)
    assert r["greedy_matched"] == 1  # greedy takes (14, 13), then nothing for 18
    assert r["greedy_disagrees"] is True
    assert any("disagreement" in rec.getMessage() and "F1/0" in rec.getMessage() for rec in caplog.records)
    micro = t1_micro([r])
    assert micro["matched"] == 2 and micro["greedy_matched"] == 1 and micro["greedy_disagreements"] == 1


def test_appendix_f4_t3_versus_legacy():
    gold = _segs((0, 9), (10, 19))
    pred = _segs((0, 19))
    assert t3_episode(pred, gold) == 0.25  # one pair at IoU 0.5, divided by max(2, 1)
    assert t3_legacy_episode(pred, gold) == 0.5
    assert len(t3_pairs(pred, gold)) == 1 and t3_pairs(pred, gold)[0][2] == 0.5


def test_appendix_f5_t4_cases():
    a = t4_episode(_segs_len([50, 50, 50, 50]), n_gold_segments=4)
    assert a["uniform"] and a["degenerate"] and not a["single_segment"]
    b = t4_episode(_segs((0, 99)), n_gold_segments=3)
    assert b["single_segment"] and b["degenerate"] and not b["uniform"]
    c = t4_episode(_segs_len([30, 70, 40]), n_gold_segments=3)
    assert not c["degenerate"] and not c["uniform"] and not c["single_segment"]
    rate = t4_rate([a, b, c])
    assert rate["degenerate"] == 2 and rate["rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert rate["single_segment"] == 1 and rate["uniform"] == 1


def test_appendix_f9_missing_output():
    gold = _segs((0, 9), (10, 19), (20, 29))
    pred = missing_output_segments(30)
    assert len(pred) == 1
    assert pred[0]["start_frame"] == 0 and pred[0]["end_frame"] == 29 and pred[0]["confidence"] == 0.5
    assert pred[0]["target"] is None and pred[0]["missing_output"] is True
    t4 = t4_episode(pred, len(gold))
    assert t4["degenerate"] is True and t4["single_segment"] is True
    r = t1_episode(boundaries(pred), boundaries(gold), tau=5)
    assert r["recall"] == 0.0 and r["matched"] == 0 and r["n_gold"] == 2 and r["n_pred"] == 0
    assert r["f1"] == 0.0  # exactly one of m, n is 0
    # the one segment overlaps each gold segment with IoU 10/30; one pair counts, divided by 3
    assert t3_episode(pred, gold) == pytest.approx((10 / 30) / 3, abs=1e-6)
    assert failed_spans_from_segments(pred) == []  # no failed attempts
    with pytest.raises(ValueError):
        missing_output_segments(0)


# --------------------------------------------------------------------------- #
# T1 matching properties
# --------------------------------------------------------------------------- #
def _brute_force(pred: list[int], gold: list[int], tau: int) -> list[tuple[int, int]]:
    """Enumerate every matching and pick it by the spec 4.1 rule."""
    best_key, best = None, []
    n = len(gold)
    options = [[None] + [i for i, p in enumerate(pred) if abs(p - g) <= tau] for g in gold]
    for choice in itertools.product(*options):
        used = [c for c in choice if c is not None]
        if len(used) != len(set(used)):
            continue
        pairs = [(c, j) for j, c in enumerate(choice) if c is not None]
        total = sum(abs(pred[i] - gold[j]) for i, j in pairs)
        lex = tuple(c if c is not None else len(pred) + 1 for c in choice)
        key = (-len(pairs), total, lex)
        if best_key is None or key < best_key:
            best_key, best = key, pairs
    assert n == len(options)
    return best


def test_match_boundaries_equals_brute_force_on_random_cases():
    rng = random.Random(7)
    for _ in range(400):
        pred = [rng.randint(0, 30) for _ in range(rng.randint(0, 4))]
        gold = sorted(rng.randint(0, 30) for _ in range(rng.randint(0, 4)))
        tau = rng.choice([0, 2, 3, 5])
        assert match_boundaries(pred, gold, tau) == _brute_force(pred, gold, tau), (pred, gold, tau)


def test_greedy_count_equals_legacy_boundary_pr_mae():
    from robolabel.metrics import boundary_pr_mae

    rng = random.Random(11)
    for _ in range(300):
        pred = [rng.randint(0, 40) for _ in range(rng.randint(0, 6))]
        gold = [rng.randint(0, 40) for _ in range(rng.randint(0, 6))]
        tau = rng.choice([3, 5, 10])
        assert greedy_match_count(pred, gold, tau) == boundary_pr_mae(pred, gold, tol=tau)["matched"]
        assert greedy_match_count(pred, gold, tau) <= len(match_boundaries(pred, gold, tau))


def test_t1_episode_empty_conventions_and_micro_zero_denominators():
    both_empty = t1_episode([], [], tau=5)
    assert (both_empty["precision"], both_empty["recall"], both_empty["f1"]) == (1.0, 1.0, 1.0)
    one_empty = t1_episode([10], [], tau=5)
    assert one_empty["f1"] == 0.0
    micro = t1_micro([both_empty])
    assert micro["f1"] == 0.0 and micro["precision"] == 0.0
    assert micro["zero_denominators"] == ["precision", "recall", "f1"]
    micro2 = t1_micro([t1_episode([10, 50, 90], [12, 48, 70], 5), t1_episode([5], [7, 30], 5)])
    # matched 2 + 1 = 3, predicted 3 + 1 = 4, gold 3 + 2 = 5
    assert micro2["precision"] == 0.75 and micro2["recall"] == 0.6
    assert micro2["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35, abs=1e-6)
    assert micro2["zero_denominators"] == []
    with pytest.raises(ValueError):
        t1_micro([t1_episode([1], [1], 5), t1_episode([1], [1], 10)])
    with pytest.raises(ValueError):
        match_boundaries([1], [1], -1)


def test_boundaries_accept_start_end_aliases():
    view = [{"start": 0, "end": 10}, {"start": 11, "end": 20}, {"start": 21, "end": 30}]
    assert boundaries(view) == [10, 20]
    assert boundaries(_segs((0, 30))) == []


# --------------------------------------------------------------------------- #
# T2 to T6
# --------------------------------------------------------------------------- #
def test_t2_near_is_capped_at_one_second():
    m = t2([12], [10, 100], fps=30)
    assert m["near_cap_frames"] == 30
    assert m["near_frames"] == [2, 30]  # 100 is 88 frames from 12, capped at 30
    assert m["near_median_frames"] == 16.0
    assert m["near_median_seconds"] == pytest.approx(16 / 30, abs=1e-6)
    none = t2([], [10, 100], fps=10)
    assert none["near_frames"] == [10, 10] and none["near_median_seconds"] == 1.0
    assert none["mae_frames"] is None
    pooled = t2_summary([m, none])
    assert pooled["n_gold_boundaries"] == 4
    assert pooled["near_median_frames"] == 10.0  # median of [2, 30, 10, 10]
    assert pooled["mae_frames"] == 2.0 and pooled["n_pairs"] == 1
    with pytest.raises(ValueError):
        t2([1], [1], fps=0)


def test_t3_summary_and_legacy_pooling():
    s = t3_summary([0.25, 0.5, 1.0, 0.75])
    assert s["mean"] == 0.625 and s["median"] == 0.625
    assert s["p10"] == pytest.approx(0.325, abs=1e-6)  # numpy linear: 0.25 + 0.3 * 0.25
    assert t3_summary([])["mean"] is None
    ious = [t3_legacy_ious(_segs((0, 19)), _segs((0, 9), (10, 19))), [1.0, 0.0]]
    assert t3_legacy_flat_mean(ious) == 0.5


def test_t3_optimal_assignment_prefers_total_iou():
    gold = _segs((0, 9), (10, 19), (20, 29))
    pred = _segs((0, 11), (12, 29))
    # pairs: pred0-gold0 IoU 10/12, pred1-gold2 IoU 10/18; pred1-gold1 IoU 8/20 is smaller
    assert t3_episode(pred, gold) == pytest.approx((10 / 12 + 10 / 18) / 3, abs=1e-6)


def test_t4_matches_legacy_uniform_split():
    from robolabel.gate import is_uniform_split

    rng = random.Random(3)
    for _ in range(200):
        lengths = [rng.randint(1, 60) for _ in range(rng.randint(1, 6))]
        segs = _segs_len(lengths)
        assert t4_episode(segs, 3)["uniform"] == is_uniform_split(segs, 0.12, 3)


def test_t5():
    assert t5_episode(7, 5) == 2
    s = t5_summary([2, -1, 0, -3])
    assert s["mean"] == -0.5 and s["mean_abs"] == 1.5 and s["share_abs_ge_2"] == 0.5
    assert t5_summary([])["mean"] is None


def test_t6_uses_coarse_boundaries():
    gold = [{"coarse_idx": 0, "start_frame": 0, "end_frame": 97}, {"coarse_idx": 1, "start_frame": 98,
                                                                    "end_frame": 200}]
    pred = [{"start": 0, "end": 100}, {"start": 101, "end": 200}]
    r = t6_episode(pred, gold, tau=5)
    assert r["matched"] == 1 and r["f1"] == 1.0


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #
def test_match_spans_one_to_one_max_total_iou():
    gold = [[10, 19], [30, 39]]
    pred = [[10, 29], [12, 19], [30, 37]]
    pairs = match_spans(pred, gold, 0.3)
    # pred1-gold0 IoU 0.8 beats pred0-gold0 IoU 0.5; pred2-gold1 IoU 0.8; pred0 stays unmatched
    assert pairs == [(1, 0, 0.8), (2, 1, 0.8)]
    assert match_spans([[0, 9]], [[7, 16]], 0.3) == []  # IoU 3/17 below 0.3
    assert match_spans([[0, 9]], [[7, 9]], 0.3) == [(0, 0, 0.3)]  # IoU exactly 0.3 counts
    assert match_spans([{"span": [5, 9]}], [{"start_frame": 5, "end_frame": 9}], 0.3) == [(0, 0, 1.0)]
    assert match_spans([], gold, 0.3) == []


def test_failed_spans_from_segments():
    segs = [
        {"start_frame": 0, "end_frame": 9, "outcome": "success"},
        {"start_frame": 10, "end_frame": 19, "outcome": "failed"},
        {"start_frame": 20, "end_frame": 29, "outcome": "success", "mistake": True},
        {"start_frame": 30, "end_frame": 39, "outcome": "success", "mistake": "false"},
        {"start": 40, "end": 49, "outcome": "failed"},
    ]
    assert failed_spans_from_segments(segs) == [[10, 29], [40, 49]]
    assert failed_spans_from_segments(missing_output_segments(50)) == []


# --------------------------------------------------------------------------- #
# K1 to K3
# --------------------------------------------------------------------------- #
def _receipts() -> list[dict]:
    usage = {"input_text_tokens": 100, "input_image_tokens": 1000, "input_video_tokens": 0,
             "input_audio_tokens": 0, "cached_tokens": 0, "output_tokens": 50, "reasoning_tokens": 20}
    return [
        {"episode_key": "F1/0", "step": "segments", "usd": 0.01, "usd_batch_eq": 0.005, "usage": usage,
         "latency_s": 2.0, "wall_s": 2.5, "cache_hit": False},
        {"episode_key": "F1/0", "step": "goal", "usd": 0.02, "usd_batch_eq": 0.01,
         "usage": {**usage, "output_tokens": 70, "cached_tokens": 40}, "latency_s": 4.0, "cache_hit": True},
        {"inputs": {"episode_key": "F1/10"}, "step": "segments", "usd": 0.03,
         "usage": {"input_text_tokens": 300, "output_tokens": 30}, "latency_s": 6.0, "cache_hit": False},
        {"episode_key": None, "step": "preflight", "usd": 0.5, "usd_batch_eq": 0.25, "usage": {},
         "latency_s": None, "cache_hit": False},
    ]


def test_k1_dollars():
    r = cost_metrics.k1(_receipts())
    assert [e["episode_key"] for e in r["per_episode"]] == ["F1/0", "F1/10"]
    assert [e["usd"] for e in r["per_episode"]] == [0.03, 0.03]
    assert r["mean_usd_per_episode"] == 0.03 and r["p50_usd_per_episode"] == 0.03
    assert r["usd_per_1000_episodes"] == 30.0
    # F1/0 batch 0.005 + 0.01; F1/10 has no usd_batch_eq so its actual 0.03 is used
    assert r["usd_batch_eq_per_1000_episodes"] == 22.5
    assert r["n_receipts_without_batch_eq"] == 1
    assert r["unattributed_usd"] == 0.5 and r["n_unattributed_receipts"] == 1
    assert r["by_step"]["segments"]["total_usd"] == 0.04 and r["by_step"]["goal"]["n_calls"] == 1
    assert r["n_cache_hits"] == 1
    no_hits = cost_metrics.k1(_receipts(), include_cache_hits=False)
    assert no_hits["n_episodes"] == 2 and no_hits["usd_per_1000_episodes"] == 20.0
    assert no_hits["n_cache_hits"] == 1 and no_hits["n_receipts"] == 3 and no_hits["cache_hits_included"] is False
    listed = cost_metrics.k1(_receipts(), episode_keys=["F1/0", "F1/10", "F1/2"])
    assert listed["n_episodes"] == 3 and listed["per_episode"][1] == {"episode_key": "F1/2", "usd": 0.0,
                                                                       "usd_batch_eq": 0.0}
    assert listed["usd_per_1000_episodes"] == 20.0


def test_k2_tokens():
    r = cost_metrics.k2(_receipts())
    first = r["per_episode"][0]
    assert first["episode_key"] == "F1/0"
    assert first["input_text"] == 200 and first["input_image"] == 2000 and first["output"] == 120
    assert first["cached"] == 40 and first["reasoning"] == 40
    second = r["per_episode"][1]
    assert second["input_text"] == 300 and second["input_image"] == 0 and second["output"] == 30
    assert r["total"]["input_text"] == 500 and r["mean_per_episode"]["input_text"] == 250.0
    assert r["p50_per_episode"]["output"] == 75.0
    assert cost_metrics.k2(_receipts(), include_cache_hits=False)["total"]["output"] == 80


def test_k3_latency():
    r = cost_metrics.k3(_receipts(), {"F1/0": 10.0, "F1/10": 20.0})
    assert r["n_calls_with_latency"] == 3
    assert r["latency_p50_s"] == 4.0
    assert r["latency_p95_s"] == 5.8  # numpy linear: 4 + 0.9 * 2
    assert r["episode_wall_p50_s"] == 15.0 and r["episode_wall_p95_s"] == 19.5
    assert r["by_step"]["segments"]["latency_p50_s"] == 4.0
    assert r["call_wall_p50_s"] == 2.5
    no_hits = cost_metrics.k3(_receipts(), None, include_cache_hits=False)
    assert no_hits["latency_p50_s"] == 4.0 and no_hits["n_calls_with_latency"] == 2
    assert no_hits["episode_wall_p50_s"] is None


# --------------------------------------------------------------------------- #
# X1 to X3
# --------------------------------------------------------------------------- #
_GOLD_PHASES = [
    {"start_frame": 0, "end_frame": 9, "phase_class": "approach"},
    {"start_frame": 10, "end_frame": 19, "phase_class": "grasp"},
    {"start_frame": 20, "end_frame": 29, "phase_class": "transport"},
    {"start_frame": 30, "end_frame": 39, "phase_class": "release"},
    {"start_frame": 40, "end_frame": 49, "phase_class": "retract"},
]


def test_x1_candidate_recall_and_early_read():
    # constrained gold boundaries: 9 (approach->grasp), 29 (transport->release), 39 (release->retract)
    ep = component.x1_episode([8, {"candidate_id": "c2", "frame": 21}, 60], _GOLD_PHASES)
    assert ep["n_gold_constrained"] == 3 and ep["n_gold_all"] == 4
    assert ep["recall_matched"] == 1  # 8 matches 9; 21 is 8 frames from 29
    assert ep["precision_matched"] == 2  # 8 -> 9 and 21 -> 19 against all gold boundaries
    s = component.x1_summary([ep])
    assert (s["numerator"], s["denominator"]) == (1, 3) and s["value"] == pytest.approx(1 / 3, abs=1e-6)
    assert s["precision"]["label"] == "not constrained" and s["precision"]["value"] == pytest.approx(2 / 3, abs=1e-6)

    legacy = [{"start_frame": g["start_frame"], "end_frame": g["end_frame"], "subtask_text": f"{g['phase_class']} it"}
              for g in _GOLD_PHASES]
    early = component.x1_early_read({"F1/3": ([8, 21, 60], legacy)},
                                    phase_of=lambda seg: seg["subtask_text"].split()[0])
    assert early["label"] == "early read on legacy gold, not acceptance"
    assert early["numerator"] == 1 and early["denominator"] == 3


def test_x2_robot_end_state():
    gold = [
        {"req_id": "r1", "kind": "robot_end_state", "predicate": "holding", "ref_object": None, "value": False,
         "achieved": True, "visibility": {"up": "visible", "side": "partial"}},
        {"req_id": "r2", "kind": "robot_end_state", "predicate": "gripper_open", "ref_object": None, "value": True,
         "achieved": False, "visibility": {"up": "partial", "side": "visible"}},  # actually closed at the end
        {"req_id": "r3", "kind": "robot_end_state", "predicate": "withdrawn", "ref_object": None, "value": True,
         "achieved": True, "visibility": {"up": "partial", "side": "not_visible"}},
        {"req_id": "r4", "kind": "object_end_state", "object": "o1", "predicate": "inside", "ref_object": "o2",
         "value": True, "achieved": True, "visibility": {"up": "visible"}},
        {"req_id": "r5", "kind": "robot_end_state", "predicate": "at_home_pose", "ref_object": None, "value": True,
         "achieved": True, "visibility": [{"camera": "up", "class": "visible"}]},
    ]
    signal = [
        {"predicate": "holding", "ref_object": "none", "value": False, "basis": "signal"},
        {"predicate": "gripper_open", "ref_object": None, "value": True},
        {"predicate": "withdrawn", "ref_object": None, "value": True},
    ]
    ep = component.x2_episode(signal, gold)
    assert (ep["correct"], ep["total"]) == (1, 3)  # r1 right, r2 wrong, r5 has no signal item
    assert ep["missing_signal_item"] == 1 and ep["excluded_not_visible"] == 1
    assert [i["req_id"] for i in ep["items"]] == ["r1", "r2", "r5"]
    s = component.x2_summary([ep])
    assert (s["numerator"], s["denominator"]) == (1, 3)
    held = [{"req_id": "r6", "kind": "robot_end_state", "predicate": "holding", "ref_object": "o3", "value": True,
             "achieved": True, "visibility": {"up": "visible"}}]
    assert component.x2_episode([{"predicate": "holding", "ref_object": None, "value": True}], held)["correct"] == 0


def test_x3_failed_grasp_spans():
    gold = [
        {"span": [10, 20], "failure_type": "missed_grasp"},
        {"span": [50, 60], "failure_type": "wrong_object"},
        {"span": [80, 90], "failure_type": "slip"},
    ]
    attempts = [
        {"start": 12, "end": 22, "outcome": "empty", "source": "signal"},
        {"start": 30, "end": 40, "outcome": "hold", "source": "signal"},
        {"start": 52, "end": 58, "outcome": "empty", "source": "signal"},
        {"start": 80, "end": 90, "outcome": "failed", "failure_type": "slip", "source": "vlm"},
        {"start": 200, "end": 210, "outcome": "drop"},
    ]
    spans = component.signal_failed_spans(attempts)
    assert spans == [[12, 22], [52, 58], [200, 210]]
    ep = component.x3_episode(spans, gold)
    assert (ep["matched"], ep["n_pred"], ep["n_gold"]) == (1, 3, 2)
    assert ep["pred_matching_excluded_types"] == 1
    s = component.x3_summary([ep])
    assert (s["numerator"], s["denominator"], s["value"]) == (2, 5, 0.4)
    assert s["precision"]["value"] == pytest.approx(1 / 3, abs=1e-6) and s["recall"]["value"] == 0.5
    empty = component.x3_summary([component.x3_episode([], [])])
    assert empty["value"] == 0.0 and empty["zero_denominator"] is True


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def _everything() -> str:
    gold = _segs((0, 9), (10, 19), (20, 29))
    pred = _segs((0, 11), (12, 25), (26, 29))
    out = {
        "t1": t1_micro([t1_episode(boundaries(pred), boundaries(gold), 4), t1_episode([10, 14], [13, 18], 4)]),
        "t2": t2(boundaries(pred), boundaries(gold), 30),
        "t3": t3_summary([t3_episode(pred, gold), t3_episode(missing_output_segments(30), gold)]),
        "t3_pairs": t3_pairs(pred, gold),
        "t4": t4_rate([t4_episode(pred, 3), t4_episode(missing_output_segments(30), 3)]),
        "t5": t5_summary([t5_episode(3, 3), t5_episode(1, 3)]),
        "spans": match_spans([[0, 9], [12, 25]], [[0, 11], [10, 19]], 0.3),
        "k1": cost_metrics.k1(_receipts()),
        "k2": cost_metrics.k2(_receipts()),
        "k3": cost_metrics.k3(_receipts(), {"F1/0": 10.0, "F1/10": 20.0}),
        "x1": component.x1_summary([component.x1_episode([8, 21], _GOLD_PHASES)]),
        "x3": component.x3_summary([component.x3_episode([[12, 22]], [{"span": [10, 20], "failure_type": "slip"}])]),
    }
    return json.dumps(out, sort_keys=True)


def test_determinism_same_input_same_bytes():
    assert _everything() == _everything()


def test_module_constants():
    assert temporal.T2_TAU == 10 and temporal.UNIFORM_CV_THRESHOLD == 0.12
    assert component.X1_TAU == 5 and component.X3_MIN_IOU == 0.3


# --------------------------------------------------------------------------- #
# Added by the verification pass: ties, empty inputs, single segments, integration shapes
# --------------------------------------------------------------------------- #
def test_tie_rule_lowest_gold_index_then_lowest_pred_index():
    # one prediction halfway between two gold boundaries: the lower gold index takes it
    assert match_boundaries([22], [20, 24], tau=5) == [(0, 0)]
    # two predictions equally far from one gold boundary: the lower predicted index wins
    assert match_boundaries([10, 20], [15], tau=5) == [(0, 0)]
    assert match_boundaries([20, 10], [15], tau=5) == [(0, 0)]  # index, not frame value
    # the pair count comes before the error sum: (10, 13) + (14, 18) beats (14, 13) alone
    assert match_boundaries([10, 14], [13, 18], tau=4) == [(0, 0), (1, 1)]
    # the error sum comes before the index rule: gold 0 takes pred 1 (error 0), not pred 0 (error 3)
    assert match_boundaries([7, 10], [10], tau=5) == [(1, 0)]
    # tau = 0 needs exact frames; tau is inclusive
    assert match_boundaries([10, 11], [11], tau=0) == [(1, 0)]
    assert match_boundaries([10], [15], tau=5) == [(0, 0)]
    assert match_boundaries([10], [16], tau=5) == []


def test_match_boundaries_brute_force_unsorted_gold_and_duplicates():
    rng = random.Random(2026)
    for _ in range(400):
        pred = [rng.randint(0, 25) for _ in range(rng.randint(0, 5))]
        gold = [rng.randint(0, 25) for _ in range(rng.randint(0, 4))]  # any order, repeats allowed
        tau = rng.choice([0, 1, 5, 10])
        assert match_boundaries(pred, gold, tau) == _brute_force(pred, gold, tau), (pred, gold, tau)


def test_t1_empty_boundaries_per_episode_and_micro():
    no_pred = t1_episode([], [10, 20], tau=5)
    assert (no_pred["precision"], no_pred["recall"], no_pred["f1"]) == (0.0, 0.0, 0.0)
    no_gold = t1_episode([10], [], tau=5)
    assert (no_gold["precision"], no_gold["recall"], no_gold["f1"]) == (0.0, 0.0, 0.0)
    micro = t1_micro([no_pred])
    assert micro["n_pred"] == 0 and micro["n_gold"] == 2 and micro["recall"] == 0.0
    assert micro["zero_denominators"] == ["precision", "f1"]
    nothing = t1_micro([])
    assert nothing["n_episodes"] == 0 and nothing["tau"] is None and nothing["f1"] == 0.0
    # both sides non-empty but nothing within tau: a real 0, not a 0 / 0
    miss = t1_micro([t1_episode([10], [50], tau=5)])
    assert miss["f1"] == 0.0 and miss["zero_denominators"] == []
    # summaries accept any iterable, including a one-shot generator
    gen = t1_micro(t1_episode(p, g, 5) for p, g in [([10], [12]), ([30], [40])])
    assert (gen["matched"], gen["n_pred"], gen["n_gold"], gen["n_episodes"]) == (1, 2, 2, 2)


def test_single_segment_cases():
    one = _segs((0, 29))
    assert boundaries(one) == []
    assert t3_episode(one, one) == 1.0
    assert t3_legacy_episode(one, one) == 1.0
    single_vs_single = t4_episode(one, n_gold_segments=1)
    assert not single_vs_single["degenerate"]  # gold has fewer than 2 segments
    r = t1_episode(boundaries(one), boundaries(one), tau=5)
    assert r["f1"] == 1.0 and r["matched"] == 0  # m = n = 0
    assert t6_episode(one, one, tau=5)["f1"] == 1.0
    # two uniform segments are not checked for uniformity (fewer than 3)
    assert not t4_episode(_segs_len([10, 10]), 2)["uniform"]
    # three equal segments are uniform whatever the gold count
    assert t4_episode(_segs_len([10, 10, 10]), 3)["uniform"]


def test_empty_inputs_everywhere():
    gold = _segs((0, 9), (10, 19))
    assert t3_episode([], gold) == 0.0
    assert t3_pairs([], gold) == []
    assert t3_legacy_episode([], gold) is None
    assert t3_legacy_flat_mean([]) is None
    no_pred = t4_episode([], 2)
    assert no_pred["n_pred"] == 0 and not no_pred["degenerate"]  # the missing-output rule runs first
    assert t4_rate([])["rate"] is None
    e = t2([], [], fps=30)
    assert e["n_pairs"] == 0 and e["near_frames"] == [] and e["near_median_frames"] is None
    assert t2_summary([])["mae_frames"] is None
    assert t5_summary([])["share_abs_ge_2"] is None
    assert match_spans([], []) == [] and match_spans([[0, 9]], []) == []
    assert failed_spans_from_segments([]) == []
    assert boundaries([]) == []
    k1 = cost_metrics.k1([])
    assert k1["n_episodes"] == 0 and k1["mean_usd_per_episode"] is None and k1["total_usd"] == 0.0
    assert cost_metrics.k2([])["mean_per_episode"]["output"] is None
    k3 = cost_metrics.k3([], None)
    assert k3["latency_p50_s"] is None and k3["episode_wall_p95_s"] is None
    x1 = component.x1_summary([])
    assert x1["value"] == 0.0 and x1["zero_denominator"] is True
    assert component.x1_episode([], [])["n_gold_all"] == 0
    x2 = component.x2_summary([component.x2_episode([], [])])
    assert (x2["numerator"], x2["denominator"], x2["zero_denominator"]) == (0, 0, True)
    assert component.signal_failed_spans([]) == []


def test_t2_near_cap_follows_fps_and_numpy_inputs_serialize():
    np = pytest.importorskip("numpy")
    m = t2([np.int64(40)], [np.int64(10)], fps=np.int64(20))
    assert m["near_cap_frames"] == 20 and m["near_frames"] == [20]  # 30 frames away, capped at 1 s
    assert m["near_median_seconds"] == 1.0 and m["fps"] == 20.0
    json.dumps(m)  # numpy fps and frames must not leak into the output


def test_spans_accept_numpy_arrays():
    np = pytest.importorskip("numpy")
    assert temporal.as_span(np.array([3, 9])) == (3, 9)
    assert temporal.as_span({"span": np.array([3, 9])}) == (3, 9)
    assert temporal.as_span("ab") is None and temporal.as_span([1, 2, 3]) is None
    assert match_spans([np.array([0, 9])], [{"span": np.array([0, 9])}]) == [(0, 0, 1.0)]


def test_match_spans_maximizes_total_iou_not_pair_count():
    # pred 0 = gold 0 exactly (IoU 1.0). Pred 0 with gold 1 (4/13) plus pred 1 with gold 0 (3/10,
    # exactly the threshold) makes two pairs but less total IoU (0.608 < 1.0); pred 1 and gold 1
    # do not overlap. Spec 4.2 S4 maximizes total IoU, so one pair is kept.
    gold = [[0, 9], [6, 12]]
    pred = [[0, 9], [3, 5]]
    assert temporal.span_iou((0, 9), (6, 12)) == pytest.approx(4 / 13)
    assert temporal.span_iou((3, 5), (0, 9)) == 0.3
    assert temporal.span_iou((3, 5), (6, 12)) == 0.0
    assert match_spans(pred, gold, 0.3) == [(0, 0, 1.0)]
    # widen pred 0 to [0, 12]: alone 10/13 = 0.769 < 7/13 + 3/10 = 0.838, so both pairs are taken
    assert match_spans([[0, 12], [3, 5]], gold, 0.3) == [(1, 0, 0.3), (0, 1, 0.538462)]


def test_signal_failed_spans_read_raw_l1_attempts(caplog):
    # raw L1 attempts (layers/signal.py) have closing_onset / event_frame / end_frame, no start or end
    raw = [
        {"attempt_idx": 1, "closing_onset": 40, "closing_offset": 45, "event_frame": 52, "end_frame": 80,
         "outcome": "empty", "failure_type": "missed_grasp"},
        {"attempt_idx": 2, "closing_onset": 90, "closing_offset": 95, "event_frame": 96, "end_frame": 150,
         "outcome": "hold", "failure_type": "none"},
        {"attempt_idx": 3, "closing_onset": 160, "closing_offset": 165, "event_frame": 190, "end_frame": 200,
         "outcome": "slip", "failure_type": "slip"},
        {"attempt_idx": 4, "closing_onset": 210, "closing_offset": 215, "event_frame": 230, "end_frame": 240,
         "outcome": "aborted", "failure_type": "aborted"},
    ]
    assert component.signal_failed_spans(raw) == [[40, 52], [160, 190]]
    # a failed attempt with no frames at all is skipped with a warning, never silently
    with caplog.at_level(logging.WARNING, logger="robolabel.eval.component"):
        assert component.signal_failed_spans([{"attempt_idx": 9, "outcome": "empty"}]) == []
    assert any("without a usable span" in rec.getMessage() for rec in caplog.records)
    ep = component.x3_episode(component.signal_failed_spans(raw), [{"span": [40, 55], "failure_type": "missed_grasp"}])
    assert (ep["matched"], ep["n_pred"], ep["n_gold"]) == (1, 2, 1)


def _req(predicate, value, achieved=True, ref=None, vis=None, kind="robot_end_state"):
    return {"req_id": f"r_{predicate}", "kind": kind, "predicate": predicate, "ref_object": ref, "value": value,
            "status": "required", "achieved": achieved, "visibility": vis or {"up": "visible"}}


def test_x2_gripper_items_answer_each_other_and_unknown_gold_facts():
    # L1 states the gripper with one item; gripper_closed = true answers gold gripper_open
    closed = [{"predicate": "gripper_closed", "ref_object": "none", "value": True, "basis": "signal"}]
    # gold: gripper_open required but not achieved, so the gripper ended closed: signal is right
    ep = component.x2_episode(closed, [_req("gripper_open", True, achieved=False)])
    assert (ep["correct"], ep["total"], ep["missing_signal_item"]) == (1, 1, 0)
    assert ep["items"][0]["signal_predicate"] == "gripper_closed" and ep["items"][0]["signal_value"] is False
    # gold gripper_open achieved: the signal (closed) is wrong, but not missing
    ep = component.x2_episode(closed, [_req("gripper_open", True)])
    assert (ep["correct"], ep["total"], ep["missing_signal_item"]) == (0, 1, 0)
    # the same predicate on both sides is used first
    both = closed + [{"predicate": "gripper_open", "ref_object": None, "value": True}]
    assert component.x2_episode(both, [_req("gripper_open", True)])["correct"] == 1
    # a null gold value is unknown, not a wrong signal item
    ep = component.x2_episode([{"predicate": "withdrawn", "ref_object": "none", "value": True}],
                              [_req("withdrawn", None)])
    assert (ep["total"], ep["excluded_unknown"]) == (0, 1)
    # achieved "unknown" is unknown too
    ep = component.x2_episode([{"predicate": "withdrawn", "value": True}], [_req("withdrawn", True, "unknown")])
    assert (ep["total"], ep["excluded_unknown"]) == (0, 1)
    # holding o3 not achieved: holding nothing or holding another object, cannot tell
    ep = component.x2_episode([{"predicate": "holding", "ref_object": "none", "value": False}],
                              [_req("holding", True, achieved=False, ref="o3")])
    assert (ep["total"], ep["excluded_unknown"]) == (0, 1)
    # holding nothing not achieved: the robot holds something, which the canonical signal form states
    ep = component.x2_episode([{"predicate": "holding", "ref_object": "none", "value": True}],
                              [_req("holding", False, achieved=False)])
    assert (ep["correct"], ep["total"]) == (1, 1)
    # use_achieved=False compares with the gold value as written
    ep = component.x2_episode(closed, [_req("gripper_open", True, achieved=False)], use_achieved=False)
    assert (ep["correct"], ep["total"]) == (0, 1)
    # object end states and items with no visible camera are not X2 items
    ep = component.x2_episode(closed, [_req("inside", True, ref="o2", kind="object_end_state"),
                                       _req("gripper_open", True, vis={"up": "partial", "side": "not_visible"})])
    assert (ep["total"], ep["excluded_not_visible"]) == (0, 1)


def test_k2_reports_unknown_token_counts_and_cost_accepts_generators():
    r = cost_metrics.k2(_receipts())
    # receipt 3 knows only text and output tokens; the preflight receipt knows none
    assert r["n_receipts_unknown"]["input_image"] == 2 and r["n_receipts_unknown"]["input_text"] == 1
    assert r["n_receipts_unknown"]["output"] == 1
    assert cost_metrics.k1(iter(_receipts()))["usd_per_1000_episodes"] == 30.0
    assert cost_metrics.k2(iter(_receipts()))["total"]["input_text"] == 500
    assert cost_metrics.k3(iter(_receipts()), None)["n_calls_with_latency"] == 3


def test_x1_ignores_boundaries_outside_the_three_transitions():
    gold = [
        {"start_frame": 0, "end_frame": 9, "phase_class": "approach"},
        {"start_frame": 10, "end_frame": 19, "phase_class": "press"},
        {"start_frame": 20, "end_frame": 29, "phase_class": "retract"},
    ]
    ep = component.x1_episode([9, 19], gold)
    assert ep["n_gold_constrained"] == 0 and ep["n_gold_all"] == 2
    assert ep["recall_matched"] == 0 and ep["precision_matched"] == 2
    s = component.x1_summary([ep])
    assert s["zero_denominator"] is True and s["precision"]["value"] == 1.0


_DETERMINISM_SCRIPT = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("t", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.stdout.write(mod._everything())
"""


def test_determinism_across_hash_seeds():
    import os
    import subprocess
    import sys

    outs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([sys.executable, "-c", _DETERMINISM_SCRIPT, __file__], env=env,
                              capture_output=True, text=True, check=True)
        outs.append(proc.stdout)
    assert outs[0] == outs[1] and outs[0] == _everything()
