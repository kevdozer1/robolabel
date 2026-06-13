"""Tests for the annotation-strategy layer: schema validation, the detectors,
the strategy presets/configs, and offline S0..S4 segmentation with the mock.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robolabel.demo import synthetic_episode
from robolabel.gate import (
    is_degenerate_single_segment,
    is_uniform_split,
    run_gate,
)
from robolabel.labelers.segmentation import (
    SchemaValidationError,
    sample_frames,
    segment_episode,
    validate_grounded_segments,
)
from robolabel.providers import build_provider
from robolabel.rubric import load_rubric
from robolabel.schema import (
    EpisodeAnnotation,
    EpisodeMetadata,
    SubtaskSegment,
    episode_records,
    read_annotations,
    write_annotations,
)
from robolabel.strategy import PRESETS, load_strategy


# --------------------------------------------------------------------------- #
# Strategy presets / configs
# --------------------------------------------------------------------------- #
def test_presets_are_cumulative():
    s0, s1, s2, s3, s4 = (PRESETS[k] for k in ("S0", "S1", "S2", "S3", "S4"))
    assert s0.is_baseline and not s1.is_baseline
    assert s1.grounded and not s1.closed_vocabulary
    assert s2.grounded and s2.closed_vocabulary and s2.enforce_min_segments
    assert s3.refine_boundaries and not s2.refine_boundaries
    assert s4.self_consistency_k == 3 and s3.self_consistency_k == 1


def test_load_strategy_by_name_and_unknown():
    assert load_strategy("S2").name == "S2"
    assert load_strategy(None).name == "S0"
    assert load_strategy(PRESETS["S3"]).name == "S3"
    with pytest.raises(ValueError):
        load_strategy("S9")


def test_load_strategy_from_json(tmp_path: Path):
    p = tmp_path / "strat.json"
    p.write_text('{"base": "S2", "name": "S2-wide", "frame_count": 20}', encoding="utf-8")
    cfg = load_strategy(str(p))
    assert cfg.name == "S2-wide"
    assert cfg.frame_count == 20
    assert cfg.grounded and cfg.closed_vocabulary  # inherited from S2


def test_strategy_provenance_roundtrips():
    prov = PRESETS["S4"].provenance()
    assert prov["strategy"]["name"] == "S4"
    assert prov["strategy"]["self_consistency_k"] == 3


# --------------------------------------------------------------------------- #
# Grounded schema validation
# --------------------------------------------------------------------------- #
_RUBRIC = load_rubric()
_S1 = PRESETS["S1"]
_S2 = PRESETS["S2"]


def test_validate_grounded_good_case():
    raw = {"segments": [
        {"phase": "approach", "target": "red brick", "end_frame": 10, "evidence": "gripper above brick", "subtask_text": "move to brick"},
        {"phase": "grasp", "target": "red brick", "end_frame": 20, "evidence": "gripper closes", "subtask_text": "grasp brick"},
        {"phase": "transport", "target": "the box", "end_frame": 30, "evidence": "brick lifted", "subtask_text": "carry brick"},
    ]}
    segs = validate_grounded_segments(raw, 31, _S2, _RUBRIC)
    assert [s.end_frame for s in segs] == [10, 20, 30]
    assert segs[0].start_frame == 0 and segs[-1].end_frame == 30
    assert all(s.evidence for s in segs)
    assert [s.phase for s in segs] == ["approach", "grasp", "transport"]
    assert [s.target for s in segs] == ["red brick", "red brick", "the box"]


def test_validate_grounded_missing_evidence_raises():
    raw = {"segments": [{"phase": "approach", "end_frame": 10, "subtask_text": "x"},
                        {"phase": "grasp", "end_frame": 20, "subtask_text": "y"}]}
    with pytest.raises(SchemaValidationError):
        validate_grounded_segments(raw, 21, _S1, _RUBRIC)


def test_validate_grounded_non_integer_end_frame_raises():
    raw = {"segments": [{"phase": "approach", "end_frame": "soon", "evidence": "e", "subtask_text": "x"}]}
    with pytest.raises(SchemaValidationError):
        validate_grounded_segments(raw, 21, _S1, _RUBRIC)


def test_validate_grounded_degenerate_single_segment_raises_when_enforced():
    raw = {"segments": [{"phase": "other", "target": "the workpiece", "end_frame": 30,
                         "evidence": "did the task", "subtask_text": "complete"}]}
    with pytest.raises(SchemaValidationError):
        validate_grounded_segments(raw, 31, _S2, _RUBRIC)  # S2 enforces min segments
    # S1 does not enforce granularity, so a single grounded segment is allowed.
    segs = validate_grounded_segments(raw, 31, _S1, _RUBRIC)
    assert len(segs) == 1


def test_validate_grounded_unknown_phase_coerced_to_other():
    raw = {"segments": [
        {"phase": "teleport", "target": "red cube", "end_frame": 10, "evidence": "e", "subtask_text": "x"},
        {"phase": "grasp", "target": "red cube", "end_frame": 20, "evidence": "e", "subtask_text": "y"},
        {"phase": "retract", "end_frame": 30, "evidence": "e", "subtask_text": "z"},  # retract: target 'none' allowed
    ]}
    segs = validate_grounded_segments(raw, 31, _S2, _RUBRIC)
    assert segs[0].phase == "other"
    assert segs[-1].phase == "retract" and segs[-1].target is None


def test_require_target_rejects_empty_non_retract():
    # require_target (S2/S3/S4) rejects a non-retract phase whose target is missing/blank.
    for bad in (None, "", "  ", "none", "the scene", "object"):
        raw = {"segments": [
            {"phase": "approach", "target": bad, "end_frame": 10, "evidence": "e", "subtask_text": "x"},
            {"phase": "grasp", "target": "red cube", "end_frame": 20, "evidence": "e", "subtask_text": "y"},
            {"phase": "transport", "target": "the box", "end_frame": 30, "evidence": "e", "subtask_text": "z"},
        ]}
        with pytest.raises(SchemaValidationError, match="target"):
            validate_grounded_segments(raw, 31, _S2, _RUBRIC)


def test_require_target_allows_none_for_retract():
    # 'retract' is the one phase where a missing/"none" target is acceptable.
    raw = {"segments": [
        {"phase": "approach", "target": "red cube", "end_frame": 10, "evidence": "e", "subtask_text": "x"},
        {"phase": "transport", "target": "the box", "end_frame": 20, "evidence": "e", "subtask_text": "y"},
        {"phase": "retract", "target": "none", "end_frame": 30, "evidence": "e", "subtask_text": "z"},
    ]}
    segs = validate_grounded_segments(raw, 31, _S2, _RUBRIC)
    assert segs[-1].phase == "retract" and segs[-1].target is None


def test_terminal_phase_dedupe_collapses_double_retract():
    # The graded "two retract steps" error: consecutive identical trailing phases collapse
    # into one segment spanning to the last frame, keeping the earlier target.
    raw = {"segments": [
        {"phase": "approach", "target": "red cube", "end_frame": 10, "evidence": "e", "subtask_text": "x"},
        {"phase": "grasp", "target": "red cube", "end_frame": 20, "evidence": "e", "subtask_text": "y"},
        {"phase": "retract", "target": "none", "end_frame": 26, "evidence": "e", "subtask_text": "pull back"},
        {"phase": "retract", "target": "none", "end_frame": 30, "evidence": "e", "subtask_text": "pull back more"},
    ]}
    segs = validate_grounded_segments(raw, 31, _S2, _RUBRIC)
    assert [s.phase for s in segs] == ["approach", "grasp", "retract"]
    assert segs[-1].end_frame == 30                    # merged tail spans to the last frame
    assert segs[-1].start_frame == 21                  # and starts where the first retract did


def test_schema_v3_target_roundtrips(tmp_path: Path):
    # A v3 annotation with targets writes and reads back the target column intact.
    ann = EpisodeAnnotation(
        episode_id="0", task="stack", num_frames=31, fps=30.0, provider="mock", model="mock",
        strategy="S2",
        subtasks=[
            SubtaskSegment(0, 0, 10, "move to red cube", phase="approach", target="red cube", evidence="e0"),
            SubtaskSegment(1, 11, 30, "pull back", phase="retract", target=None, evidence="e1"),
        ],
    )
    write_annotations([ann], tmp_path)
    back = read_annotations(tmp_path)
    recs = sorted(episode_records(back, "0")["subtasks"], key=lambda r: r["segment_idx"])
    assert recs[0]["target"] == "red cube"
    assert recs[0]["phase"] == "approach"
    # retract's None target survives the round-trip as a falsy/empty value.
    assert not _clean_str(recs[1].get("target"))


def _clean_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "none") else s


# --------------------------------------------------------------------------- #
# Offline segmentation S0..S4
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["S0", "S1", "S2", "S3", "S4"])
def test_segment_episode_offline_contiguous_full_coverage(strategy: str, tmp_path: Path):
    rubric = load_rubric()
    provider = build_provider("mock")
    ep = synthetic_episode(0)
    res = segment_episode(ep, provider, rubric, load_strategy(strategy), tmp_path)
    segs = res.segments
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == ep.num_frames - 1
    for prev, nxt in zip(segs, segs[1:], strict=False):
        assert nxt.start_frame == prev.end_frame + 1
    if strategy != "S0":
        assert all(s.evidence for s in segs)            # grounded: evidence present
        assert len(segs) >= rubric.strategy_min_segments  # granularity floor met


class _SingleSegProvider:
    """Fake provider that always returns a single grounded segment (below the floor)."""
    name = "single"
    model = "single"

    def ask(self, frames, frame_labels, question, receipt_path, *, frame_captions=None, temperature=None):
        import json

        from robolabel.providers.base import ProviderResponse, write_receipt
        q = question.lower()
        last = int(max(frame_labels)) if frame_labels else 0
        # grounded-label check before events (the label prompt embeds the observe events).
        if "end_frame" in q and "phase" in q:
            ans = json.dumps({"segments": [{"phase": "other", "target": "the workpiece", "end_frame": last,
                                            "evidence": "one continuous action", "subtask_text": "do the whole task"}]})
        elif '"events"' in q:
            ans = json.dumps({"events": [{"frame": last // 2, "evidence": "e"}], "objects": ["o"]})
        else:
            ans = json.dumps({"answer": "x"})
        write_receipt(receipt_path, {"provider": self.name, "model": self.model, "response_json": {"answer": ans}})
        return ProviderResponse(ans, {}, self.name, self.model, 0.0, 0.0)


def test_min_granularity_warn_accepts_single_segment(tmp_path: Path):
    from dataclasses import replace
    rubric = load_rubric()
    ep = synthetic_episode(0)
    cfg = replace(load_strategy("S2"), min_granularity_policy="warn", self_consistency_k=1,
                  refine_boundaries=False, max_label_attempts=3)
    with pytest.warns(UserWarning, match="single_segment_candidate"):
        res = segment_episode(ep, _SingleSegProvider(), rubric, cfg, tmp_path)
    assert res.granularity_warning is True            # flagged single_segment_candidate
    assert len(res.segments) == 1                     # the single segment is accepted, not forced up
    assert res.segments[0].evidence                   # grounded fields retained
    assert len(res.calls) == 2                         # observe + 1 label (accepts on attempt 0, no re-prompt)


def test_min_granularity_reject_reprompts(tmp_path: Path):
    from dataclasses import replace
    rubric = load_rubric()
    ep = synthetic_episode(0)
    cfg = replace(load_strategy("S2"), min_granularity_policy="reject", self_consistency_k=1,
                  refine_boundaries=False, max_label_attempts=3)
    res = segment_episode(ep, _SingleSegProvider(), rubric, cfg, tmp_path)
    # reject re-prompts up to max_label_attempts=3 before the lenient fallback.
    assert len(res.calls) == 1 + 3                     # observe + 3 label attempts


def test_presets_default_to_warn_policy():
    for k in ("S2", "S3", "S4"):
        assert load_strategy(k).min_granularity_policy == "warn"


def test_s4_draws_multiple_label_samples(tmp_path: Path):
    rubric = load_rubric()
    provider = build_provider("mock")
    ep = synthetic_episode(1)
    res = segment_episode(ep, provider, rubric, load_strategy("S4"), tmp_path)
    # observe(1) + k=3 label samples + 3 refine calls on a 4-segment mock = 7 calls.
    assert len(res.calls) >= 1 + 3


def test_sample_frames_count_and_bounds():
    ep = synthetic_episode(0)
    idx = sample_frames(ep, 12)
    assert idx[0] == 0 and idx[-1] == ep.num_frames - 1
    assert idx == sorted(set(idx))
    assert len(idx) <= 12


# --------------------------------------------------------------------------- #
# Failure-band detectors
# --------------------------------------------------------------------------- #
def test_degenerate_single_segment_detector():
    assert is_degenerate_single_segment([{"start_frame": 0, "end_frame": 99}])
    assert not is_degenerate_single_segment(
        [{"start_frame": 0, "end_frame": 49}, {"start_frame": 50, "end_frame": 99}])


def test_uniform_split_detector():
    # Five equal fifths of a 100-frame episode == uniform.
    uniform = [{"start_frame": i * 20, "end_frame": i * 20 + 19} for i in range(5)]
    assert is_uniform_split(uniform, cv_threshold=0.12, min_segments=3)
    # Clearly unequal lengths are not uniform.
    varied = [{"start_frame": 0, "end_frame": 4}, {"start_frame": 5, "end_frame": 70},
              {"start_frame": 71, "end_frame": 99}]
    assert not is_uniform_split(varied, cv_threshold=0.12, min_segments=3)


# --------------------------------------------------------------------------- #
# Gate: quality-outlier needs_review + never drops
# --------------------------------------------------------------------------- #
def _ann(eid, quality, subtasks, reason="clean successful placement with clear boundaries"):
    return EpisodeAnnotation(
        episode_id=eid, task="put block in bowl", num_frames=100, fps=10.0,
        provider="mock", model="m",
        metadata=EpisodeMetadata(quality=quality, task_success_quality=quality, mistake=False, reason=reason),
        subtasks=[SubtaskSegment(i, *span, text) for i, (span, text) in enumerate(subtasks)],
    )


def test_gate_quality_outlier_needs_review_and_never_drops(tmp_path: Path):
    spans = [((0, 24), "approach brick"), ((25, 49), "grasp brick"),
             ((50, 74), "carry brick"), ((75, 99), "place brick")]
    anns = [_ann(f"e{i}", 5, spans) for i in range(8)]
    anns.append(_ann("bad1", 1, spans))   # hallucinated low score on a good episode
    write_annotations(anns, tmp_path)
    report = run_gate(tmp_path)
    outliers = [i for i in report.issues if i.check == "quality_outlier_needs_review"]
    assert any(i.episode_id == "bad1" for i in outliers)
    assert report.dropped_episode_count == 0
    # Every episode is still present in the set the gate inspected.
    assert report.episode_count == 9


def test_gate_flags_degenerate_and_uniform(tmp_path: Path):
    degenerate = _ann("degen", 5, [((0, 99), "complete the task")])
    uniform = _ann("unif", 5, [((0, 24), "a"), ((25, 49), "b"), ((50, 74), "c"), ((75, 99), "d")])
    write_annotations([degenerate, uniform], tmp_path)
    report = run_gate(tmp_path)
    checks = {i.check for i in report.issues}
    assert "degenerate_single_segment" in checks
    assert "uniform_split" in checks
    assert report.band_counts["degenerate_single_segment"] == 1
    assert report.band_counts["uniform_split"] == 1


# --------------------------------------------------------------------------- #
# Schema v2 round-trip + backward-compatible v1 read
# --------------------------------------------------------------------------- #
def test_schema_v2_phase_evidence_roundtrip(tmp_path: Path):
    ann = EpisodeAnnotation(
        episode_id="ep0", task="t", num_frames=40, fps=10.0, provider="mock", model="m",
        strategy="S2",
        subtasks=[SubtaskSegment(0, 0, 19, "approach", phase="approach", evidence="gripper above brick"),
                  SubtaskSegment(1, 20, 39, "grasp", phase="grasp", evidence="gripper closes")],
    )
    write_annotations([ann], tmp_path)
    df = read_annotations(tmp_path)
    assert "phase" in df.columns and "boundary_evidence" in df.columns and "strategy" in df.columns
    rec = episode_records(df, "ep0")
    assert rec["subtasks"][0]["phase"] == "approach"
    assert rec["subtasks"][0]["boundary_evidence"] == "gripper above brick"


def test_v1_parquet_without_new_columns_still_reads(tmp_path: Path):
    ann = EpisodeAnnotation(
        episode_id="ep0", task="t", num_frames=40, fps=10.0, provider="mock", model="m",
        subtasks=[SubtaskSegment(0, 0, 39, "do it")],
    )
    write_annotations([ann], tmp_path)
    # Simulate a v1 file: drop the v2 columns before re-reading.
    import pandas as pd
    p = tmp_path / "annotations.parquet"
    df = pd.read_parquet(p).drop(columns=["phase", "boundary_evidence", "strategy"])
    df.to_parquet(p, index=False)
    rec = episode_records(read_annotations(tmp_path), "ep0")  # must not KeyError
    assert rec["subtasks"][0]["subtask_text"] == "do it"


# --------------------------------------------------------------------------- #
# Eval harness pure scoring functions
# --------------------------------------------------------------------------- #
def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_strategies.py"
    spec = importlib.util.spec_from_file_location("eval_strategies", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_harness_scoring_against_gold():
    ev = _load_eval_module()
    segs = [SubtaskSegment(0, 0, 10, "a"), SubtaskSegment(1, 11, 20, "b")]
    auto = ev.segments_to_auto(segs, 21)
    assert auto["subtasks"][0]["end_frame"] == 10
    assert auto["subgoals"][0]["frame_idx"] == 10  # subgoal == segment end
    gold = {"schema_version": "robolabel/gold/v1", "episodes": [
        {"episode_id": "0", "task": "t", "num_frames": 21,
         "auto": {"subtasks": [], "metadata": {}, "subgoals": []},
         "gold": {"metadata": {"quality": 4},
                  "subtasks": [{"segment_idx": 0, "start_frame": 0, "end_frame": 10},
                               {"segment_idx": 1, "start_frame": 11, "end_frame": 20}],
                  "subgoals": []}}]}
    eval_gold = ev.build_eval_gold(gold, {"0": auto}, {"0": 4}, ["0"])
    rep = ev.score_against_gold(eval_gold)
    assert rep["subtask_boundary_temporal_iou_mean"] == 1.0  # exact match
    assert rep["quality_exact_agreement"] == 1.0
