from __future__ import annotations

from pathlib import Path

from robovid_conditioner.gate import run_gate
from robovid_conditioner.rubric import load_rubric
from robovid_conditioner.schema import EpisodeAnnotation, EpisodeMetadata, SubtaskSegment, write_annotations


def test_rubric_fill_preserves_json_braces():
    rubric = load_rubric()
    prompt = rubric.subtask_label_prompt(task="put block in bowl", last_frame=23, observations="[]")
    assert "put block in bowl" in prompt
    assert '"segments"' in prompt  # literal JSON example survived substitution
    assert "{task}" not in prompt


def test_rubric_custom_path(tmp_path: Path):
    custom = tmp_path / "r.yaml"
    custom.write_text(
        "schema_version: robovid_conditioner/rubric/v1\nname: custom\nkeyframes: 3\n"
        "subtasks: {min_segments: 1, max_segments: 2, observe_prompt: 'o {task}', label_prompt: 'l {task}'}\n"
        "metadata: {quality_scale: {'1': low, '5': high}, observe_prompt: 'o', label_prompt: 'l {quality_scale}'}\n"
        "subgoals: {source: subtask_end}\ngate: {min_distinct_quality_scores: 3}\n",
        encoding="utf-8",
    )
    rubric = load_rubric(custom)
    assert rubric.name == "custom"
    assert rubric.keyframes == 3
    assert rubric.gate["min_distinct_quality_scores"] == 3


def _ann(episode_id, quality, reason, subtasks, mistake=False):
    return EpisodeAnnotation(
        episode_id=episode_id, task="put block in bowl", num_frames=12, fps=10.0,
        provider="mock", model="m",
        metadata=EpisodeMetadata(quality=quality, task_success_quality=quality, mistake=mistake, reason=reason),
        subtasks=[SubtaskSegment(i, *span, text) for i, (span, text) in enumerate(subtasks)],
    )


def test_gate_flags_repeated_subtask_text(tmp_path: Path):
    ann = _ann("e0", 4, "clear boundaries are visible",
               [((0, 5), "grasp block"), ((6, 11), "grasp block")])
    write_annotations([ann], tmp_path)
    report = run_gate(tmp_path)
    assert any(i.check == "repeated_subtask_text" for i in report.issues)


def test_gate_flags_score_reason_contradiction(tmp_path: Path):
    ann = _ann("e0", 2, "the robot successfully completed the placement cleanly",
               [((0, 5), "grasp block"), ((6, 11), "place block")])
    write_annotations([ann], tmp_path)
    report = run_gate(tmp_path)
    assert any(i.check == "score_reason_contradiction" for i in report.issues)


def test_gate_flags_collapsed_distribution(tmp_path: Path):
    anns = [
        _ann(f"e{i}", 5, "clean successful placement with clear boundaries",
             [((0, 5), "grasp block"), ((6, 11), "place block")])
        for i in range(10)
    ]
    write_annotations(anns, tmp_path)
    report = run_gate(tmp_path)
    assert any(i.check == "collapsed_score_distribution" for i in report.issues)


def test_gate_clean_set_passes(tmp_path: Path):
    anns = [
        _ann(f"e{i}", q, "boundaries visible with imperfect final placement",
             [((0, 5), f"grasp item{i}"), ((6, 11), f"place item{i}")])
        for i, q in enumerate([3, 4, 5, 4, 3, 5, 4, 3, 5, 4])
    ]
    write_annotations(anns, tmp_path)
    report = run_gate(tmp_path)
    assert report.passed, report.to_text()
