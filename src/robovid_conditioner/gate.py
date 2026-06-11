"""Quality gate: cheap, automatic red flags on a VLM annotation set.

The gate does not decide whether labels are *correct* (only the human
calibration loop and the reliability report do that). It flags the failure modes
that are detectable without a human: a collapsed score distribution, templated /
repeated subtask text, subtask objects ungrounded in the stage-one observation,
and score↔reason contradictions. Thresholds come from the rubric's ``gate``
section.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .rubric import Rubric, load_rubric
from .schema import episode_records, list_episode_ids, read_annotations

_SUCCESS_WORDS = {"success", "successfully", "completed", "clean", "placed", "resting", "inside"}
_FAILURE_WORDS = {"fail", "failed", "failing", "unfinished", "wrong", "incomplete", "mistake", "incorrect", "drop"}
# Tokens that are NOT objects: manipulation action verbs, robot self-parts,
# colors, and connectives. Object-grounding only flags tokens outside this set,
# so it must be generous about verbs to avoid false positives like
# "'reach' not in observation". Extend it for your task family as needed.
_STOP = {
    # connectives / determiners
    "after", "across", "above", "and", "before", "beside", "from", "into", "onto", "over", "that",
    "the", "toward", "towards", "with", "without", "then", "while", "back", "down",
    # action verbs (manipulation)
    "approach", "approaches", "align", "aligns", "ascend", "bring", "brings", "carry", "carries",
    "complete", "completes", "completed", "descend", "drop", "drops", "finish", "finishes", "grasp",
    "grasps", "grasping", "grip", "grips", "hold", "holds", "hover", "hovers", "insert", "inserts",
    "leave", "lift", "lifts", "lower", "lowers", "lowered", "move", "moves", "moving", "pick", "picks",
    "pickup", "place", "places", "placing", "position", "positions", "positioned", "pull", "pulls",
    "push", "pushes", "raise", "raises", "raised", "reach", "reaches", "reaching", "relocate",
    "relocates", "release", "releases", "retract", "retracts", "retreat", "retreats", "return",
    "returns", "rotate", "rotates", "scoop", "set", "sets", "settle", "slide", "slides", "stabilize",
    "stabilizes", "steady", "transport", "transports", "transporting", "turn", "turns", "withdraw",
    # robot self-parts (not objects to ground)
    "arm", "gripper", "robot", "hand", "effector", "manipulator", "wrist", "claw",
    # generic / colors / materials
    "clear", "destination", "edge", "empty", "object", "objects", "partial", "task", "target",
    "black", "blue", "green", "gray", "grey", "metal", "orange", "pink", "purple", "red", "silver",
    "sink", "white", "yellow",
}


@dataclass(frozen=True)
class GateIssue:
    episode_id: str
    check: str
    detail: str

    def format(self) -> str:
        return f"{self.episode_id}: {self.check}: {self.detail}"


@dataclass(frozen=True)
class GateReport:
    passed: bool
    issues: tuple[GateIssue, ...]
    episode_count: int
    quality_counts: dict[int, int]
    band_counts: dict[str, int] = field(default_factory=dict)

    # The gate is advisory only. It flags episodes; it never removes them from the
    # annotation set. This count is always 0 and is reported to make that explicit.
    dropped_episode_count: int = 0

    def to_text(self) -> str:
        lines = [
            f"Quality gate: {'PASS' if self.passed else 'FLAGS'}",
            f"Episodes checked: {self.episode_count}",
            f"Quality score counts: {self.quality_counts}",
        ]
        if self.band_counts:
            lines.append(f"Failure-band counts: {self.band_counts}")
        if self.issues:
            lines.append(f"Flags ({len(self.issues)}):")
            lines.extend(f"- {issue.format()}" for issue in self.issues)
        else:
            lines.append("No automatic flags. (This does not mean the labels are correct.)")
        lines.append(
            f"Episodes dropped by the gate: {self.dropped_episode_count} "
            "(the gate only flags; it never drops episodes)."
        )
        return "\n".join(lines)


def run_gate(annotations_dir: str | Path, rubric: Rubric | None = None) -> GateReport:
    """Evaluate the gate over an annotation set directory (or parquet path)."""
    rubric = rubric or load_rubric()
    gate_cfg = rubric.gate
    out_dir = Path(annotations_dir)
    df = read_annotations(out_dir)
    receipts_root = (out_dir if out_dir.is_dir() else out_dir.parent) / "raw_receipts"

    issues: list[GateIssue] = []
    quality_values: list[int] = []
    quality_by_episode: dict[str, int] = {}
    band_counts = {"degenerate_single_segment": 0, "uniform_split": 0}
    episode_ids = list_episode_ids(df)
    cv_threshold = float(gate_cfg.get("uniform_split_cv_threshold", 0.12))
    min_seg_uniform = int(gate_cfg.get("min_segments_for_uniform_check", 3))
    for episode_id in episode_ids:
        rec = episode_records(df, episode_id)
        metadata = rec["metadata"]
        subtasks = rec["subtasks"]

        quality = _safe_int(metadata.get("quality"))
        if quality is not None:
            quality_values.append(quality)
            quality_by_episode[episode_id] = quality

        if gate_cfg.get("flag_repeated_subtask_text", True):
            texts = [str(s.get("subtask_text", "")).strip().lower() for s in subtasks if s.get("subtask_text")]
            if texts and len(set(texts)) < len(texts):
                issues.append(GateIssue(episode_id, "repeated_subtask_text", "duplicate subtask text within episode"))

        if gate_cfg.get("flag_object_grounding", True):
            issues.extend(_object_grounding_issues(episode_id, subtasks, rec["task"], receipts_root))

        if gate_cfg.get("flag_score_reason_contradiction", True):
            issues.extend(_score_reason_issues(episode_id, metadata))

        # Failure-band detectors (boundary quality).
        if gate_cfg.get("flag_degenerate_single_segment", True) and is_degenerate_single_segment(subtasks):
            band_counts["degenerate_single_segment"] += 1
            issues.append(GateIssue(episode_id, "degenerate_single_segment",
                                    "a single subtask spans the whole episode"))
        elif gate_cfg.get("flag_uniform_split", True) and is_uniform_split(subtasks, cv_threshold, min_seg_uniform):
            band_counts["uniform_split"] += 1
            issues.append(GateIssue(episode_id, "uniform_split",
                                    "segment lengths are near-uniform (boundaries not grounded to frames)"))

    quality_counts = _counts(quality_values)
    min_distinct = int(gate_cfg.get("min_distinct_quality_scores", 2))
    min_episodes = int(gate_cfg.get("min_episodes_for_dispersion", 8))
    if len(quality_values) >= min_episodes and len(quality_counts) < min_distinct:
        issues.append(GateIssue(
            "__dataset__", "collapsed_score_distribution",
            f"only {len(quality_counts)} distinct quality score(s) {sorted(quality_counts)} "
            f"across {len(quality_values)} episodes",
        ))

    # Quality-score outlier policy: a score this many points or more below the
    # dataset neighborhood (median) is flagged needs_review — NOT dropped. This
    # catches hallucinated low scores on otherwise-good episodes.
    if gate_cfg.get("flag_quality_outlier", True) and len(quality_values) >= 3:
        margin = int(gate_cfg.get("quality_outlier_margin", 2))
        neighborhood = statistics.median(quality_values)
        for episode_id, q in quality_by_episode.items():
            if neighborhood - q >= margin:
                issues.append(GateIssue(
                    episode_id, "quality_outlier_needs_review",
                    f"quality {q} is >= {margin} below the dataset median {neighborhood:g}; needs_review",
                ))

    return GateReport(
        passed=not issues,
        issues=tuple(issues),
        episode_count=len(episode_ids),
        quality_counts=quality_counts,
        band_counts=band_counts,
    )


def _segment_lengths(subtasks: list[dict[str, Any]]) -> list[int]:
    lengths: list[int] = []
    for s in subtasks:
        start = _safe_int(s.get("start_frame"))
        end = _safe_int(s.get("end_frame"))
        if start is not None and end is not None:
            lengths.append(end - start + 1)
    return lengths


def is_degenerate_single_segment(subtasks: list[dict[str, Any]]) -> bool:
    """Band (a): the VLM collapsed the episode into one "complete the task" segment."""
    return len(subtasks) <= 1


def is_uniform_split(subtasks: list[dict[str, Any]], cv_threshold: float = 0.12,
                     min_segments: int = 3) -> bool:
    """Band (b): segment lengths are near-uniform (boundaries at fixed fractions).

    Measured by the coefficient of variation (std/mean) of segment lengths. Equal
    segments give CV 0; a CV below ``cv_threshold`` means the model never grounded
    the boundaries to the video. Only checked for episodes with enough segments.
    """
    lengths = _segment_lengths(subtasks)
    if len(lengths) < min_segments:
        return False
    mean = statistics.mean(lengths)
    if mean <= 0:
        return False
    cv = statistics.pstdev(lengths) / mean
    return cv < cv_threshold


def _object_grounding_issues(
    episode_id: str, subtasks: list[dict[str, Any]], task: Any, receipts_root: Path
) -> list[GateIssue]:
    observation_text = _observe_text(receipts_root / episode_id / "subtasks_observe.json")
    if observation_text is None:
        return []  # no observation receipt to ground against; skip rather than guess
    task_text = str(task or "").lower()
    issues: list[GateIssue] = []
    for subtask in subtasks:
        for token in _object_tokens(str(subtask.get("subtask_text", ""))):
            if token not in observation_text and token not in task_text:
                issues.append(GateIssue(episode_id, "object_grounding", f"{token!r} not in stage-one observation"))
                break
    return issues


def _score_reason_issues(episode_id: str, metadata: dict[str, Any]) -> list[GateIssue]:
    reason = str(metadata.get("reason", "")).strip()
    lower = reason.lower()
    issues: list[GateIssue] = []
    if not reason or lower in {"one short sentence", "specific evidence consistent with both scores",
                               "mock: placeholder reason, not derived from the frames"}:
        issues.append(GateIssue(episode_id, "placeholder_reason", "missing or templated reason"))
    quality = _safe_int(metadata.get("task_success_quality")) or _safe_int(metadata.get("quality"))
    if quality is None:
        return issues
    says_success = _has_unnegated_word(lower, _SUCCESS_WORDS)
    says_failure = _has_unnegated_word(lower, _FAILURE_WORDS)
    if quality <= 2 and says_success and not says_failure:
        issues.append(GateIssue(episode_id, "score_reason_contradiction", "low quality with success reason"))
    if quality >= 4 and says_failure and not says_success:
        issues.append(GateIssue(episode_id, "score_reason_contradiction", "high quality with failure reason"))
    if bool(metadata.get("mistake")) and quality >= 5:
        issues.append(GateIssue(episode_id, "score_reason_contradiction", "mistake=true with perfect quality"))
    return issues


def _observe_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    answer = data.get("response_json", {}).get("answer") if isinstance(data.get("response_json"), dict) else None
    return json.dumps(answer or data.get("response_text") or "", sort_keys=True).lower()


def _object_tokens(text: str) -> list[str]:
    tokens = [t.strip(".,:;()[]{}").lower() for t in text.split()]
    return [t for t in tokens if len(t) >= 4 and t not in _STOP]


def _has_unnegated_word(text: str, target_words: set[str]) -> bool:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    negators = {"no", "not", "without", "none", "never"}
    for idx, token in enumerate(tokens):
        if token in target_words and not any(n in tokens[max(0, idx - 10):idx] for n in negators):
            return True
    return False


def _counts(values: list[int]) -> dict[int, int]:
    if not values:
        return {}
    return {int(k): int(v) for k, v in pd.Series(values, dtype="int64").value_counts().sort_index().to_dict().items()}


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
