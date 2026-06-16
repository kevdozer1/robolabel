"""Strategy-driven subtask segmentation (S1+).

This implements the grounded segmentation path selected by a
:class:`~robolabel.strategy.StrategyConfig`. The baseline (S0) path is
left in :func:`robolabel.labelers.subtasks.label_subtasks` and is
delegated to unchanged when the strategy is baseline.

The grounded path:

1. samples ``frame_count`` evenly-spaced frames and presents each with its frame
   index + timestamp (caption + a textual manifest in the prompt);
2. runs the two-stage observe→label flow, but the label stage must return
   per-segment ``end_frame`` (a concrete frame index) and ``evidence`` (one line),
   optionally constrained to a closed phase vocabulary and a minimum granularity
   (schema-validated, with re-prompts);
3. for self-consistency (S4) draws ``self_consistency_k`` label samples and takes
   the per-boundary median;
4. for refinement (S3+) sends a dense ±``refine_window`` frame window per internal
   boundary and pins it to the exact transition frame.
"""

from __future__ import annotations

import json
import math
import statistics
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..episode import Episode
from ..providers.base import ProviderResponse, VLMProvider, try_extract_json
from ..rubric import Rubric
from ..schema import SubtaskSegment
from ..strategy import StrategyConfig
from . import validate_segments
from .subtasks import SubtaskResult, label_subtasks


class SchemaValidationError(ValueError):
    """A grounded segmentation answer violated the strategy's output schema."""


class GranularityError(SchemaValidationError):
    """The answer was valid but below the minimum-granularity floor.

    Carries the finalized (below-min) ``segments`` so a ``warn`` policy can accept
    them; a ``reject`` policy treats it like any other schema failure.
    """

    def __init__(self, message: str, segments: list[SubtaskSegment]):
        super().__init__(message)
        self.segments = segments


def segment_episode(
    episode: Episode,
    provider: VLMProvider,
    rubric: Rubric,
    config: StrategyConfig,
    receipt_dir: Path,
) -> SubtaskResult:
    """Segment one episode under ``config``. Baseline strategies delegate to S0."""
    if config.is_baseline:
        return label_subtasks(episode, provider, rubric, receipt_dir)

    indices = sample_frames(episode, config.frame_count)
    frames = episode.frames(indices)
    captions = frame_captions(indices, episode.fps) if config.caption_timestamps else None
    manifest = frame_manifest(indices, episode.fps)
    last_frame = episode.num_frames - 1
    task = episode.task or episode.episode_id
    calls: list[ProviderResponse] = []

    # Stage one: physical events, grounded to frame indices (observed once, reused
    # across all self-consistency samples — it is deterministic).
    observe_q = rubric.grounded_observe_prompt(task=task, frame_manifest=manifest)
    observed = provider.ask(frames, indices, observe_q, receipt_dir / "subtasks_observe.json",
                            frame_captions=captions)
    calls.append(observed)
    observations = try_extract_json(observed.answer)
    obs_text = json.dumps(observations, sort_keys=True) if observations is not None else "[]"

    # Stage two: k grounded label samples.
    k = max(1, config.self_consistency_k)
    samples: list[list[SubtaskSegment]] = []
    granularity_warning = False
    for s in range(k):
        temperature = config.temperature if k > 1 else None
        segs, sample_calls, gflag = _grounded_label(
            provider, frames, indices, rubric, config, receipt_dir,
            task=task, last_frame=last_frame, observations=obs_text, manifest=manifest,
            captions=captions, sample_idx=s, temperature=temperature, num_samples=k,
        )
        calls.extend(sample_calls)
        samples.append(segs)
        granularity_warning = granularity_warning or gflag

    segments = _median_combine(samples, last_frame) if k > 1 else samples[0]

    # Post-pass: dense-window boundary refinement.
    if config.refine_boundaries and len(segments) > 1:
        segments, refine_calls = _refine_boundaries(
            episode, provider, rubric, segments, config, receipt_dir
        )
        calls.extend(refine_calls)

    for i, seg in enumerate(segments):
        seg.segment_idx = i
    # Flag if the final segmentation is below the granularity floor (a
    # single_segment_candidate), whether from a sample or the combined result.
    if config.enforce_min_segments and len(segments) < rubric.strategy_min_segments:
        granularity_warning = True
    return SubtaskResult(segments, observations, calls, indices,
                         granularity_warning=granularity_warning)


# --------------------------------------------------------------------------- #
# Frame sampling + presentation
# --------------------------------------------------------------------------- #
def sample_frames(episode: Episode, frame_count: int) -> list[int]:
    """Evenly-spaced, de-duplicated, sorted frame indices for the contact sheet."""
    n = episode.num_frames
    if n <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, min(max(2, frame_count), n))})


def frame_captions(indices: list[int], fps: float) -> list[str]:
    f = max(float(fps), 1e-6)
    return [f"f{i}  {i / f:.2f}s" for i in indices]


def frame_manifest(indices: list[int], fps: float) -> str:
    f = max(float(fps), 1e-6)
    return ", ".join(f"frame {i} ({i / f:.2f}s)" for i in indices)


# --------------------------------------------------------------------------- #
# Grounded label stage (with schema validation + re-prompts)
# --------------------------------------------------------------------------- #
def _grounded_label(
    provider: VLMProvider,
    frames: list[np.ndarray],
    indices: list[int],
    rubric: Rubric,
    config: StrategyConfig,
    receipt_dir: Path,
    *,
    task: str,
    last_frame: int,
    observations: str,
    manifest: str,
    captions: list[str] | None,
    sample_idx: int,
    temperature: float | None,
    num_samples: int,
) -> tuple[list[SubtaskSegment], list[ProviderResponse]]:
    prompt_fn = (rubric.grounded_label_prompt_open if config.open_vocabulary
                 else rubric.grounded_label_prompt)
    base_q = prompt_fn(
        task=task, last_frame=last_frame, observations=observations, frame_manifest=manifest
    )
    calls: list[ProviderResponse] = []
    segments: list[SubtaskSegment] | None = None
    granularity_warning = False
    for attempt in range(max(1, config.max_label_attempts)):
        question = base_q if attempt == 0 else base_q + "\n\n" + _retry_suffix(rubric)
        receipt = receipt_dir / _label_receipt_name(num_samples, sample_idx, attempt)
        resp = provider.ask(frames, indices, question, receipt,
                            frame_captions=captions, temperature=temperature)
        calls.append(resp)
        try:
            segments = validate_grounded_segments(
                try_extract_json(resp.answer), last_frame + 1, config, rubric
            )
            break
        except GranularityError as ge:
            # Below the granularity floor. "warn" (default): accept it and flag a
            # single_segment_candidate — some episodes really are one segment (ep7).
            # "reject": treat like any failure and re-prompt.
            if config.min_granularity_policy == "warn":
                segments = ge.segments
                granularity_warning = True
                warnings.warn(
                    "grounded segmentation below min granularity "
                    "(single_segment_candidate); accepted under 'warn' policy",
                    stacklevel=2,
                )
                break
            segments = None
        except SchemaValidationError:
            segments = None
    if segments is None:
        # Honest fallback: keep whatever boundaries we can parse leniently so the
        # episode still produces a segmentation. It will be caught by the
        # degenerate / uniform-split gate detectors rather than silently dropped.
        segments = validate_segments(
            try_extract_json(calls[-1].answer), last_frame + 1,
            1, rubric.strategy_max_segments,
        )
    return segments, calls, granularity_warning


def validate_grounded_segments(
    raw: object, num_frames: int, config: StrategyConfig, rubric: Rubric
) -> list[SubtaskSegment]:
    """Validate a grounded segmentation answer into contiguous frame-indexed segments.

    Raises :class:`SchemaValidationError` when the answer violates the strategy's
    contract (no segments, a non-integer ``end_frame``, missing per-boundary
    ``evidence`` when grounded, or fewer than the minimum segments when
    ``enforce_min_segments``). Unknown phases are coerced to ``other`` rather than
    rejected.
    """
    items = raw.get("segments", raw.get("subtasks", [])) if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise SchemaValidationError("no segments in answer")
    last = max(0, num_frames - 1)
    vocab = set(rubric.phase_vocabulary)
    clean: list[SubtaskSegment] = []
    cursor = 0
    for item in items[: rubric.strategy_max_segments]:
        if not isinstance(item, dict):
            continue
        end = item.get("end_frame", item.get("end_step"))
        try:
            end_i = int(end)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("segment missing integer end_frame") from exc
        evidence = str(item.get("evidence") or "").strip()
        if config.grounded and not evidence:
            raise SchemaValidationError("segment missing per-boundary evidence")
        phase = str(item.get("phase") or "").strip().lower()
        if config.closed_vocabulary and phase not in vocab:
            phase = "other"
        target = _clean_target(item.get("target"))
        if config.require_target and target is None and not _target_optional(phase, config):
            raise SchemaValidationError(f"segment ({phase or 'phase'}) missing a target object")
        text = str(item.get("subtask_text") or item.get("text") or item.get("description") or "").strip()
        if not text:
            text = f"{phase} {target}".strip() if (phase or target) else "subtask"
        end_i = min(max(cursor, end_i), last)
        clean.append(SubtaskSegment(
            segment_idx=len(clean), start_frame=cursor, end_frame=end_i,
            subtask_text=text[:160], phase=phase or None, evidence=evidence or None, target=target,
        ))
        cursor = end_i + 1
        if cursor > last:
            break
    if not clean:
        raise SchemaValidationError("no valid segments after parsing")
    clean = _dedupe_trailing_phases(clean)  # collapse e.g. two consecutive "retract" tails
    clean[0].start_frame = 0
    clean[-1].end_frame = last
    if config.enforce_min_segments and len(clean) < rubric.strategy_min_segments:
        # Finalized but below the floor — caller decides (reject vs warn) by policy.
        for i, seg in enumerate(clean):
            seg.segment_idx = i
        raise GranularityError(
            f"below min granularity: {len(clean)} < {rubric.strategy_min_segments} segments", clean
        )
    for i, seg in enumerate(clean):
        seg.segment_idx = i
    return clean


_TARGET_NONE = {"", "none", "n/a", "na", "-", "null", "the scene", "scene", "object"}
_RETRACT_LIKE = ("retract", "withdraw", "retreat", "return", "go home", "home", "reset", "back away")


def _clean_target(value: object) -> str | None:
    """Normalize a target string; '', 'none', 'n/a', etc. -> None."""
    s = str(value or "").strip()
    return s[:80] if s and s.lower() not in _TARGET_NONE else None


def _target_optional(phase: str, config: StrategyConfig) -> bool:
    """Phases for which a missing target is allowed under require_target.

    Closed-vocab (S2/S3/S4): only the exact phase ``retract``. Open-vocab (S2-open):
    also any free-text phase that reads like a final withdraw (``withdraw``, ``go home``,
    ...), since those have no object. Leaves the closed-vocab path byte-identical.
    """
    if phase == "retract":
        return True
    if config.open_vocabulary:
        p = phase.lower()
        return any(tok in p for tok in _RETRACT_LIKE)
    return False


def _is_winddown(phase: str | None) -> bool:
    """True for any retract/withdraw/return/home-style terminal wind-down phase (by category)."""
    p = (phase or "").lower()
    return bool(p) and any(tok in p for tok in _RETRACT_LIKE)


_CONTACT_TOKENS = ("grasp", "release", "place", "pick", "grip", "contact", "lift", "drop", "set down")


def _is_contact_phase(phase: str | None) -> bool:
    """True for grasp/release-style contact-event phases (the hard-to-time boundaries)."""
    p = (phase or "").lower()
    return bool(p) and any(tok in p for tok in _CONTACT_TOKENS)


def _dedupe_trailing_phases(segs: list[SubtaskSegment]) -> list[SubtaskSegment]:
    """Collapse consecutive *trailing* phases that are either string-identical (two 'retract')
    OR both wind-down-like with different labels (e.g. 'withdraw gripper' + 'retract arm') into
    one segment spanning to the last frame. Keeps the earlier (first-observed) label."""
    while len(segs) >= 2:
        a, b = segs[-2], segs[-1]
        same = bool(a.phase) and a.phase == b.phase
        winddown = _is_winddown(a.phase) and _is_winddown(b.phase)
        if not (same or winddown):
            break
        merged_end = b.end_frame
        merged_target = a.target or b.target
        segs.pop()
        segs[-1].end_frame = merged_end
        segs[-1].target = merged_target
    return segs


def _retry_suffix(rubric: Rubric) -> str:
    return (
        f"Your previous answer was rejected. Return at least {rubric.strategy_min_segments} "
        "segments, each with an integer end_frame chosen from the captioned frame indices, a "
        "one-clause evidence string, and a specific 'target' object named from the scene (which "
        "one of the visible objects — disambiguate when several are present); 'target' may be "
        "'none' only for the 'retract' phase. Do not return a single segment."
    )


def _label_receipt_name(num_samples: int, sample_idx: int, attempt: int) -> str:
    if num_samples == 1 and attempt == 0:
        return "subtasks_label.json"  # stable name (matches resume cache + gate)
    return f"subtasks_label_s{sample_idx}_a{attempt}.json"


# --------------------------------------------------------------------------- #
# Self-consistency: per-boundary median over k samples
# --------------------------------------------------------------------------- #
def _median_combine(samples: list[list[SubtaskSegment]], last: int) -> list[SubtaskSegment]:
    samples = [s for s in samples if s]
    if not samples:
        return [SubtaskSegment(0, 0, last, "complete the task")]
    modal_n = Counter(len(s) for s in samples).most_common(1)[0][0]
    matching = [s for s in samples if len(s) == modal_n]
    ref = matching[0]  # phase/evidence/text taken from the first sample at the modal count
    if modal_n <= 1:
        return ref
    boundaries: list[int] = []
    for j in range(modal_n - 1):
        ends = [s[j].end_frame for s in matching]
        boundaries.append(int(round(statistics.median(ends))))
    boundaries = _monotonic(boundaries, last)
    segments: list[SubtaskSegment] = []
    cursor = 0
    for j in range(modal_n):
        end = boundaries[j] if j < modal_n - 1 else last
        end = min(max(cursor, end), last)
        r = ref[j]
        segments.append(SubtaskSegment(
            segment_idx=j, start_frame=cursor, end_frame=end,
            subtask_text=r.subtask_text, phase=r.phase, evidence=r.evidence,
        ))
        cursor = end + 1
    segments[-1].end_frame = last
    return segments


def _monotonic(boundaries: list[int], last: int) -> list[int]:
    out: list[int] = []
    prev = 0
    for b in boundaries:
        b = max(prev + 1, min(int(b), last - 1))
        out.append(b)
        prev = b
    return out


# --------------------------------------------------------------------------- #
# Refinement: dense-window per-boundary transition pinning
# --------------------------------------------------------------------------- #
def _refine_boundaries(
    episode: Episode,
    provider: VLMProvider,
    rubric: Rubric,
    segments: list[SubtaskSegment],
    config: StrategyConfig,
    receipt_dir: Path,
) -> tuple[list[SubtaskSegment], list[ProviderResponse]]:
    task = episode.task or episode.episode_id
    last = episode.num_frames - 1
    calls: list[ProviderResponse] = []
    for i in range(len(segments) - 1):
        # contact-only mode: refine only grasp/release boundaries (the hard contact events)
        if config.refine_contact_only and not (
                _is_contact_phase(segments[i].phase) or _is_contact_phase(segments[i + 1].phase)):
            continue
        boundary = segments[i].end_frame
        lo = max(0, boundary - config.refine_window)
        hi = min(last, boundary + config.refine_window)
        window = _dense_window(lo, hi, config.refine_max_frames)
        if len(window) < 2:
            continue
        frames = episode.frames(window)
        captions = frame_captions(window, episode.fps)
        manifest = frame_manifest(window, episode.fps)
        question = rubric.refine_prompt(
            task=task,
            phase_before=segments[i].phase or segments[i].subtask_text or "previous subtask",
            phase_after=segments[i + 1].phase or segments[i + 1].subtask_text or "next subtask",
            boundary_evidence=segments[i].evidence or "the subtask transition",
            frame_manifest=manifest,
        )
        resp = provider.ask(frames, window, question, receipt_dir / f"refine_b{i}.json",
                            frame_captions=captions)
        calls.append(resp)
        refined = _extract_frame(try_extract_json(resp.answer))
        if refined is None:
            continue
        # Clamp into the window and strictly inside both neighbours.
        lower = segments[i].start_frame + 1
        upper = segments[i + 1].end_frame - 1
        refined = min(max(lo, refined), hi)
        refined = min(max(lower, refined), upper)
        segments[i].end_frame = refined
        segments[i + 1].start_frame = refined + 1
    return segments, calls


def _dense_window(lo: int, hi: int, max_frames: int) -> list[int]:
    full = list(range(lo, hi + 1))
    if len(full) <= max_frames:
        return full
    step = math.ceil(len(full) / max_frames)
    sub = full[::step]
    if sub[-1] != hi:
        sub.append(hi)
    return sub


def _extract_frame(data: Any) -> int | None:
    if isinstance(data, dict):
        for key in ("frame", "frame_index", "end_frame", "transition_frame"):
            if key in data:
                data = data[key]
                break
    try:
        return int(data)
    except (TypeError, ValueError):
        return None
