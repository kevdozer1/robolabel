"""S_grip proprioceptive segmenter — offline unit tests on synthetic state."""

from __future__ import annotations

import numpy as np

from robovid_conditioner.labelers.gripper_baseline import segment_from_state
from robovid_conditioner.rubric import load_rubric


def _pick_place_state(n=120):
    """Synthetic SO-101-like state: arm moves, pauses, grips, transports, releases."""
    arm = np.zeros((n, 5))
    # arm position ramps then plateaus in phases (so speed dips at phase ends)
    t = np.linspace(0, 1, n)
    arm[:, 0] = np.piecewise(t, [t < 0.25, (t >= 0.25) & (t < 0.5), (t >= 0.5) & (t < 0.75), t >= 0.75],
                             [lambda x: x, 0.25, lambda x: 0.25 + (x - 0.5), 0.75])
    grip = np.full(n, 20.0)         # open
    grip[30:90] = 2.0               # closed during grasp+transport (frames 30..89)
    state = np.column_stack([arm, grip])
    return state


def test_gripper_baseline_produces_ordered_phase_segments():
    cfg = load_rubric().gripper_baseline
    vocab = load_rubric().phase_vocabulary
    segs = segment_from_state(_pick_place_state(), cfg, vocab)
    # contiguous, full coverage
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == 119
    for a, b in zip(segs, segs[1:], strict=False):
        assert b.start_frame == a.end_frame + 1
    # phases drawn from the closed vocabulary, in order
    assert all(s.phase in vocab for s in segs)
    assert segs[0].phase == "approach"
    # the grasp boundary (~frame 30) and release (~frame 90) shape the segmentation
    assert 2 <= len(segs) <= 5
    boundaries = [s.end_frame for s in segs[:-1]]
    assert any(25 <= b <= 35 for b in boundaries)   # grasp transition captured


def test_gripper_baseline_degenerate_on_flat_signal():
    # No gripper motion, no arm motion -> a single (degenerate) segment, not a crash.
    state = np.ones((50, 6))
    segs = segment_from_state(state, {}, ["approach", "grasp"])
    assert len(segs) == 1
    assert segs[0].start_frame == 0 and segs[0].end_frame == 49


def test_gripper_baseline_zero_api_no_provider_needed():
    # Sanity: the segmenter is a pure function of the state array (no VLM/provider).
    segs = segment_from_state(_pick_place_state(60), {"min_segment_frames": 3}, None)
    assert segs and all(s.evidence for s in segs)
