"""Tests for proprioception-fused grasp/release boundary snapping (deterministic)."""

from __future__ import annotations

import numpy as np

from robolabel.schema import SubtaskSegment
from robolabel.snap import _directed_transitions, gripper_dim, snap_contact_boundaries


def _grip(n=60):
    """Gripper open→closed(frames 20-44)→open. Convention: low value = closed (S_grip)."""
    g = np.full(n, 1.0)        # open
    g[20:45] = 0.0             # closed (grasped) -> close transition @20, open transition @45
    return g


def test_directed_transitions_grasp_then_release():
    t = _directed_transitions(_grip(), 0.5, 8)
    kinds = [d for _, d in t]
    assert kinds == ["close", "open"]                 # grasp (close) then release (open)


def test_snap_moves_grasp_boundary_to_gripper_close():
    grip = _grip()                                    # close at 20, open at 45
    segs = [SubtaskSegment(0, 0, 15, "approach", phase="approach object"),
            SubtaskSegment(1, 16, 40, "grasp", phase="grasp object"),
            SubtaskSegment(2, 41, 59, "release", phase="release object")]
    out, n = snap_contact_boundaries(segs, grip, window=10)
    assert n == 2                                     # grasp-onset and release-onset both snapped
    assert out[0].end_frame == 20 and out[1].start_frame == 21    # grasp boundary -> gripper close
    assert out[1].end_frame == 45 and out[2].start_frame == 46    # release boundary -> gripper open


def test_snap_no_op_when_gripper_holds():
    # gripper never crosses the threshold (e.g. pour/fold holding the object) -> 0 snaps
    held = np.full(60, 0.9)
    segs = [SubtaskSegment(0, 0, 20, "approach", phase="approach cup"),
            SubtaskSegment(1, 21, 40, "tilt", phase="tilt to pour"),
            SubtaskSegment(2, 41, 59, "lower", phase="lower cup")]
    out, n = snap_contact_boundaries([s for s in segs], held, window=10)
    assert n == 0 and [s.end_frame for s in out] == [20, 40, 59]   # untouched


def test_snap_no_op_when_no_transition_in_window():
    grip = _grip()                                    # close at 20
    # grasp boundary at 50 — gripper close (20) is 30 frames away, outside window 8 -> not snapped
    segs = [SubtaskSegment(0, 0, 50, "approach", phase="approach object"),
            SubtaskSegment(1, 51, 59, "grasp", phase="grasp object")]
    out, n = snap_contact_boundaries(segs, grip, window=8)
    assert n == 0 and out[0].end_frame == 50


def test_gripper_dim():
    assert gripper_dim(["shoulder.pos", "gripper.pos"], 2) == 1
    assert gripper_dim(None, 6) == 5
