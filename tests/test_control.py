"""Tests for deterministic control annotations (control_modality + active_dof)."""

from __future__ import annotations

import numpy as np

from robolabel.control import (
    classify_control_modality,
    enrich_control,
    gripper_dims,
    segment_active_dof,
)
from robolabel.schema import (
    EpisodeAnnotation,
    EpisodeMetadata,
    SubtaskSegment,
    episode_records,
    to_dataframe,
)

JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
EE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw", "gripper.pos"]


def test_classify_control_modality():
    assert classify_control_modality(JOINT_NAMES) == "joint"
    assert classify_control_modality(EE_NAMES) == "end-effector"
    assert classify_control_modality(None) is None
    assert classify_control_modality([]) is None


def test_gripper_dims():
    assert gripper_dims(JOINT_NAMES, 6) == [5]
    assert gripper_dims(None, 6) == [5]           # convention: last dim
    assert gripper_dims(["a", "b"], 2) == [1]


def test_segment_active_dof():
    # 6 dims (0..4 arm, 5 gripper); episode range = 1.0 on every dim.
    n = 40
    actions = np.zeros((n, 6), dtype="float32")
    actions[:, 0] = np.linspace(0, 1, n)          # arm dim 0 sweeps full range
    actions[:, 5] = np.linspace(0, 1, n)          # gripper sweeps full range
    ep_range = actions.max(0) - actions.min(0)
    grip = [5]
    # first half: arm + gripper both move a lot -> both
    assert segment_active_dof(actions, 0, 39, grip, ep_range, 0.15) == "both"
    # a tiny window where neither moves beyond threshold -> none
    flat = np.zeros((10, 6), dtype="float32")
    assert segment_active_dof(flat, 0, 9, grip, np.ones(6), 0.15) == "none"
    # only gripper moves
    g = np.zeros((20, 6), dtype="float32")
    g[:, 5] = np.linspace(0, 1, 20)
    assert segment_active_dof(g, 0, 19, grip, np.ones(6), 0.15) == "gripper"
    # only arm moves
    a = np.zeros((20, 6), dtype="float32")
    a[:, 2] = np.linspace(0, 1, 20)
    assert segment_active_dof(a, 0, 19, grip, np.ones(6), 0.15) == "arm"


def test_enrich_control_writes_fields():
    ann = EpisodeAnnotation(
        episode_id="0", task="t", num_frames=40, fps=30.0, provider="mock", model="mock",
        metadata=EpisodeMetadata(quality=4),
        subtasks=[SubtaskSegment(0, 0, 19, "grasp", phase="grasp"),
                  SubtaskSegment(1, 20, 39, "retract", phase="retract")],
    )
    df = to_dataframe([ann])
    actions = np.zeros((40, 6), dtype="float32")
    actions[0:20, 5] = np.linspace(0, 1, 20)      # gripper active in segment 0
    actions[20:40, 1] = np.linspace(0, 1, 20)     # arm active in segment 1
    out = enrich_control(df, {"0": actions}, JOINT_NAMES, 0.15)
    rec = episode_records(out, "0")
    assert rec["metadata"]["control_modality"] == "joint"
    dofs = {int(s["segment_idx"]): s["active_dof"] for s in rec["subtasks"]}
    assert dofs[0] == "gripper" and dofs[1] == "arm"
