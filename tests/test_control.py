"""Tests for deterministic control annotations (control_modality + active_dof set)."""

from __future__ import annotations

import numpy as np

from robolabel.control import (
    classify_control_modality,
    component_groups,
    enrich_control,
    gripper_dims,
    segment_active_groups,
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
MOTION = {"threshold": 0.25, "smooth": 5, "groups": {"gripper": ["gripper"]}, "default_group": "arm"}


def test_classify_control_modality():
    assert classify_control_modality(JOINT_NAMES) == "joint"
    assert classify_control_modality(EE_NAMES) == "end-effector"
    assert classify_control_modality(None) is None
    assert classify_control_modality([]) is None


def test_gripper_dims():
    assert gripper_dims(JOINT_NAMES, 6) == [5]
    assert gripper_dims(None, 6) == [5]           # convention: last dim
    assert gripper_dims(["a", "b"], 2) == [1]


def test_component_groups_generalizes():
    # single arm + gripper: gripper split out, remainder -> default 'arm'
    assert component_groups(JOINT_NAMES, 6) == {"arm": [0, 1, 2, 3, 4], "gripper": [5]}
    # no dim named 'gripper' (pour) -> everything is 'arm', NO gripper group is invented
    pour = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    assert component_groups(pour, 6) == {"arm": [0, 1, 2, 3, 4, 5]}
    # bimanual: both grippers collapse into the gripper group (generalizes to N members)
    bi = ["left_shoulder_pan.pos", "left_gripper.pos", "right_shoulder_pan.pos", "right_gripper.pos"]
    gb = component_groups(bi, 4)
    assert gb["gripper"] == [1, 3] and gb["arm"] == [0, 2]
    # no names -> we never guess a gripper; all dims are the default group
    assert component_groups(None, 3) == {"arm": [0, 1, 2]}


def test_segment_active_groups_set():
    n = 40
    groups = {"arm": [0, 1, 2, 3, 4], "gripper": [5]}
    er = np.ones(6, dtype="float32")
    # arm and gripper both transition -> the set {arm, gripper}
    a = np.zeros((n, 6), dtype="float32")
    a[:, 0] = np.linspace(0, 1, n)
    a[:, 5] = np.linspace(0, 1, n)
    assert segment_active_groups(a, 0, n - 1, groups, er, 0.15) == "arm+gripper"
    # only the gripper transitions -> "gripper"
    g = np.zeros((n, 6), dtype="float32")
    g[:, 5] = np.linspace(0, 1, n)
    assert segment_active_groups(g, 0, n - 1, groups, er, 0.15) == "gripper"
    # nothing moves -> "none"
    assert segment_active_groups(np.zeros((10, 6), "float32"), 0, 9, groups, er, 0.15) == "none"


def test_held_gripper_vs_release_excursion():
    """A gripper that HOLDS a position (or jitters for one frame) is not active, but a gripper that
    opens to release and then recloses IS: we measure the within-segment excursion, not the net
    start-to-end displacement."""
    n = 40
    groups = {"arm": [0, 1, 2, 3, 4], "gripper": [5]}
    er = np.ones(6, dtype="float32")
    # gripper held closed at a fixed position while the arm moves -> {arm} only
    held = np.zeros((n, 6), dtype="float32")
    held[:, 1] = np.linspace(0, 1, n)     # arm transports
    held[:, 5] = 0.8                       # gripper held closed at a fixed position
    assert segment_active_groups(held, 0, n - 1, groups, er, 0.25) == "arm"
    # single-frame jitter spike on the gripper -> smoothed away, not active
    spike = np.zeros((n, 6), dtype="float32")
    spike[:, 2] = np.linspace(0, 1, n)
    spike[n // 2, 5] = 1.0
    assert segment_active_groups(spike, 0, n - 1, groups, er, 0.25) == "arm"
    # gripper opens for a sustained stretch then recloses (a release) -> gripper IS active
    release = np.zeros((n, 6), dtype="float32")
    release[:, 1] = np.linspace(0, 1, n)   # the arm also moves during release
    release[15:28, 5] = 1.0                # gripper opens ~13 frames, returns to closed
    assert segment_active_groups(release, 0, n - 1, groups, er, 0.25) == "arm+gripper"


def test_enrich_control_writes_fields():
    ann = EpisodeAnnotation(
        episode_id="0", task="t", num_frames=40, fps=30.0, provider="mock", model="mock",
        metadata=EpisodeMetadata(quality=4),
        subtasks=[SubtaskSegment(0, 0, 19, "grasp", phase="grasp"),
                  SubtaskSegment(1, 20, 39, "retract", phase="retract")],
    )
    df = to_dataframe([ann])
    actions = np.zeros((40, 6), dtype="float32")
    actions[0:20, 5] = np.linspace(0, 1, 20)      # gripper changes in segment 0
    actions[20:40, 1] = np.linspace(0, 1, 20)     # arm moves in segment 1
    out = enrich_control(df, {"0": actions}, JOINT_NAMES, MOTION)
    rec = episode_records(out, "0")
    assert rec["metadata"]["control_modality"] == "joint"
    dofs = {int(s["segment_idx"]): s["active_dof"] for s in rec["subtasks"]}
    assert dofs[0] == "gripper" and dofs[1] == "arm"
