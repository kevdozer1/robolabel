"""Deterministic, data-derived control annotations (no VLM, no inference).

* ``control_modality`` (per episode) — the action **coordinate frame**: ``"joint"`` (the actions
  are joint-position/velocity targets) vs ``"end-effector"`` (Cartesian end-effector poses),
  decided from the **action feature names** (LeRobot stores them in ``meta/info.json``). This is
  NOT about whether the gripper is involved — an SO-101 arm is joint control whether or not it
  also actuates a gripper. Honest even when constant for a dataset (the SO-arms are all joint).

* ``active_dof`` (per grounded segment) — ``"arm" | "gripper" | "both" | "none"``: which dof
  group moves over the segment (per-dim range normalized to the dim's full-episode range vs
  ``rubric.yaml -> control.active_dof_threshold``). **Optional and off by default** (the run
  config's ``control.active_dof``): on these pick/pour/fold tasks it is *low-discrimination* —
  most manipulation segments move both the arm and the gripper, so it is mostly ``"both"`` and
  carries little signal. Kept for datasets where arm-only vs gripper-only phases are meaningful.

Nothing here is inferred by a model; it is a deterministic function of the action array and the
feature names. See ``CLAIMS.md``.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

# Tokens that mark an end-effector / Cartesian action feature (vs a joint position).
_EE_TOKENS = ("pose", "eef", "tcp", "cartesian", "ee_", "_ee")
_EE_AXES = ("x", "y", "z", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw")


def classify_control_modality(action_names: list[str] | None) -> str | None:
    """'joint' if the action features are joint positions, 'end-effector' if Cartesian/pose.

    Returns None when there are no names to read (honest: we don't guess).
    """
    if not action_names:
        return None
    low = [str(n).lower() for n in action_names]
    ee = 0
    joint = 0
    for n in low:
        comps = n.replace("_", ".").split(".")
        if any(t in n for t in _EE_TOKENS) or any(c in _EE_AXES for c in comps):
            ee += 1
        if "pos" in comps or "joint" in n:
            joint += 1
    if ee > joint:
        return "end-effector"
    if joint > 0:
        return "joint"
    return "end-effector" if ee > 0 else None


def gripper_dims(action_names: list[str] | None, n_dims: int) -> list[int]:
    """Indices of gripper action dims (by name), else the last dim by convention."""
    if action_names:
        dims = [i for i, n in enumerate(action_names) if "gripper" in str(n).lower()]
        if dims:
            return dims
    return [n_dims - 1] if n_dims > 0 else []


def segment_active_dof(actions: np.ndarray, start: int, end: int, grip_dims: list[int],
                       ep_range: np.ndarray, threshold: float) -> str:
    """Classify which dof group moves over [start, end] (inclusive frames)."""
    seg = actions[start:end + 1]
    if seg.shape[0] < 2:
        return "none"
    norm = (seg.max(0) - seg.min(0)) / np.maximum(ep_range, 1e-6)
    moved = norm > threshold
    arm_dims = [d for d in range(actions.shape[1]) if d not in grip_dims]
    arm = bool(any(moved[d] for d in arm_dims))
    grip = bool(any(moved[d] for d in grip_dims))
    return "both" if (arm and grip) else "arm" if arm else "gripper" if grip else "none"


def enrich_control(df, actions_by_ep: dict, action_names: list[str] | None, threshold: float):
    """Write control_modality (episode) + active_dof (per subtask) into a copy of ``df``."""
    df = df.copy()
    for col in ("control_modality", "active_dof"):       # ensure object dtype for string writes
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("object")
    modality = classify_control_modality(action_names)
    for eid in df["episode_id"].astype(str).unique():
        meta_mask = (df["episode_id"].astype(str) == eid) & (df["record_type"] == "episode_metadata")
        df.loc[meta_mask, "control_modality"] = modality
        actions = actions_by_ep.get(eid)
        if actions is None or len(actions) < 2:
            continue
        ep_range = actions.max(0) - actions.min(0)
        grip = gripper_dims(action_names, actions.shape[1])
        sub_mask = (df["episode_id"].astype(str) == eid) & (df["record_type"] == "subtask")
        for idx in df[sub_mask].index:
            s = int(df.at[idx, "start_frame"])
            e = min(int(df.at[idx, "end_frame"]), len(actions) - 1)
            df.at[idx, "active_dof"] = segment_active_dof(actions, s, e, grip, ep_range, threshold)
    return df


def load_actions(repo_id: str, root: str | Path | None = None) -> tuple[dict, list[str] | None]:
    """Read per-episode action arrays + action feature names from a cached LeRobot dataset.

    Reads the ``data/**/*.parquet`` ``action`` column (grouped by ``episode_index``) and the
    ``meta/info.json`` action feature names. Dataset read only — no network if cached.
    """
    import json

    import pandas as pd
    base = Path(root) if root else Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}"))
    names = None
    info = base / "meta" / "info.json"
    if info.exists():
        feat = json.loads(info.read_text(encoding="utf-8")).get("features", {}).get("action", {})
        names = feat.get("names")
        if isinstance(names, dict):                  # some datasets nest names under a key
            names = next((v for v in names.values() if isinstance(v, list)), None)
    files = sorted(glob.glob(str(base / "data" / "**" / "*.parquet"), recursive=True))
    out: dict = {}
    if files:
        frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "action" in frames.columns:
            order = "frame_index" if "frame_index" in frames.columns else frames.columns[0]
            for ep_idx, grp in frames.groupby("episode_index"):
                out[str(int(ep_idx))] = np.stack(grp.sort_values(order)["action"].to_numpy())
    return out, names


def load_states(repo_id: str, root: str | Path | None = None) -> tuple[dict, list[str] | None]:
    """Read per-episode ``observation.state`` arrays + feature names (the physical gripper signal
    for grasp/release snapping). Dataset read only — no network if cached."""
    import json

    import pandas as pd
    base = Path(root) if root else Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}"))
    names = None
    info = base / "meta" / "info.json"
    if info.exists():
        feat = json.loads(info.read_text(encoding="utf-8")).get("features", {}).get("observation.state", {})
        names = feat.get("names")
        if isinstance(names, dict):
            names = next((v for v in names.values() if isinstance(v, list)), None)
    files = sorted(glob.glob(str(base / "data" / "**" / "*.parquet"), recursive=True))
    out: dict = {}
    if files:
        frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "observation.state" in frames.columns:
            order = "frame_index" if "frame_index" in frames.columns else frames.columns[0]
            for ep_idx, grp in frames.groupby("episode_index"):
                out[str(int(ep_idx))] = np.stack(grp.sort_values(order)["observation.state"].to_numpy())
    return out, names
