"""Deterministic, data-derived control annotations (no VLM, no inference).

Two INDEPENDENT axes that are easy to conflate:

* ``control_modality`` (per EPISODE) — the action **coordinate frame**: ``"joint"`` (actions are
  joint-position targets) vs ``"end-effector"`` (Cartesian poses), decided from the action feature
  names (LeRobot stores them in ``meta/info.json``). Dataset-level and typically constant (an
  SO-101 is joint control either way). This is pi0.7's control-modality field. It says nothing
  about which parts move; it is NOT a motion signal.

* ``active_dof`` (per SEGMENT) — the **set of component groups that actually move** over the
  segment, e.g. ``"arm"``, ``"gripper"``, ``"arm+gripper"``, or ``"none"``. Groups are auto-derived
  from the action feature names and generalize to N configurable groups (the default splits the
  gripper out and calls the remainder ``arm``; a dataset with no gripper-named dim simply has no
  gripper group). A group is active iff any of its dims moves through more than a motion threshold
  of that dim's full-episode range within the segment, measured as the **smoothed within-segment
  range** (its excursion). Using the range rather than the net start-to-end displacement is
  deliberate: it catches any real actuation, so a gripper that closes to grasp and one that opens
  to release and then recloses both register. A light moving-average (window ``motion_smooth``)
  rejects single-frame jitter, and a gripper merely HOLDING a fixed position has near-zero range
  and is not counted. A separate, non-pi0.7 descriptor.

Everything here is a deterministic function of the action array and the feature names.
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


def component_groups(action_names: list[str] | None, n_dims: int,
                     groups_cfg: dict[str, list[str]] | None = None,
                     default_group: str = "arm") -> dict[str, list[int]]:
    """Map action dims to named component groups (generalizes to N configurable groups).

    ``groups_cfg`` maps a group name to substring tokens; a dim joins the first group whose token
    is in its (lowercased) feature name. Dims matching no named group fall into ``default_group``.
    With no feature names, every dim falls into ``default_group`` (we never guess a gripper).
    Empty groups are omitted, so a dataset with no gripper-named dim simply has no gripper group.
    """
    if n_dims <= 0:
        return {}
    groups_cfg = groups_cfg or {"gripper": ["gripper"]}
    if not action_names or len(action_names) < n_dims:
        return {default_group: list(range(n_dims))}
    low = [str(x).lower() for x in action_names]
    out: dict[str, list[int]] = {}
    for i, nm in enumerate(low):
        g = next((name for name, toks in groups_cfg.items() if any(t in nm for t in toks)), default_group)
        out.setdefault(g, []).append(i)
    return out


def _excursion(seg: np.ndarray, ep_range: np.ndarray, smooth: int) -> np.ndarray:
    """Per-dim smoothed within-segment range, normalized by the episode range.

    A moving-average (window ``smooth``) rejects single-frame jitter; the range (not the net
    start-to-end displacement) counts any significant excursion, so a gripper that opens to
    release and then recloses still registers as motion.
    """
    if smooth > 1 and seg.shape[0] >= smooth:
        k = np.ones(smooth) / smooth
        seg = np.stack([np.convolve(seg[:, d], k, mode="valid") for d in range(seg.shape[1])], axis=1)
    return (seg.max(axis=0) - seg.min(axis=0)) / np.maximum(ep_range, 1e-6)


def segment_active_groups(actions: np.ndarray, start: int, end: int, groups: dict[str, list[int]],
                          ep_range: np.ndarray, threshold: float, smooth: int = 5) -> str:
    """The set of component groups that move over [start, end] (inclusive), as a ``+``-joined,
    alphabetically-sorted string (e.g. ``"arm+gripper"``), or ``"none"``."""
    seg = actions[start:end + 1]
    if seg.shape[0] < 2:
        return "none"
    m = _excursion(seg, ep_range, smooth)
    active = [name for name, dims in groups.items()
              if dims and max(float(m[d]) for d in dims) > threshold]
    return "+".join(sorted(active)) if active else "none"


def enrich_control(df, actions_by_ep: dict, action_names: list[str] | None, motion: dict):
    """Write control_modality (episode) + the active_dof set (per subtask) into a copy of ``df``.

    ``motion`` is the rubric ``control_motion`` dict: ``{threshold, smooth, groups, default_group}``.
    """
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
        groups = component_groups(action_names, actions.shape[1], motion["groups"], motion["default_group"])
        sub_mask = (df["episode_id"].astype(str) == eid) & (df["record_type"] == "subtask")
        for idx in df[sub_mask].index:
            s = int(df.at[idx, "start_frame"])
            e = min(int(df.at[idx, "end_frame"]), len(actions) - 1)
            df.at[idx, "active_dof"] = segment_active_groups(actions, s, e, groups, ep_range,
                                                             motion["threshold"], motion["smooth"])
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
