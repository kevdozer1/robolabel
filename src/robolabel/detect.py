"""Auto-detect run parameters so a standard LeRobot dataset needs ZERO user config.

From the LeRobot metadata we read: the camera keys, fps, the control space (joint vs
end-effector — the action *coordinate frame*, from the action feature names), and the
arm-vs-gripper action dims. Non-LeRobot inputs (the DirectoryAdapter, raw mp4 folders) carry
no such metadata, so they pass a tiny explicit config instead (see ``PORTING.md``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .control import classify_control_modality, gripper_dims


@dataclass
class DetectedParams:
    camera_key: str | None = None
    camera_keys: list[str] = field(default_factory=list)
    fps: float = 30.0
    control_space: str | None = None        # 'joint' | 'end-effector' (the action coordinate frame)
    action_names: list[str] | None = None
    n_action_dims: int = 0
    arm_dims: list[int] = field(default_factory=list)
    gripper_dims: list[int] = field(default_factory=list)
    phase_vocabulary: list[str] | None = None   # only the directory fallback may override the rubric

    def summary(self) -> dict:
        return {"camera_key": self.camera_key, "fps": self.fps, "control_space": self.control_space,
                "n_action_dims": self.n_action_dims, "arm_dims": self.arm_dims,
                "gripper_dims": self.gripper_dims}


def _action_feature(meta) -> dict:
    feats = getattr(meta, "features", None) or {}
    return feats.get("action", {}) if isinstance(feats, dict) else {}


def detect_lerobot(source, action_names: list[str] | None = None) -> DetectedParams:
    """Detect from a built ``LeRobotAdapter`` (its ``.meta`` + resolved ``.camera_key``)."""
    meta = source.meta
    feat = _action_feature(meta)
    names = action_names if action_names is not None else feat.get("names")
    if isinstance(names, dict):                          # some datasets nest names
        names = next((v for v in names.values() if isinstance(v, list)), None)
    shape = feat.get("shape") or ([len(names)] if names else [0])
    n = int(shape[-1]) if shape else (len(names) if names else 0)
    grip = gripper_dims(names, n)
    return DetectedParams(
        camera_key=getattr(source, "camera_key", None),
        camera_keys=list(getattr(meta, "camera_keys", []) or []),
        fps=float(getattr(meta, "fps", 30.0) or 30.0),
        control_space=classify_control_modality(names),
        action_names=names, n_action_dims=n,
        gripper_dims=grip, arm_dims=[i for i in range(n) if i not in grip],
    )


def detect_directory(config_path: str | Path | None, fps: float = 10.0) -> DetectedParams:
    """Non-LeRobot fallback: read the tiny explicit config (see ``PORTING.md``).

    ``{control_space, arm_dims, gripper_dims, phase_vocabulary?}`` — none of which can be
    inferred from a bare folder of mp4s. Returns sensible empties when no config is given.
    """
    cfg: dict = {}
    if config_path and Path(config_path).exists():
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    arm = list(cfg.get("arm_dims", []) or [])
    grip = list(cfg.get("gripper_dims", []) or [])
    return DetectedParams(
        camera_key=cfg.get("camera_key"), fps=float(cfg.get("fps", fps)),
        control_space=cfg.get("control_space"), action_names=cfg.get("action_names"),
        n_action_dims=len(arm) + len(grip), arm_dims=arm, gripper_dims=grip,
        phase_vocabulary=cfg.get("phase_vocabulary"),
    )
