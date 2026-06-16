# Porting robolabel to a dataset

## Standard LeRobot dataset — you provide nothing

For a LeRobot dataset, robolabel **auto-detects everything it needs from the metadata** — you
give only `source: lerobot` and the `target` repo id (or local path). Auto-detected, with zero
config:

| detected | from |
|---|---|
| **camera key** (`camera_key: auto`) | `meta.camera_keys` (first stream, or name your own) |
| **fps** | `meta.fps` |
| **control space** (`joint` vs `end-effector`) | the `action` feature **names** in `meta/info.json` |
| **arm vs gripper action dims** | action dim names containing `gripper` → gripper; the rest → arm |

```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace }
```

That's the whole story for LeRobot. (`robolabel run` prints what it detected, so you can verify.)

## Non-LeRobot input (DirectoryAdapter, raw mp4 folders) — one tiny config

A bare folder of videos carries none of that metadata, so the **only** thing a non-LeRobot input
needs is a small JSON describing the action layout (and, optionally, a task-specific phase
vocabulary). Point at it with `run.dataset.directory_config`:

```json
{
  "control_space": "joint",
  "arm_dims": [0, 1, 2, 3, 4],
  "gripper_dims": [5],
  "phase_vocabulary": ["approach", "grasp", "transport", "release-place", "retract"]
}
```

```yaml
run:
  dataset:
    source: directory
    target: ./my_episodes            # folder of per-episode mp4s / frame dirs
    directory_config: ./my_dataset.json
```

- `control_space`: `joint` or `ee` (the action coordinate frame — see `control_modality` in
  `SCHEMA.md`). Only needed if you enable the `control` module.
- `arm_dims` / `gripper_dims`: 0-based indices into your action vector. Only needed for `control`
  (and `active_dof`) and `speed`.
- `phase_vocabulary` (optional): only if you want **closed**-vocabulary segmentation on a new
  task family; the default open-vocab path needs nothing here.

If you enable only `segmentation` + `quality` (the default), even a non-LeRobot folder needs no
config at all — those modules read frames, not actions.
