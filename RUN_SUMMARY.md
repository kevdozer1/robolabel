# Run summary (conditioning fields round-out — schema v4)

Additive, honest conditioning fields. No world model, no image generation; the grounded
strategy is untouched and remains the validated core.

## Fields added (schema v4, additive — v1/v2/v3 still read)

- **`control_modality`** (per episode) — `joint` vs `end-effector`.
- **`active_dof`** (per subtask) — `arm` / `gripper` / `both` / `none`.
- **`retrieved_subgoal_episode_id` / `retrieved_subgoal_frame_idx`** (per subgoal) — an
  optional retrieved subgoal stored **alongside** the real `subgoal_frame_idx`, never replacing it.

## How they're computed (one line each — so captions/docs are accurate)

- **control_modality** — read the dataset's `action` feature names: all `*.pos` ⇒ `joint`,
  Cartesian/pose axes (x/y/z/roll/…) ⇒ `end-effector`. Deterministic; `None` if no names.
- **active_dof** — per segment, a dof dim "moved" if its range over the segment exceeds
  `rubric.yaml control.active_dof_threshold` (0.15) of that dim's full-episode range; arm vs
  gripper split by the `gripper`-named action dim (else last dim). Deterministic, no VLM.
- **real subgoal** — unchanged: the real end-of-sub-step frame of the segment (ground truth).
- **retrieved subgoal** — the end frame of a **same-phase** segment from a **different** episode,
  chosen by nearest cheap frame embedding (12×12 grayscale) or a seeded random pick; left null
  when no other episode shares the phase. robolabel **selects** real frames; it does **not**
  generate images.

## Artifacts

- `src/robolabel/control.py`, `retrieve.py`; `robolabel enrich --control --retrieve-subgoals`.
- `docs/figures/grounded_annotations.gif` — 3 panels (pick-place ep7, pour ep5, fold ep4):
  `phase → target` + timeline/playhead, quality, real keyframes ("selected — not generated") +
  retrieved subgoals, and the control line. `scripts/make_gif.py`.
- Docs: `SCHEMA.md` (v4 columns), README (GIF + scope note), `docs/why.md` (copy-shortcut /
  retrieved-vs-generated, π0.7 reference, no image gen), `CLAIMS.md` rows 17–18, CHANGELOG.

## State

- **114 tests pass** (+7: control + retrieve), ruff clean. Frozen SO-101 ablation numbers, the
  eval split, S0, and the closed-vocab default all untouched. `.env` git-ignored.
- **New API spend: $0.015** — one episode (pick-place ep7, grounded S2 Flash); the existing
  pour/fold probe sets were reused; everything else is dataset reads + frame extraction.
- Nothing pushed or published.

## What still needs your eyes

1. **The GIF** (`docs/figures/grounded_annotations.gif`, ~3 MB) — eyeball it and decide if it's
   the README hero you want (size/layout are easy to tune in `scripts/make_gif.py`).
2. **`active_dof_threshold = 0.15`** is a documented choice (CLAIMS row 17), not validated
   against human DoF labels — confirm it reads right on your tasks (it currently gives the
   sensible `pour water → arm`, `retract → arm`, manipulation → `both`).
3. The **retrieved subgoal is a selection, not a proven win** — its downstream training/eval
   benefit is explicitly untested (CLAIMS row 18).
4. Whether to **commit the GIF** and **push** (still your call from the prior run).
