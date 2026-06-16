# Claims audit

Every claim robolabel makes publicly (README, `STRATEGY_REPORT.md`, `docs/blog_post.md`,
the drafted issue replies) mapped to the artifact that supports it and a status:

- **verified** — checked by a test or a reproducible number, not dataset-specific.
- **verified-on-one-dataset** — true on SO-101 (and, where noted, the fresh stacking set),
  but measured on one task family with one annotator's gold.
- **mechanical-only** — the code runs / a format check passes, but it is not evidence of
  usefulness.
- **untested** — not demonstrated. Stated as such; never implied otherwise.
- **fixed-and-spot-checked** — a code/format change is verified by tests and a live
  re-annotation, and a frame-level spot-check confirms the result; any residual is named in
  the row. The author's full blind re-grade, where noted, remains the gold-standard check.

The README may only assert **verified** and **verified-on-one-dataset (with the caveat)**
rows. Everything else is described as not-yet-shown.

| # | claim | evidence artifact | status |
|---|---|---|---|
| 1 | Drafts subtask boundaries, episode quality, and subgoal frames for LeRobot datasets with any VLM | `tests/test_labelers_schema.py`, `test_demo.py`; live SO-101 + fresh stacking runs | **verified** |
| 2 | Measures those drafts against a human gold set (boundary IoU, quality agreement, subgoal agreement) | `reliability.py` + `tests/test_calibration.py`; run on the 50-episode gold | **verified** |
| 3 | The gate flags but never drops episodes (`dropped_episode_count` = 0) | `gate.py` + `tests/test_strategy.py::test_gate_quality_outlier...` | **verified** |
| 4 | Frame grounding eliminates the degenerate + uniform-split failure bands | `STRATEGY_REPORT.md` tune (S0 3+9 → grounded 0+0) and **held-out test (S0 5/20 → grounded 0/20)** | **verified-on-one-dataset** |
| 5 | On held-out data the grounded strategy places ~36% more gold boundaries within ±5 frames than S0 (recall 0.307 vs 0.226) | `STRATEGY_REPORT.md` "Boundary placement"; `scripts/compute_metrics.py` | **verified-on-one-dataset** |
| 6 | The strategy layer does **not** improve mean held-out boundary IoU (0.444 vs S0 0.460) | `STRATEGY_REPORT.md` test table | **verified-on-one-dataset** (a negative result, stated) |
| 7 | The stronger model (Pro) cuts catastrophic quality false-negatives 3→1 | `STRATEGY_REPORT.md` quality section | **verified-on-one-dataset** |
| 8 | Quality exact-agreement is near-degenerate here (constant-5 baseline beats both models) | `STRATEGY_REPORT.md` quality table | **verified-on-one-dataset** |
| 9 | The free proprioceptive baseline (S_grip) underperforms the VLM (0.18–0.20 vs 0.40–0.45 IoU) | `STRATEGY_REPORT.md`; `scripts/score_gripper.py` | **verified-on-one-dataset** |
| 10 | Exports to the pinned-LeRobot subtask convention; every frame's `subtask_index` resolves to the correct subtask | `tests/test_export_lerobot.py` + `scripts/consumability_check.py` (all frames pass) | **verified** |
| 11 | A full SARM/VLA dataloader can train off the exported annotations | `docs/consumability.md` — overlay only; per-frame column not written into `data/` | **mechanical-only / not shown** |
| 12 | Evidence strings are factually true of their cited frame ("evidence factual-accuracy") | the `inspect` evidence tab + blind trial → `FRESH_TRIAL_REPORT.md` | **untested until the blind trial is graded** |
| 13 | The findings generalize to a dataset with no S0-anchored gold | fresh stacking set: failure-band rate computable now (objective); boundary/phase/evidence acceptance via the blind trial | **untested until the blind trial is graded** (objective failure-band: see fresh report) |
| 14 | These conditioning annotations improve downstream training | `docs/why.md` — preregistered head-to-head; not shown, and on one careful test, no | **untested (explicitly; on one test, negative)** |
| 15 | Grounded labels name the specific target object (`phase → target`), so disambiguation is possible when several similar objects are present (the "which cube?" gap that blind grading surfaced). Residual: the target *naming rule* on transport/place is not yet uniform (moved-object vs destination) | schema v3 `target` column + validation tests (`test_require_target_*`, `test_terminal_phase_dedupe_*`); re-annotated fresh set `fresh_stacking/grounded_flash_v3`; frame-level spot-check (8/20 eps, 36 segs, `scripts/spotcheck_frames.py`): phases correct, targets present | **fixed-and-spot-checked** (target-naming convention → v0.2; author blind re-grade is the gold-standard check) |

| 16 | Frame-grounding still eliminates the degenerate/uniform failure bands on tasks **outside** pick-and-place (pour, cloth-fold: **0/8** each, both vocab conditions), while the closed pick-and-place phase vocabulary degrades off-task (pour: **17.5%** of segments coerced to `other`; fold: silently mislabeled as `release-place`) where the open-vocab `S2-open` reads sensibly (`pour water`, `perform fold`; 0% `other`) | `FRESH_TRIAL_REPORT.md` → "Cross-task generalization probe"; `probe_metrics.json`; `scripts/run_probe.py` + `probe_metrics.py` | **verified gold-free on 2 new task families** (8 eps each; objective band/vocab metrics + a 10-boundary author spot-check; caveat: pour=sim, fold=real-but-bimanual/one-camera; no human gold) |
| 17 | `control_modality` (`joint`/`end-effector`) is **dataset-derived, not inferred** — it is the action **coordinate frame** (joint targets vs Cartesian poses), read from the action feature names, NOT whether the gripper is involved (an SO-101 is joint control either way). The per-segment `active_dof` is optional/off by default and **low-discrimination** on these tasks (mostly `both`) | `src/robolabel/control.py` + `tests/test_control.py`; detected on pick-place/pour/fold (all `joint`) | **verified (deterministic, no VLM)**; `active_dof` demoted — kept for datasets where arm-only/gripper-only phases matter; `active_dof_threshold` is a documented choice, not validated vs human dof labels |
| 18 | Stores an optional **retrieved** subgoal — the same-phase end frame from a *different* **gate-passed** episode — **alongside** the real keyframe (never replacing it; both are pointers, no image files), for copy-shortcut-free policy eval; robolabel **does not generate** images | `src/robolabel/retrieve.py` (`allowed_sources` = gate-passed) + `tests/test_retrieve.py::test_retrieve_only_from_allowed_sources`; run via `subgoals.retrieval` | **verified the selection is a real same-phase frame from another gate-passed episode** (deterministic/seeded; unmatched left null); **downstream training/eval utility untested** |
| 19 | Deterministic episode **`speed`**: a continuous, **motion-defined, phase-agnostic** descriptor (`active_frames`/`active_seconds`/`active_fraction` from motion onset→offset, + raw `speed_norm`) — generalizes to any task, not tied to named phases. The `fast`/`medium`/`slow` tier is emitted only when corpus-relative (else null). One of pi0.7's two metadata signals; more informative than quality on uniform datasets | `src/robolabel/speed.py` (`active_window`) + `tests/test_modules.py::test_active_window_motion_defined` | **verified (deterministic, no VLM)**; motion threshold is a documented choice (`rubric.yaml -> speed`); training-informativeness unvalidated |
| 20 | Deterministic per-episode **`novelty`** (mean distance to nearest neighbours in a cheap frame embedding) as a diversity/coverage signal | `src/robolabel/novelty.py` + `tests/test_modules.py` | **verified (deterministic, no VLM)**; it measures embedding-space isolation, **not** validated as a training-utility signal |
| 20 | Deterministic per-episode **`novelty`** (mean distance to nearest neighbours in a cheap frame embedding), computed **corpus-pooled** when rescored across datasets | `src/robolabel/novelty.py` + `src/robolabel/corpus.py` + `tests/test_modules.py` | **verified (deterministic, no VLM)**; embedding-space isolation, **not** validated as a training-utility signal |
| 21 | **Curation**: raw `curation_value = f(quality, novelty)` (always emitted) plus an optional value-tiered **overlay** (`full`/`reduced`/`minimal`, or `keep`/`cut`) that never deletes data — precedented by the Smart Black Box (value-tiered storage/compression) and "train on the most valuable ~20%" curation work. Tiers are **corpus-relative and guarded**: pooled across all episodes with global thresholds, left **null** ("insufficient population to tier") on a population too small/homogeneous to tier honestly | `src/robolabel/curation.py` (`tierable`/`assign_tiers`) + `corpus.py` + `tests/test_modules.py::test_curation_tier_guard` + `test_run.py::test_tier_guard_on_small_run`; docs cite precedents | **machinery sound + precedented; tiering no longer fabricated on small same-y runs; downstream training utility UNVALIDATED**; off by default |
| 22 | **Open-vocabulary grounded segmentation is the default** of the `robolabel run` pipeline (closed-vocab `S2` stays available via `vocabulary: closed`); the frozen S0–S4 ablation, eval split, and S0 are untouched | `src/robolabel/run.py::resolve_strategy` + `tests/test_run.py`; cross-task evidence in row 16 | **verified (default wiring + tests)**; the open-vocab *quality* evidence is row 16's (gold-free, 2 task families) |
| 23 | **Open-vocab segmentation de-primed of a hallucinated terminal retract** (the prompt no longer lists "retract" as an example or presumes a final wind-down phase; explicit "do not invent a retract you don't see"); the **terminal wind-down collapse** now merges consecutive trailing retract/withdraw/return/home phases even with *different* labels | `rubric.yaml grounded_label_prompt_open` + `src/robolabel/labelers/segmentation.py::_dedupe_trailing_phases` (`_is_winddown`) + `tests/test_strategy.py::test_terminal_winddown_collapses_different_labels`; re-probe in `FRESH_TRIAL_REPORT.md` | **verified: collapse logic by test; pour terminal-retract hallucination cut 8/8 → 3/8 on the re-probe** (gold-free, ≤8 eps) |
| 24 | **Grasp/release boundary timing is the known precision limit** of grounded segmentation: a bounded contact-only dense-window refinement attempt did not move boundaries materially closer to the human gold (recall@±5 0.211 → 0.211; MAE 3.75 → 2.80 on matches). The `refine_contact_only` flag exists but is off by default | `src/robolabel/labelers/segmentation.py` (`refine_contact_only`, `_is_contact_phase`); 8-ep pick-place vs `so101_gemini/gold.json`; `FRESH_TRIAL_REPORT.md` "(3c)" | **verified-on-one-dataset (objective vs gold)**: refinement available but unhelpful here; grasp/release timing documented as the limit |

## Caveats that must travel with the "verified-on-one-dataset" rows

- One task family (SO-101 tabletop pick-and-place; fresh set is SO-100 stacking).
- One annotator's gold, 50 episodes; the held-out test is 20.
- The gold was built by **correcting S0 drafts**, so it is S0-anchored — every boundary
  metric gives S0 a quiet home-field advantage (which makes rows 4–5 *stronger*, since
  grounding wins boundary placement despite it).
- The +0.05 tune-selection margin was 0.003 — small-sample fragile.

## What the README is allowed to say

Rows **1, 2, 3, 10** as plain capabilities. Rows **4, 5, 6, 7, 8, 9** as
"how well it works, measured on one dataset," each with a one-line caveat. Rows **11, 12,
13, 14** only under "what is not yet shown." The fresh-trial rows (12, 13) flip to
verified-on-two-datasets **only after** the blind trial in `REVIEW_GUIDE.md` is graded and
`FRESH_TRIAL_REPORT.md` is filled.
