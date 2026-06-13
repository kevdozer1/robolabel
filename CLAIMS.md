# Claims audit

Every claim robolabel makes publicly (README, `STRATEGY_REPORT.md`, `docs/blog_post.md`,
the drafted issue replies) mapped to the artifact that supports it and a status:

- **verified** — checked by a test or a reproducible number, not dataset-specific.
- **verified-on-one-dataset** — true on SO-101 (and, where noted, the fresh stacking set),
  but measured on one task family with one annotator's gold.
- **mechanical-only** — the code runs / a format check passes, but it is not evidence of
  usefulness.
- **untested** — not demonstrated. Stated as such; never implied otherwise.
- **fixed-and-spot-checked-pending** — a code/format change is verified by tests and a live
  re-annotation, but the *quality* of the result still awaits the author's spot-check.

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
| 15 | Grounded labels name the specific target object (`phase → target`), so disambiguation is possible when several similar objects are present (the "which cube?" gap that blind grading surfaced) | schema v3 `target` column + validation tests (`test_require_target_*`, `test_terminal_phase_dedupe_*`); re-annotated fresh set `fresh_stacking/grounded_flash_v3` | **fixed-and-spot-checked-pending** |

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
