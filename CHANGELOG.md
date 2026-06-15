# Changelog

## 0.2.0 (unreleased) — cross-task generalization + open vocabulary

- **Deterministic conditioning fields (schema v4, additive; `robolabel enrich`).** Three
  data-derived columns, no VLM: `control_modality` (`joint`/`end-effector`, from the action
  feature names), per-segment `active_dof` (`arm`/`gripper`/`both`, from which action dims move
  beyond a threshold), and an optional **retrieved subgoal** (`retrieved_subgoal_episode_id` /
  `_frame_idx` — a same-phase end frame from a *different* episode, stored **alongside** the real
  keyframe, never replacing it). The retrieved subgoal is for copy-shortcut-free policy eval;
  robolabel **selects** real frames and **does not generate** images (`docs/why.md`). v1/v2/v3
  files still read. `control.py`, `retrieve.py`, `robolabel enrich --control --retrieve-subgoals`.
- **Presentation GIF** (`docs/figures/grounded_annotations.gif`) — grounded annotations on three
  tasks (pick-place, pour, fold): `phase → target` + timeline/playhead, quality, the real
  end-of-sub-step subgoal keyframes ("selected — not generated") + retrieved subgoals, and the
  control line. `scripts/make_gif.py`.
- **`robolabel gallery`** — load several task datasets into one task-grouped view (the
  grounded lane shown per task), so you can eyeball grounded across pick-place / stacking /
  pour / fold in one browsable page. Multi-source frame routing (each task keeps its own
  dataset + camera + episode range); dependency-free and offline like `inspect`.
  `scripts/make_gallery.py` writes the config (`robolabel gallery --config gallery.json`).
  Also adds **`inspect --episodes 0-7`** to load only the annotated episode range instead of
  the whole dataset.
- **Open-vocabulary grounded variant (`S2-open`).** A first-class strategy that keeps S2's
  frame-grounding + per-boundary evidence + required `target`, but turns the closed phase
  vocabulary OFF so the model names each phase in free text ("tilt to pour", "grasp corner").
  The closed-vocab S2 default is untouched (a dedicated `open_vocabulary` flag selects the open
  label prompt and relaxes the target-optional exemption to retract/withdraw-like phases). 6
  new tests; `--strategy S2-open`.
- **Cross-task generalization probe (pour + cloth-fold).** A gold-free check of whether
  frame-grounding still eliminates the degenerate/uniform failure bands *outside*
  pick-and-place, and whether the hand-authored pick-place phase vocabulary degrades where the
  grounding does not. Datasets, exact numbers, and the finding: `FRESH_TRIAL_REPORT.md` →
  "Cross-task generalization probe", `CLAIMS.md`, `ROADMAP.md`. Helper scripts:
  `scripts/discover_datasets.py`, `scripts/run_probe.py`, `scripts/probe_metrics.py`.
- **VLA-vs-world-model scoping.** README + `docs/why.md` now state plainly that these
  annotations target VLA subtask conditioning + dataset curation, **not** world-model training
  (the one careful test of that was negative) — with subgoal keyframes the one cross-paradigm
  output.

## 0.1.0 (unreleased)

First public release. Highlights:

- **Acceptance-review kit.** `robolabel inspect` — a verification viewer (multi-track
  boundary timeline, evidence-vs-cited-frame tab, per-episode metric panel, sort/filter,
  and a blind-grading mode); `robolabel query` — retrieve segments by phase as a contact
  sheet + list gate `needs_review` episodes; `robolabel trial-report` — tally a blind trial
  by strategy (incl. the **evidence factual-accuracy** metric). Plus `CLAIMS.md` (claim →
  evidence → status), `REVIEW_GUIDE.md` (a 90-minute author sign-off session), and
  `docs/consumability.md` (the annotate→export→reload→resolve chain).
- **Generalization tested on a second dataset, paired.** `lerobot/svla_so100_stacking`
  (apache-2.0, never-touched): grounded-Flash and S0-Flash both ran on the same 20 episodes —
  baseline **8/20** failure-band, grounded **0/20**. Failure-band elimination now verified on
  two datasets (`FRESH_TRIAL_REPORT.md`). During the first pass the Gemini key hit a credit
  limit (HTTP 429); the resilient `annotate_source` checkpointing kept all completed episodes
  — confirming the resilience behaves correctly under a real mid-run failure.
- **Fix: NaN in empty parquet columns** (found on the fresh dataset). The baseline S0 has no
  `phase` / `boundary_evidence`, which read back from parquet as float `NaN` (truthy) — the
  inspect viewer would have rendered the literal string "nan". `segments_from_records` now
  coerces NaN/empty to `None`; regression test added.
- **Structured grounded labels (`phase → target`), schema v3.** Blind grading of the
  fresh-dataset clips surfaced that grounded's first label was *under*specified — "approach"
  with both a red and a blue cube in frame doesn't say which. The acceptance kit caught this
  pre-launch. Grounded strategies (S2+) now emit a required `target` slot (the object/destination
  named from the scene; `none` only for `retract`) alongside the closed-vocabulary `phase`, and
  display everywhere as `phase → target`. Schema **v3** adds one additive `target` column —
  **v1/v2 files still read**. Validation rejects an empty target on a non-retract phase (with the
  capped re-prompt) and collapses the common "two retract steps" error via terminal-phase dedupe.
  The frozen SO-101 ablation numbers are unchanged (reconstruction pins `require_target=False`).


- **Annotation-strategy layer (S0–S4)** between adapter and provider: frame-indexed
  grounding, closed phase vocabulary, dense-window boundary refinement, and
  self-consistency. Off by default (S0 reproducible). Measured on SO-101 in
  `STRATEGY_REPORT.md`.
- **Min-granularity rule demoted to a configurable policy.** The S2+ "reject a
  segmentation below the minimum number of segments" rule is now
  `min_granularity_policy`, **defaulting to `warn`**: a below-floor answer is
  *accepted* and flagged as a `single_segment_candidate` rather than re-prompted away.
  This is the **ep7** lesson from the ablation — the human gold for episode 7 is a
  *single* continuous segment, which the old hard-reject (`reject`) could never match
  because it forced ≥3 phases. Set `min_granularity_policy="reject"` to reproduce
  `STRATEGY_REPORT.md` exactly (the ablation was run under the prior reject default;
  the difference is immaterial on the SO-101 data, where grounded cells had 0
  degenerate episodes).
- **Failure-band gate detectors** (degenerate single-segment, near-uniform split) and a
  **quality-outlier `needs_review`** policy. The gate flags, never drops.
- **`export --format lerobot`**: writes our subtask boundaries into the pinned-lerobot
  `meta/subtasks.parquet` convention (+ a per-episode boundary table), round-trip-tested
  through lerobot's own `load_subtasks`.
- **Free baselines**: `S_grip` (proprioceptive gripper/EE segmentation, zero-API) and
  the uniform-fifths blind baseline, both scored against the human gold as honest floors.
- **Reliability + calibration** (browser review GUI, `reliability_report`) and per-call
  provider **receipts + cost**.
- **Cost-projection fix**: the budget-projection probe now bypasses the receipt cache (or
  falls back to the price table), so per-call cost is measured rather than read as $0 off
  a cached receipt (the wrinkle documented in the first ablation run).
- **Renamed to `robolabel`** — the import package, distribution name, and CLI command
  are all `robolabel` (previously the working name `robovid_conditioner`).
