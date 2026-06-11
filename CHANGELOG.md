# Changelog

## 0.1.0 (unreleased)

First public release. Highlights:

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
- **Brand/CLI is `robolabel`.** (The import package is still `robovid_conditioner` pending
  the rename — see `docs/launch_checklist.md`.)
