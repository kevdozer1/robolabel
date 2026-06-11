# Release readiness

Status: **beta, pre-publication.** Honest assessment of what works, what is only
smoke-tested, what is missing, and what to fix first.

## What works (tested)

- **Core pipeline, offline.** `robovid_conditioner demo` runs end to end in ~1s with no API
  key (synthetic episodes → mock provider → valid `annotations.parquet` → gate).
  Covered by `tests/test_demo.py` and the CI integration step.
- **Provider layer.** Two-stage observe→label flow, raw-response receipts,
  per-call cost, contact sheet, and a credential loader that names the exact
  missing env var. Registry + one-file-per-provider. Mock fully tested.
- **Live Gemini provider (dogfooded).** Ran `robovid_conditioner annotate
  --provider gemini` against the real `lerobot/svla_so101_pickplace` dataset and
  got sensible labels (5-segment approach→grasp→move→release→retreat
  segmentation, quality/mistake judgments, subgoal frames) with usage-based cost
  recorded (~$0.015/episode for Gemini 2.5 Flash). JSON parsing, frame encoding,
  the two-stage flow, and cost estimation all work against the live API.
- **Rubric as config.** All prompts, the 1–5 quality scale, and gate thresholds
  load from `rubric.yaml`; custom rubrics via `--rubric`. Tested.
- **Schema.** Deterministic, versioned `annotations.parquet` (episode metadata /
  subtask / subgoal). Round-trip and determinism tested.
- **Adapters.** `DirectoryAdapter` (frame dirs) tested offline. `LeRobotAdapter`
  verified end to end against the real Apache-2.0 dataset
  `lerobot/svla_so101_pickplace` (50 eps, 30 fps, 480×640 frames) — both with the
  mock provider (tests) and with live Gemini (dogfood) → valid parquet, using the
  installed **lerobot 0.4.4** API.
- **Calibration.** Gold sets keep human labels in a separate block from VLM labels
  and never overwrite either side (tested). Reliability metrics (subtask boundary
  temporal IoU, quality exact / within-one, subgoal frame agreement) tested.
- **Browser review GUI.** A stdlib `http.server` single-page app (no Streamlit)
  that plays the clip, scrubs frame by frame, highlights the active subtask with
  the playhead, and sets a subtask boundary / subgoal frame from the current
  frame. Frames are served as exact per-index JPEGs from the adapter (works for
  LeRobot too). Tested via the `ReviewSession` data layer and a real HTTP
  round-trip (`tests/test_review_server.py`, runs in CI with core deps), and
  verified live on the 50-episode SO-101 run (50 episodes, real frames served in
  ~0.05 s each). Human edits land in the gold block; the VLM auto labels are
  never touched.
- **Gate.** Collapsed-score, repeated-text, object-grounding, and score↔reason
  contradiction flags, plus the failure-band detectors (degenerate single-segment,
  near-uniform split) and the quality-outlier `needs_review` policy. The gate is
  advisory: it flags, never drops (`dropped_episode_count` always 0). Thresholds
  from the rubric. Tested.
- **Annotation strategy layer (S0–S4).** Config-selectable (`--strategy`),
  cumulative grounding → closed phase vocabulary + min granularity → dense-window
  boundary refinement → self-consistency. Off by default (S0 baseline is
  bit-for-bit reproducible); resolved config recorded in `strategy.json`; schema
  bumped to v2 (adds `phase`, `boundary_evidence`, `strategy`; v1 still reads).
  Schema validation, detectors, strategy configs, and S0–S4 segmentation are
  tested offline with the mock provider; the grounded prompts live in
  `rubric.yaml`. **The live SO-101 ablation (the actual numbers) is pending — see
  "Smoke-tested / unverified only".**
- **Strategy eval harness.** `scripts/eval_strategies.py` scores every
  (strategy, model) cell with the existing `reliability_report` against the frozen
  `eval/so101_split.json` (30 tune / 20 test, seeded). Its scoring plumbing has an
  offline `--self-test`; the pure scoring functions are unit-tested.
- **Quality bar.** `ruff` clean; 56 tests pass on Python 3.10; CI matrix 3.10/3.12.

## Smoke-tested / unverified only

- **Strategy ablation numbers (STRATEGY_REPORT.md).** The S0–S4 strategy *system*
  is built and tested offline; the live SO-101 ablation that fills the
  strategy × model tables (Gemini Flash + a stronger model, on the frozen
  tune/test split) has not been run yet. The methodology and protocol are written;
  the result tables are marked pending. Run `scripts/eval_strategies.py` to fill them.
- **Live OpenAI calls.** The Responses-API request/response handling is written
  against the documented shape but has not been run against the live API (only
  Gemini was dogfooded). Verify before claiming OpenAI support.
- **Qwen local provider.** Imports and registers without the extra; the actual
  generate path is unrun here (needs GPU + the `qwen` extra).
- **mp4 DirectoryAdapter path.** Frame-directory mode is tested; the `imageio`/PyAV
  video path is written but unexercised by tests.

## Known gaps

- **Name.** The project is named `robovid_conditioner`, which is **available on
  PyPI** (checked) and free as a GitHub repo name. Reserve the PyPI name before
  first publish so it cannot be sniped. (The earlier placeholder `labelkit` was
  taken on PyPI; that is why this name was chosen.)
- **LeRobot write-back.** Writing annotations back into the dataset's own metadata
  is documented as planned but not implemented; only the parquet sidecar and JSONL
  export exist today. The pinned LeRobot version (0.4.x) and its metadata layout
  should be confirmed to support per-episode annotation fields before building it.
- **LeRobot version pinning.** Verified against 0.4.4 only. The adapter reads
  `meta.episodes[ep]["dataset_from_index"/"dataset_to_index"]`; a future LeRobot
  metadata change would require an adapter update. Pin and test a version matrix
  before 1.0.
- **Single camera.** Labels come from one camera (the first, or `--camera-key`).
  Multi-view reasoning is out of scope for now.
- **Cost estimates are provider-dependent.** Gemini cost uses a small hardcoded
  price table; OpenAI reports no dollar cost (token counts only, in receipts).
  Prices drift; treat the dollar number as indicative and audit via receipts.

## Fixed while building the strategy layer

Caught and fixed during the offline build + zero-cost preflight (each with a test
or a verification step), recorded here so the live run starts from a known-good base:

- **Mock prompt-marker collision.** The offline mock matched the refinement prompt
  on the manifest text "exact frame index", and matched the grounded-label prompt's
  embedded stage-one `events` block — both made S1–S4 fall back to one segment.
  Fixed with unique markers (`single integer frame index`; `end_frame`+`phase`
  before `events`); covered by the offline S1–S4 segmentation tests.
- **`ask()` signature change broke a test double.** Adding `frame_captions`/
  `temperature` broke `test_resilience`'s `_FlakyProvider`; updated to the new
  keyword-only contract.
- **Windows temp-file handle.** The scorer's `mkstemp` left an open fd, so
  `reliability_report` could not unlink it on Windows; close the fd first.
- **B023 late-binding closures** in the per-episode retry wrapper; bound via default
  args. **Py3.10 f-string** nesting in the selection rationale; precomputed.
- **Gemini Pro pricing** was absent from the cost table (Pro cost would read as
  `None`); added the 2.5-pro tier so the budget gate and `$/episode` are real.

The live-provider path is **dogfooded at single-run scale** (the earlier 50-episode
Gemini Flash annotate) but **not yet exercised at ablation scale** — that is exactly
what `scripts/run_ablation.py` does once the credential is present.

## First three issues to file

1. **Verify the OpenAI and Qwen providers end to end.** Gemini is dogfooded; do a
   one-episode live run for OpenAI and a local run for Qwen, capture a recorded
   request/response fixture for each, and replay it in CI so the parsing/cost
   paths are covered without keys or spend.
2. **Implement (or formally defer) LeRobot metadata write-back.** Decide whether
   0.4.x can carry per-episode annotation fields cleanly; either implement
   `--writeback` or document it as out of scope and remove the forward-reference
   from `SCHEMA.md`.
3. **Reserve the PyPI name and tag 0.1.0.** `robovid_conditioner` is free on PyPI;
   register it, then publish the first release with the pinned `lerobot` version
   range documented in the README.
