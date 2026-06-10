# Release readiness

Status: **beta, pre-publication.** Honest assessment of what works, what is only
smoke-tested, what is missing, and what to fix first.

## What works (tested)

- **Core pipeline, offline.** `robovid_conditioner demo` runs end to end in ~1s with no API
  key (synthetic episodes → mock provider → valid `annotations.parquet` → gate).
  Covered by `tests/test_demo.py` and the CI integration step.
- **Provider layer.** Two-stage observe→label flow, raw-response receipts,
  per-call cost, contact sheet, and a credential loader that names the exact
  missing env var. Registry + one-file-per-provider. Mock fully tested; the
  Gemini/OpenAI request shapes are written but exercised only against the live
  APIs (see below).
- **Rubric as config.** All prompts, the 1–5 quality scale, and gate thresholds
  load from `rubric.yaml`; custom rubrics via `--rubric`. Tested.
- **Schema.** Deterministic, versioned `annotations.parquet` (episode metadata /
  subtask / subgoal). Round-trip and determinism tested.
- **Adapters.** `DirectoryAdapter` (frame dirs) tested offline. `LeRobotAdapter`
  verified end to end against the real Apache-2.0 dataset
  `lerobot/svla_so101_pickplace` (50 eps, 30 fps, 480×640 frames) → mock annotate
  → valid parquet, using the installed **lerobot 0.4.4** API.
- **Calibration.** Gold sets keep human labels in a separate block from VLM labels
  and never overwrite either side (tested). Reliability metrics (subtask boundary
  temporal IoU, quality exact / within-one, subgoal frame agreement) tested.
- **Gate.** Collapsed-score, repeated-text, object-grounding, and score↔reason
  contradiction flags, thresholds from the rubric. Tested.
- **Quality bar.** `ruff` clean; 29 tests pass on Python 3.10; CI matrix 3.10/3.12.

## Smoke-tested / unverified only

- **Live Gemini / OpenAI calls.** The request/response handling is written against
  each API's documented shape but is **not** run in CI (no keys, real cost). The
  real-provider path needs a manual run before any "it works with Gemini" claim.
- **Qwen local provider.** Imports and registers without the extra; the actual
  generate path is unrun here (needs GPU + the `qwen` extra).
- **Streamlit review GUI.** Pure helpers are tested; the interactive app is not
  launched in CI. It has been written against the same gold data layer the tests
  cover, but treat the UI as unproven until a human runs `robovid_conditioner review`.
- **mp4 DirectoryAdapter path.** Frame-directory mode is tested; the `imageio`/PyAV
  video path is written but unexercised by tests.

## Known gaps

- **Name collision (blocking for PyPI).** The working name `robovid_conditioner` is **taken
  on PyPI** by an unrelated package. Do not publish under it. Available, checked
  alternatives: **`robolabel`** (PyPI free, no `robolabel/robolabel` GitHub org)
  and **`vlalabel`** (PyPI free); `calibrated-labels` is also free. Avoid
  `lerobot-annotate` (HuggingFace already owns that GitHub repo). Recommendation:
  rename the distribution to `robolabel`, keep a short import alias.
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

## First three issues to file

1. **Rename the distribution off `robovid_conditioner` (PyPI collision).** Pick `robolabel`,
   update `pyproject.toml`/`[project.scripts]`/docs, add an import alias, and
   reserve the PyPI name. Blocking for any `pip install` story.
2. **Add a recorded-fixture provider test for Gemini/OpenAI.** Capture one real
   request/response per provider, store as a fixture, and replay it in CI so the
   parsing/cost paths are covered without keys or spend.
3. **Implement (or formally defer) LeRobot metadata write-back.** Decide whether
   0.4.x can carry per-episode annotation fields cleanly; either implement
   `--writeback` or document it as out of scope and remove the forward-reference
   from `SCHEMA.md`.
