# robolabel

[![CI](https://github.com/kevdozer1/robolabel/actions/workflows/ci.yml/badge.svg)](https://github.com/kevdozer1/robolabel/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Automated, model-agnostic **conditioning-annotation and curation for VLA finetuning on
[LeRobot](https://github.com/huggingface/lerobot) data**. One config drives a modular pipeline
(`robolabel run --config run.yaml`) that, per episode, drafts the signals a VLA finetune wants —
subtask boundaries (`phase → target`), an episode quality score, optional speed/control metadata,
and subgoal keyframes — plus dataset-level curation (novelty + a value score). Any VLM can drive
it, and every draft is **measured against a human gold set** rather than assumed correct. Output is
a parquet sidecar plus an export in LeRobot's own subtask convention.

![What robolabel adds to one raw LeRobot episode: subtask boundaries, a quality judgment, and subgoal keyframes.](docs/figures/annotation_overview.png)

![Grounded annotations on three tasks (pick-place, pour, fold): the current phase → target sub-step, a segment timeline with playhead, the episode quality, the real end-of-sub-step subgoal keyframes (selected, never generated), and the deterministic control line.](docs/figures/grounded_annotations.gif)

> Subgoal keyframes are **real frames selected** from the episode — robolabel does **not** generate
> images. The control line (`joint` / `end-effector`) is read from the action stream, not inferred.

## Install & quickstart

```bash
pip install -e '.[lerobot]'      # core needs no extra deps; lerobot for datasets
export GEMINI_API_KEY=...

# draft annotations (boundaries as frame indices + per-segment evidence):
robolabel annotate --source lerobot --target lerobot/svla_so101_pickplace \
  --provider gemini --strategy S2 --limit 5 --out ann

robolabel gate        --annotations ann                    # automatic red flags (never drops)
robolabel reliability --gold so101_gold.json               # VLM-vs-human agreement
robolabel query       --annotations ann --phase grasp ...  # phase -> contact sheet
robolabel export      --annotations ann --format lerobot --out ann_lerobot
robolabel cost        --annotations ann                    # token + USD accounting
```

`robolabel demo` runs the whole pipeline offline with no API key. For the full config-driven
pipeline see [`CONFIG.md`](CONFIG.md); for non-LeRobot inputs see [`PORTING.md`](PORTING.md). To
eyeball results, `robolabel inspect` opens a per-episode verification viewer (gold + strategies on
parallel boundary tracks, with an evidence-string-vs-frame check) and `robolabel gallery` shows
several task datasets in one task-grouped view.

Each grounded segment is labeled **`phase → target`** — a fixed-vocabulary phase plus the specific
object it acts on, named from the scene, so two cubes don't both come back as a bare "approach":

```text
approach      → red cube    frames 0–41     "gripper descends toward the red cube"
grasp         → red cube    frames 42–70    "fingers close on the red cube"
transport     → blue cube   frames 71–119   "red cube lifted over the blue cube"
release-place → blue cube   frames 120–168  "red cube set on top of the blue cube"
retract                     frames 169–199  "arm withdraws, gripper empty"
```

`robolabel query --phase grasp` turns those labels into a contact sheet — e.g. every grasp in the
dataset, one tile per episode. See [`SCHEMA.md`](SCHEMA.md) for every output column.

![Every segment robolabel labeled "grasp", one tile per episode.](docs/figures/grasp_montage.png)

## Providers, cost & batching

robolabel is model-agnostic — a provider is one file (subclass `VLMProvider`, call
`register_provider`). Built in:

| provider | example model | credential | cost |
|---|---|---|---|
| `gemini` | `gemini-2.5-flash` (default) | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | estimated from token counts |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | tokens recorded; no USD asserted |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` (local) | — (GPU) | free |
| `mock` | — | — | free (offline; powers `demo`) |

- **Per episode, not per frame.** A labeler sends a single contact sheet of sampled keyframes per
  call, so cost scales with the number of episodes, not the frame count. Override the model with
  `--model` or `$ROBOVID_MODEL`.
- **Free resume / caching.** Every call writes a raw receipt (`raw_receipts/`) with exact token
  counts; re-running a partial or interrupted batch **reuses the successful receipts for free** and
  only pays for the episodes still missing.
- **Accounting.** `robolabel cost` sums per-episode and total USD from the provider pricing table
  (Gemini Flash ≈ $0.30 / $2.50 per Mtok in/out; Flash-Lite ≈ $0.10 / $0.40; Pro ≈ $1.25 / $10.00).
  Raw token counts are always in the receipts for an exact audit.
- The deterministic modules (speed, control, novelty, curation, the gripper baseline) and the entire
  offline `demo` path cost **$0**.

## Measured, not assumed

On `lerobot/svla_so101_pickplace` against a 50-episode human gold set — one caveat applies to every
number: one task family, one annotator's gold, built by correcting the baseline's drafts.

- **Failure tail eliminated.** Out of the box, 25% (5/20) held-out episodes come back as a single
  "do the task" blob or boundaries at uniform fifths; the grounded strategy brings that to **0/20**
  (and 0/20 on a fresh, never-touched stacking set). The most robust result.
- **Better boundary placement.** Grounded lands **36% more** gold boundaries within ±5 frames
  (recall 0.307 vs 0.226), even though mean segment-overlap IoU doesn't improve (0.444 vs 0.460).
- **Quality scores** discriminate only on variable-quality corpora; on a uniform dataset the
  deterministic, motion-defined **speed** descriptor is the more informative episode metadata.

Not shown: that these annotations improve downstream training (one preregistered test was negative),
or that they generalize beyond this task family. Both are stated as untested, never implied.

## Composes with LeRobot Annotate

robolabel is the *automated, at-scale* front of the same workflow as
[LeRobot Annotate](https://github.com/huggingface/lerobot-annotate) (the manual GUI): draft with
robolabel → review and correct the flagged cases by hand → train, all in the subtask convention the
trainer already reads (`export --format lerobot`, round-trip-tested through lerobot's `load_subtasks`).

## Status

Beta, single-author. Schemas are versioned but may change before 1.0. Linux and macOS (Windows is
not a target). License: [Apache-2.0](LICENSE).
