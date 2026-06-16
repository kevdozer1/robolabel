# robolabel

[![CI](https://github.com/kevdozer1/robolabel/actions/workflows/ci.yml/badge.svg)](https://github.com/kevdozer1/robolabel/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Automated, model-agnostic conditioning-annotation and curation for VLA finetuning on
[LeRobot](https://github.com/huggingface/lerobot) data. One config drives a modular pipeline
(`robolabel run --config run.yaml`) that drafts, per episode, the signals a VLA finetune wants:
subtask boundaries (`phase → target`), an episode quality score, optional speed and control
metadata, and subgoal keyframes, plus dataset-level curation. Any VLM can drive it, and every draft
is scored against a human gold set instead of assumed correct. Output is a parquet annotations file
plus an export in LeRobot's own subtask convention.

The annotation set mirrors the π0.7 data recipe (subtask language, episode quality, speed, and
subgoal images), with one change: subgoal keyframes are real frames retrieved from the dataset
rather than world-model generations, which keeps the pipeline lightweight.

![Grounded annotations on three tasks (pick-place, pour, fold): the current phase to target sub-step, a segment timeline with playhead, the episode quality, the real end-of-sub-step subgoal keyframes (selected, never generated), and the per-segment active components (which component groups actually move).](docs/figures/grounded_annotations.gif)

> Subgoal keyframes are real frames selected from the episode; robolabel does not generate images.
> The control line (`joint` or `end-effector`) is read from the action stream, not inferred.

## Install & quickstart

```bash
pip install -e '.[lerobot]'      # core needs no extra deps; lerobot for datasets
export GEMINI_API_KEY=...

# draft annotations (boundaries as frame indices, with per-segment evidence):
robolabel annotate --source lerobot --target lerobot/svla_so101_pickplace \
  --provider gemini --strategy S2 --limit 5 --out ann

robolabel gate        --annotations ann                    # automatic red flags (never drops)
robolabel reliability --gold so101_gold.json               # VLM-vs-human agreement
robolabel query       --annotations ann --phase grasp ...  # phase to contact sheet
robolabel export      --annotations ann --format lerobot --out ann_lerobot
robolabel cost        --annotations ann                    # token and USD accounting
```

`robolabel demo` runs the whole pipeline offline with no API key. The full config-driven pipeline is
documented in [`CONFIG.md`](CONFIG.md); non-LeRobot inputs in [`PORTING.md`](PORTING.md). To look at
results, `robolabel inspect` opens a per-episode viewer (gold and strategies on parallel boundary
tracks, plus an evidence-string-versus-frame check) and `robolabel gallery` shows several task
datasets in one view.

Each grounded segment is labeled `phase → target`: a fixed-vocabulary phase plus the specific object
it acts on, named from the scene, so two cubes don't both come back as a bare "approach".

```text
approach      → red cube    frames 0-41     "gripper descends toward the red cube"
grasp         → red cube    frames 42-70    "fingers close on the red cube"
transport     → blue cube   frames 71-119   "red cube lifted over the blue cube"
release-place → blue cube   frames 120-168  "red cube set on top of the blue cube"
retract                     frames 169-199  "arm withdraws, gripper empty"
```

See [`SCHEMA.md`](SCHEMA.md) for every output column.

## Demo

The three episodes in the figure above (pick-place, pour, fold) are bundled under [`demo/`](demo/)
as real ~200-frame clips plus their grounded annotations, so you can see the output with no API key
and no dataset download:

```bash
pip install -e '.[lerobot]'      # or: pip install -e . && pip install 'imageio[ffmpeg]'
python demo/demo.py
```

It prints each episode's grounded annotation (the `phase → target` sub-steps, the deterministic
per-segment active components, quality, motion-defined speed, and the selected subgoal frames) and
regenerates the annotated figure at `demo/grounded_annotations.gif`. Separately, `robolabel demo`
runs the whole pipeline on synthetic data with the mock provider.

## Providers & cost

Model-agnostic: a provider is one file (subclass `VLMProvider`, call `register_provider`). Built in:

| provider | example model | credential |
|---|---|---|
| `gemini` (default) | `gemini-2.5-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `qwen` (local, free) | `Qwen/Qwen2.5-VL-7B-Instruct` | none, needs a GPU |
| `mock` (offline, free) | none | none |

What it actually costs, measured from this project's receipts on Gemini 2.5 Flash (the default):

- About **$0.02 to $0.03 per episode** for the full stack. Only the two VLM modules (segmentation
  and quality) cost anything; speed, control, subgoals, novelty, curation, and the gripper baseline
  are deterministic, so a full run costs about the same as a minimal one.
- A full conditioning and curation pass over **1,000 episodes is roughly $20 to $28** on Flash.
  Annotation is one independent call per episode, so it batches cleanly: Gemini's asynchronous batch
  tier is about 50% cheaper, and caching the shared prompt prefix is about 90% cheaper on those
  tokens, which together bring 1,000 episodes toward **$6 to $10**.
- List prices used for the estimate (USD per million tokens, input / output): Flash $0.30 / $2.50,
  Flash-Lite $0.10 / $0.40, Pro $1.25 / $10.00. Pro's roughly 4x cost bought better quality judgment
  but not better boundary placement, so Flash is the default. OpenAI logs token counts so you can
  apply its current `gpt-4o` rate (about $2.50 / $10.00 per million at time of writing).

Every call writes a receipt with exact token counts; `robolabel cost` sums per-episode and total
USD, and re-running an interrupted batch reuses finished receipts for free.

## Curation

Each episode gets a value score `value = f(quality, novelty)`. With compression on, curation assigns
a fidelity tier (`full`, `reduced`, or `minimal`) so a loader can keep high-value episodes at full
fidelity and store low-value ones compressed; with a top-cut it marks `keep` or `cut`. It writes the
tier as an overlay and never drops or re-encodes data. Tiers are corpus-relative and are left empty
when the population is too small or too uniform to rank honestly.

## Accuracy

Drafts are scored against a human gold set. On `lerobot/svla_so101_pickplace` against a 50-episode
gold (one task family, one annotator's gold, built by correcting the baseline):

- The grounded strategy removes the degenerate "one blob / uniform fifths" segmentations: 5 of 20
  held-out episodes out of the box, **0 of 20** grounded (and 0 of 20 on a fresh stacking set).
- It places **36% more** gold boundaries within ±5 frames (recall 0.307 versus 0.226).
- Mean segment-overlap IoU is unchanged (0.444 versus 0.460).

Whether the annotations improve downstream training is untested (one preregistered test was
negative), as is generalization beyond this task family.

## Export

`robolabel export --format lerobot` writes LeRobot's subtask convention, round-trip-tested through
lerobot's own `load_subtasks` so every frame's `subtask_index` resolves to its segment. It composes
with the manual [LeRobot Annotate](https://github.com/huggingface/lerobot-annotate) GUI: draft with
robolabel, correct the flagged cases by hand, then train. A JSONL export is also available.

## Status

Beta, single-author. Schemas are versioned but may change before 1.0. Linux and macOS (Windows is
not a target). License: [Apache-2.0](LICENSE).
