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

## Configuration

`robolabel run --config run.yaml` drives the whole pipeline from one file: a `run` block (dataset,
model, probe size) and a `modules` block where every module is an independent toggle. For a standard
LeRobot dataset you provide nothing but `source`/`target`; camera, fps, control space, and
arm/gripper dims are auto-detected. The minimal default runs just segmentation + quality.

```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace }   # camera_key: auto
  model:   { provider: gemini, name: gemini-2.5-flash }
  probe:   { max_episodes: 5 }
  out: run_out/full
modules:
  segmentation: { enabled: true, strategy: grounded, vocabulary: open }  # closed = S2; baseline = S0
  quality:      { enabled: true }
  speed:        { enabled: true }
  subgoals:     { enabled: true, retrieval: true }
  control:      { enabled: true }
  novelty:      { enabled: true }
  curation:     { enabled: true, compress: true }
```

The module set is the implemented surface, all additive in the parquet output:

| module | default | produces |
|---|---|---|
| `segmentation` | on | grounded `phase → target` subtask boundaries (open-vocab; `closed`/`baseline` available) |
| `quality` | on | episode quality 1-5 (VLM) |
| `speed` | off | motion-defined active duration (`active_frames`/`_seconds`/`_fraction`) plus a corpus-relative fast/medium/slow tier |
| `subgoals` | off | the real end-of-sub-step keyframe (a pointer); optionally a same-phase keyframe retrieved from another episode |
| `control` | off | `control_modality` (joint vs end-effector frame) plus the per-segment `active_dof` set (which component groups move) |
| `novelty` | off | per-episode novelty (distance in a cheap frame embedding) |
| `curation` | off | `curation_value = f(quality, novelty)` plus optional corpus-relative fidelity tiers (an overlay, never deletes) |

Only `segmentation` and `quality` call the VLM; the rest are deterministic and cost $0. Modules run
in dependency order, dataset-level ones (novelty, curation, retrieval) after the per-episode pass.
Full reference: [`CONFIG.md`](CONFIG.md).

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
| `openrouter` (experimental) | any OpenRouter model id | `OPENROUTER_API_KEY` |
| `mock` (offline, free) | none | none |

Built by name (`--provider openrouter`, or `provider: openrouter` in a run config), the OpenRouter
provider runs without a spend guard or response cache; those are set up in Python, see
[the V-lite section](#experimental-v-lite-pipeline).

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

## Experimental: V-lite pipeline

V-lite is an experimental, opt-in redesign that sits next to the pipeline above and changes nothing in
it: `robolabel run`, `annotate` and `demo` still write schema v6 as before. It has no CLI subcommand
yet; you call it from Python, and its API may change. It labels one episode at a time in five layers:

| layer | what it does | model calls |
|---|---|---|
| L1 signal (`robolabel.layers.signal`) | finds gripper closing and opening events in `observation.state` and `action`, groups them into grasp attempts with an outcome (hold, empty, slip, released), reads the robot's end state, and proposes candidate boundaries | none |
| L2 scene (`robolabel.layers.scene`) | an object inventory (IDs, names, a point and a box per camera) and scene facts per keyframe: what is visible, what is in the gripper, a few relations between objects | 2 |
| L3 segments (`robolabel.layers.segment`) | phase segments with target, destination, attempt and outcome; a segment end moves onto an L1 candidate the model confirms; coarse subtasks from fixed templates | 1 |
| L4 goal (`robolabel.layers.goal`) | the goal as end-state requirements (such as "the brick is inside the box" or "the gripper is open"), each `required`, `incidental` or `unsure`, and the episode outcome over the required ones | 1 |
| L5 checks (`robolabel.layers.check`) | ten rule checks (for example: a grasp segment starts near a gripper closing onset), an episode risk and a route-for-review flag | none |

`robolabel.vlite.run_episode` runs L2 to L5 with one model and returns schema v7 rows (the new record
types are in [`SCHEMA.md`](SCHEMA.md)) and a view record: one plain-JSON dict per episode with the
objects, segments, coarse subtasks, attempts, goal, checks, risk, cost and the status of each call. A
call that is still invalid after one repair retry does not end the episode; the gap is listed in
`repairs`. L1 knows two gripper layouts, `so101` and `libero`. `robolabel.baselines` builds view
records for comparison arms, such as `sig_only_view` (L1 events only, no model).

V-lite uses three new optional extras; the pipeline above needs none of them:

| extra | installs | used for |
|---|---|---|
| `eval` | `scipy`, `jsonschema` | the measurement harness in `robolabel.eval` (segment matching, gold v2 validation). The OpenRouter provider also checks every answer against its JSON Schema with `jsonschema`, and skips that check when it is not installed |
| `video` | `av` (PyAV) | `robolabel.adapters.lerobot_v3`, which reads a LeRobot v3.0 folder directly (every camera plus `observation.state` and `action`) without a `lerobot` install |
| `hub` | `huggingface_hub` | fetching a dataset at a pinned revision, for example with `huggingface_hub.snapshot_download(repo_id, repo_type="dataset", revision=..., local_dir=...)`; robolabel does not import it |

```bash
pip install -e '.[eval,video,hub]'
```

### Example (offline)

This runs with no key, no network and no dataset: a synthetic episode, L1, and the mock provider,
whose answers are valid against the schemas but describe nothing.

```python
import numpy as np

from robolabel.episode import Episode
from robolabel.layers.signal import Calibration, run_l1
from robolabel.providers.mock import MockProvider
from robolabel.schema_v7 import write_v7
from robolabel.vlite import run_episode

# A synthetic SO-101 episode: 120 frames, an external and a wrist camera, and one grasp that closes on
# an object and later releases it (commanded to 1, the gripper stops at 8: something is in the way).
n = 120
rng = np.random.default_rng(0)
cams = ["observation.images.up", "observation.images.wrist"]
video = {c: rng.integers(0, 255, size=(n, 60, 80, 3), dtype=np.uint8) for c in cams}
getters = {c: (lambda i, c=c: video[c][int(i)]) for c in cams}
state, action = np.zeros((n, 6)), np.zeros((n, 6))
action[:, 5] = np.r_[[20.0] * 30, np.linspace(20, 1, 10), [1.0] * 40, np.linspace(1, 20, 10), [20.0] * 30]
state[:, 5] = np.r_[[20.0] * 30, np.linspace(20, 8, 10), [8.0] * 40, np.linspace(8, 20, 10), [20.0] * 30]
state[:, 0] = np.linspace(0, 90, n)  # one arm joint moves throughout
ep = Episode(episode_id="demo/0", num_frames=n, fps=30.0, task="put the brick in the box",
             get_frame=getters[cams[0]], camera_key=cams[0],
             extra={"family": "demo", "cameras": getters, "camera_order": cams,
                    "external_cameras": cams[:1], "wrist_cameras": cams[1:]})

# L1: gripper events, grasp attempts and candidate boundaries from state and action (no model call).
cal = Calibration(layout="so101", fps=30.0, cmd_open=20.0, cmd_closed=1.0, meas_open=20.0,
                  meas_closed=1.0, pause_speed=5.0, withdraw_threshold=5.0)
l1 = run_l1(state, action, cal, episode_key=ep.episode_id, family="demo")

# L2 to L4: four model calls (inventory, scene facts, segments, goal); then the L5 rule checks.
provider = MockProvider()  # offline, $0, placeholder answers that describe nothing
out = run_episode(ep, l1, provider, arm="v@mock", model_key="mock")

view = out["view"]
print([(s["start"], s["end"], s["phase_class"], s["outcome"]) for s in view["segments"]])
print(view["episode_outcome"], view["risk"], view["routed"], view["calls"], view["valid"])
print(write_v7(out["rows"], "vlite_out"))  # vlite_out/annotations.parquet, schema v7
```

On a real dataset, a local LeRobot v3.0 folder (the `video` extra) gives the episode and the
calibration:

```python
from robolabel.adapters.lerobot_v3 import LeRobotV3Folder, LeRobotV3Source
from robolabel.layers.signal import calibrate, run_l1

folder = LeRobotV3Folder("path/to/dataset")  # a downloaded LeRobot v3.0 dataset
cal = calibrate(folder.stats, "so101", folder.fps)  # optionally pass (state, action) pairs of a few episodes
ep = LeRobotV3Source(folder, [0], family="mydata").episode(0)
l1 = run_l1(ep.extra["state"], ep.extra["action"], cal, episode_key=ep.episode_id, family="mydata")
```

### A real model: OpenRouter, the spend guard and the cache

`OpenRouterProvider` (`robolabel.providers.openrouter`) calls OpenRouter's OpenAI-compatible API. It
reads the key from the `OPENROUTER_API_KEY` environment variable (or a line `OPENROUTER_API_KEY=...`
in a local `.env` file). To use it, replace the two lines `provider = MockProvider()` and
`out = run_episode(...)` in the example with:

```python
from robolabel.eval.receipts import JsonlWriter, ResponseCache
from robolabel.providers.openrouter import OpenRouterProvider
from robolabel.spend_guard import GuardConfig, SpendGuard

MODEL = "vendor/model-name"  # an OpenRouter model id
guard = SpendGuard(GuardConfig(run_cap=5.00, available_at_start=20.00, balance_floor=2.00,
                               bucket_caps={"sweep": 5.00}), "runs/spend_ledger.jsonl")
provider = OpenRouterProvider(MODEL, guard=guard,
                              prices={MODEL: (0.30, 2.50)},  # its USD per million tokens (input, output)
                              cache=ResponseCache("runs/responses.jsonl"),
                              receipts=JsonlWriter("runs/receipts.jsonl"))
out = run_episode(ep, l1, provider, arm="v@my-model", model_key="my-model", reasoning={"effort": "low"})
guard.close()
```

The spend guard (`robolabel.spend_guard`):

- Before every HTTP attempt, retries included, the provider reserves that attempt's worst case: the
  estimated input tokens at the input price plus `max_tokens` at the output price. So `prices` must
  hold the model's current list prices. The provider also sends 1.5 times them to OpenRouter as the
  highest price it will pay (`max_price`).
- The guard refuses a reservation, and nothing is sent, when it would take the committed total (spent,
  unreconciled and still reserved) past the run cap (`run_cap`), the balance floor
  (`available_at_start`, the credit left on the key when the run starts, minus `balance_floor`, so the
  key keeps at least `balance_floor`), the cap of the bucket the call is charged to (`bucket_caps`; `run_episode` charges the bucket `sweep` unless you
  pass `bucket=`), or a per-model cap (`model_caps`, sweep bucket only). A refused call has status
  `refused`, and the episode goes on without it.
- After a response, the call's reported cost (`usage.cost`) replaces the reservation. A timeout, a
  dropped connection, a 5xx or a 200 without a cost keeps its reservation as spent; a request that
  OpenRouter rejects before any generation (an HTTP 400 or 403) is recorded at $0.
- Every event is appended to a JSONL ledger, flushed and fsynced. A restarted guard replays it, and
  one process at a time may reserve against a ledger (a `.lock` file next to it). An optional drift
  check (`key_usage=` plus `record_key_usage_start`) compares the change in the key's own usage with
  the ledger and stops paid calls when the key's usage stays above it.

`ResponseCache` keeps answers in one append-only JSONL file, keyed by a SHA-256 of the provider,
model, prompt, generation settings and image hashes, so re-running the same request is answered from
the cache without an HTTP request. `JsonlWriter` appends one receipt per call (tokens, cost, status,
structured-output mode). Receipts and the cache never hold the key, request headers or image bytes.

### Results so far

One development sweep so far: 4 dev episodes, with quick ratings by one rater. The results are
descriptive only: they rank no models and support no quality claim. Two things the sweep showed about
the pipeline itself:

- Segment ends move onto the L1 candidates the model confirms, so in the sweep the boundary positions
  came mostly from the signal layer, not from the model.
- L1 proposes candidates only at each closing onset, where the arm starts moving after a close that
  held, at the opening onset of a release, and where the arm starts moving after it. After a missed
  grasp it proposes none for the re-opening, the backing off or the re-approach, and in the sweep most
  segmentation errors fell in those stretches.

### PyAV and codecs

robolabel uses PyAV (the `video` extra) only to decode video, in `robolabel.adapters.lerobot_v3`; it
encodes no video and does not bundle or redistribute PyAV or FFmpeg. PyAV itself is BSD-3-Clause,
but its PyPI wheels bundle an FFmpeg build that includes GPL codecs (libx264, libx265). For an
LGPL-only setup, install an FFmpeg with no GPL components (built without `--enable-gpl` and without
libx264, libx265 or other GPL libraries, with its development headers), then build PyAV from source
against it:

```bash
pip install av --no-binary av
```

LeRobot v3.0 datasets often store AV1 video, so that FFmpeg needs an AV1 decoder such as libdav1d
(BSD-licensed).

## Status

Beta, single-author. Schemas are versioned but may change before 1.0. Linux and macOS (Windows is
not a target). License: [Apache-2.0](LICENSE).
