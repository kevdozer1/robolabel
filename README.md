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
| `video` | `av` (PyAV) | `robolabel.adapters.lerobot_v3`, which reads a LeRobot v3.0 folder directly (every camera plus `observation.state` and `action`) without a `lerobot` install, and `robolabel.adapters.clip_folder` (v1.1), which reads a folder of short clips |
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
- L1 proposed candidates only at each closing onset, where the arm starts moving after a close that
  held, at the opening onset of a release, and where the arm starts moving after it. After a missed
  grasp it proposed none for the re-opening, the backing off or the re-approach, and in the sweep most
  segmentation errors fell in those stretches. v1.1 adds the re-open and the back-off as recovery
  candidates, in a field of their own (see [Failure convention](#failure-convention)).

### PyAV and codecs

robolabel uses PyAV (the `video` extra) only to decode video, in `robolabel.adapters.lerobot_v3` and
`robolabel.adapters.clip_folder`; it encodes no video and does not bundle or redistribute PyAV or
FFmpeg. PyAV itself is BSD-3-Clause,
but its PyPI wheels bundle an FFmpeg build that includes GPL codecs (libx264, libx265). For an
LGPL-only setup, install an FFmpeg with no GPL components (built without `--enable-gpl` and without
libx264, libx265 or other GPL libraries, with its development headers), then build PyAV from source
against it:

```bash
pip install av --no-binary av
```

LeRobot v3.0 datasets often store AV1 video, so that FFmpeg needs an AV1 decoder such as libdav1d
(BSD-licensed).

## Experimental: video first (v1.1)

v1.1 is a second opt-in redesign next to V-lite, built on one principle: **the video alone must be
good; signals make it better.** Every step works from the video of one camera, so it also runs on
robot data without gripper channels, on humanoids and people, and on tasks that are not pick and
place. A robot signal is an optional *event source*: when there is one it is used for timing,
attempts and robot end states, and when there is none nothing breaks. "v1.1" names this pipeline
design (its rows carry `pipeline_version` `v1.1 ...`), not a package release.

Nothing above changes: `robolabel run`, `annotate` and `demo` still write schema v6, V-lite's
`run_episode` still writes the same v7 rows (the written v7 file is identical; the `pipeline_code` in
its view and row dicts changes, since it hashes `layers/*.py` and `schema_v7.py`, which v1.1 extends),
and v1 to v7 files still read. Like V-lite, v1.1 has no CLI subcommand; you call it from Python and
its API may change (options in [`CONFIG.md`](CONFIG.md#v11-video-first-experimental)). Its prompts are
version v8; the response cache is keyed by the prompt text, so a v8 step is never answered from a
cached V-lite (v7) call.

`robolabel.vfirst.run_episode_v11` labels one episode from one camera, in this order:

| step | what it does | model calls |
|---|---|---|
| event source | typed candidate events from `none`, `motion` or `gripper` (below) | none |
| L2 inventory | object IDs and names from up to 8 evenly spaced frames, first and last included | 1 |
| coarse pass | every segment of the clip, from frames or from native video | 1 |
| crawl | each typed boundary refined to its onset frame | at most 3 per boundary, 12 boundaries |
| L2 facts | scene facts at frame 0, each boundary and the last frame (at most 8; with the gripper source, L1's keyframe plan over every event); skipped when the inventory is empty | 1 |
| L4 goal | the goal as end-state requirements, plus `has_end_state` and `goal_command` | 1 |
| L5 checks | the ten V-lite rules and three video-only rules | none |

It returns a view record and v7 rows with the v1.1 fields, which `robolabel.schema_v7.write_v11` writes
(see [`SCHEMA.md`](SCHEMA.md#v11-additions-video-first-experimental)). A call that fails, is refused
or is still invalid after its repair retry does not end the episode: the gap is listed in `repairs`,
and the episode gets risk 1.0 and is routed for review with the reason. `robolabel.vfirst.run_timing`
runs only the event source, the coarse pass and the crawl, for timing experiments.

### Event sources

A source (`robolabel.events.get_source(name)`) returns events
`{type, frame, confidence, source, attempt_idx}` in time order. The coarse pass sees them as plain
lines such as `c1: frame 152 (5.07 s), pause_start (motion)`, which the prompt calls hints, not
boundaries to copy.

| source | reads | events |
|---|---|---|
| `none` (default) | nothing | none: the video alone |
| `motion` | the camera's pixels | `pause_start` and `pause_end` around runs of at least 0.3 s whose frame-to-frame difference (grayscale, long side 128 px, 3-frame smoothing) is at or below the clip's 20th percentile. Free and deterministic |
| `gripper` | the L1 record (`l1=`, from `observation.state` and `action`) | L1's `close_start` and `open_start` onsets, `arm_move` (low confidence), and after a failed close the re-open and the back-off (source `gripper_recovery`) |

Choosing one:

- `gripper` when the robot records its gripper in a layout L1 knows (`so101`, `libero`). A
  `close_start` or `open_start` boundary that the coarse pass ties to a gripper event takes the L1
  frame (`boundary_source: signal`) and is not crawled, a robot end-state item the model left
  `required` with `achieved` unknown, or marked `unsure` / `perception`, takes L1's answer when L1 can
  give one (`basis: signal`), and the five L5 rules that need a signal run.
- `none` for any other video, and whenever the gripper signal is the truth you score against: that
  signal must stay hidden, so `run_episode_v11` raises ValueError when `l1=` is passed with any other
  source (`run_timing` ignores it there).
- `motion` adds free pause hints to any video; how much they help is not established yet.

### Coarse pass

One call (`robolabel.layers.coarse`) proposes all the segments of the clip. It sees frames at 2 per
second, first and last included, at most 48 (a clip longer than about 24 s gets 48 evenly spaced
frames), long side at most 448 px, each after a caption such as
`frame 160 of 303 (5.33 s), camera up`; or the clip as native video (below). Per segment it returns:

- `phase_text`, the phase in open text (always), and `phase_class`, a class from the V-lite list or
  `other`: the prompt says the list is a vocabulary, not a template;
- `end_event`, the type of the boundary at the segment's end: `close_start`, `open_start`,
  `contact_start`, `contact_end` or `other` (always `other` on the last segment). A grasp boundary is
  `close_start` and a let-go `open_start`; the contact types are for touches in which the fingers
  neither close nor open (a pressed control, a held tool on a surface, a push or a wipe);
- the target and destination (inventory IDs, or plain words when there is no inventory), the attempt
  index, the outcome and attempt outcome (below), and the candidate the boundary sits on, if any.

Post-processing is deterministic and records every repair: the segments are made contiguous over the
clip, times become frames, gripper-tied boundaries take their L1 frame, and the failure convention is
applied.

### The crawl

The crawl (`robolabel.layers.crawl`) refines every boundary typed `close_start`, `open_start`,
`contact_start` or `contact_end`, by rule: never only because the model said it was unsure. Each call
shows up to 8 images of the camera and asks one narrow question, such as "Which of these frames is
the first where the fingers (gripper or hand) have started to close?" (for a contact: "... where the
hand or the tool touches the <object>"):

1. **Stage 1**: 8 frames evenly spaced over ±1.0 s around the proposed onset.
2. **Stage 2**: up to 8 frames from the stage-1 frame before the pick to the pick, at native spacing
   when they fit, else at the finest spacing that does.

The answer is one integer: 2 to 8 for the first image that shows the event, 0 when image 1 already
does, 9 when it has not begun by image 8, -1 when it cannot be judged. A 0 or a 9 in stage 1 shifts the
window by its own width and asks once more. The result is the onset frame: the next segment starts
there and the one before ends one frame earlier (`boundary_source: crawl`; `coarse_end_frame` keeps the
proposal). A boundary the crawl cannot decide keeps its coarse frame. Caps: at most 3 calls per
boundary and 12 crawled boundaries per episode, in time order. The view's crawl log records, per
boundary, the frames shown, each answer, the cost and flags such as `crawl_edge`, `crawl_none`,
`crawl_inconsistent`, `crawl_cross` or `skipped_cap` (each flag is explained in
[`SCHEMA.md`](SCHEMA.md#crawl-log-flags-v11)). The crawl uses the coarse model unless you pass
`crawl_caller=`; `crawl=False` turns it off.

### Native video input

With `coarse_mode="video"` and `video=` a `VideoPart` (mp4 bytes), the coarse pass sends the whole clip
as one video instead of frames. The model then gives times in seconds, in steps of 0.1 s, which become
frames as `round(t * fps)`. Only the coarse pass uses the video; the inventory, the crawl, the facts and
the goal still see frames.

- robolabel never encodes video. `ClipFolderSource.video_part(clip_id)` returns the clip file's own
  bytes when it is an H.264 mp4 of at most 60 s, and None otherwise; then encode the clip yourself and
  build `VideoPart(data=mp4_bytes, mime="video/mp4", seconds=duration_s)`. Set `seconds` to the clip's
  duration: the spend guard's worst-case reservation counts one image's tokens per second of video
  (at least one), and `seconds` defaults to 0.
- The OpenRouter provider sends it as a `video_url` part, so the model must accept video input on
  OpenRouter (see the model's input modalities). The model's provider samples the video itself, at a
  rate robolabel neither sets nor records; keep that in mind when comparing video with frames.

Like V-lite, v1.1 calls models through `.call(CallRequest)`, which the `openrouter` and `mock`
providers implement; the `gemini`, `openai` and `qwen` providers do not yet.

### Failure convention

- A phase's `outcome` is its own result. In a missed grasp the approach that reached the object is
  `success` and the grasp is `failed` with `failure_type: missed_grasp`; `mistake` (pi0.7's
  per-segment flag) is true only on that grasp.
- `attempt_outcome` (`success`, `failed` or `aborted`) is the result of the whole attempt, copied onto
  each of its phases, so a consumer can still drop whole failed attempts. The attempt record keeps the
  whole span, from the attempt's first phase to its last, and its `evident_frame`.
- Recovery after a failed grasp: the grasp ends when the fingers start to reopen (`open_start`), then
  comes `retract` if the arm backs off, then the `approach` of the next attempt (`attempt_idx` one
  higher). L1 now marks the re-open (high confidence) and the back-off, where the arm starts moving
  away (low confidence), as candidates in a field of their own, `recovery_candidates`. The gripper
  source passes them to the coarse pass; V-lite does not use them.
- Older outputs have no `attempt_outcome`. Readers derive it by the old rule
  (`robolabel.eval.derive_attempt_outcome`), and the scorer picks the rule per view, so older views
  score as before.

### Goals: states, plus a command

- `goal_objective` is kept a state ("the pink brick is inside the transparent box"). An objective that
  the state check reads as a command (it starts with a known imperative verb, a fixed list in
  `robolabel.layers.goal.is_state_objective`) is replaced by a state sentence rendered from the
  requirements.
- `goal_command` is rendered deterministically from the required object end states, in their order,
  joined with "then": `inside` gives "Put <object> in <ref>", `on_top_of` "Put <object> on <ref>",
  `activated` "Press <object>" for a control or "Turn on <object>" otherwise, and `state` "Set <object>
  to <value>". Robot items never appear, and it is empty when nothing renders. This is the text for
  training prompts: the pink brick required inside the transparent box, plus an open gripper, gives
  "Put the pink brick in the transparent box". `robolabel.layers.goal.goal_command` renders a few
  more verbs (such as "Take ... out of" or "Pick up") only with `extra_forms=True`, which the pipeline
  does not pass.
- `has_end_state` (from the goal call) is false for an activity with no object end state, such as a
  dance, a wave or a gesture; then no `object_end_state` item may appear (L5 rule 13).

### Example (offline, v1.1)

This runs with no key, no network and no dataset: a synthetic clip with no robot signal, the `motion`
source and the mock provider, whose answers are valid against the schemas but describe nothing (it
gives one segment, so nothing is crawled).

```python
import numpy as np

from robolabel.episode import Episode
from robolabel.providers.mock import MockProvider
from robolabel.schema_v7 import write_v11
from robolabel.vfirst import run_episode_v11

# A synthetic clip: 90 frames at 30 fps from one camera, and no robot signal.
n = 90
rng = np.random.default_rng(0)
video = rng.integers(0, 255, size=(n, 60, 80, 3), dtype=np.uint8)
ep = Episode(episode_id="C/demo", num_frames=n, fps=30.0, task="put the brick in the box",
             get_frame=lambda i: video[int(i)], camera_key="cam")

# Inventory, coarse pass (frames at 2 per second), crawl, scene facts and goal; then the checks.
provider = MockProvider()  # offline, $0, placeholder answers that describe nothing
out = run_episode_v11(ep, camera="cam", caller=provider, event_source="motion",
                      context={"arm": "v11@mock", "episode_key": ep.episode_id})

view = out["view"]
print([(s["start"], s["end"], s["phase_class"], s["end_event"], s["boundary_source"]) for s in view["segments"]])
print(view["event_sources"], view["coarse_mode"], view["coarse_fps"], view["crawl_calls"], view["step_status"])
print(view["has_end_state"], repr(view["goal_command"]), view["risk"], view["routed"])
print(write_v11(out["rows"], "v11_out"))  # v11_out/annotations.parquet: the v7 layout plus the v1.1 columns
```

With a real model, build an `OpenRouterProvider` as in
[the V-lite section](#a-real-model-openrouter-the-spend-guard-and-the-cache) and pass it as `caller=`
(the spend-guard bucket is `context["bucket"]`, `sweep` by default). With a robot signal, pass
`event_source="gripper", l1=l1`, with `l1` from `run_l1` as in the V-lite example. For native video,
a folder of short clips (the `video` extra) gives the episode and the video part:

```python
from robolabel.adapters.clip_folder import ClipFolderSource

clips = ClipFolderSource("path/to/clips")  # <root>/<clip id>/clip.mp4, and optionally task.txt
ep = clips.episode("my_clip")              # key C/my_clip, one camera named "video"
out = run_episode_v11(ep, camera="video", caller=provider, coarse_mode="video",
                      video=clips.video_part("my_clip"),  # None unless an H.264 mp4 of at most 60 s
                      context={"arm": "v11@my-model", "episode_key": ep.episode_id})
```

## Status

Beta, single-author. Schemas are versioned but may change before 1.0. Linux and macOS (Windows is
not a target). License: [Apache-2.0](LICENSE).
