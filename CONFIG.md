# Run config

`robolabel run --config run.yaml` drives the whole pipeline from one YAML file. A run-config has
a `run` block (dataset / model / probe) and a `modules` block where **each module is
independently toggleable**. The minimal default runs only **segmentation + quality** with
**open-vocabulary grounded** segmentation. Modules execute in dependency order; dataset-level
modules (novelty, curation, retrieval) run after the per-episode pass.

For a standard LeRobot dataset you provide **nothing** beyond `source`/`target`: camera key,
fps, control space, and arm/gripper dims are auto-detected (see [`PORTING.md`](PORTING.md)).

## Minimal (copy-paste)

```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace }   # camera_key: auto
  model:   { provider: gemini, name: gemini-2.5-flash }
  probe:   { max_episodes: 10 }
  out: run_out/pickplace
# modules: omitted -> the default is segmentation + quality only
```

```bash
robolabel run --config run.yaml
```

## Everything on

```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace, camera_key: auto }
  model:   { provider: gemini, name: gemini-2.5-flash }
  probe:   { max_episodes: 5 }
  out: run_out/full
  seed: 0
modules:
  segmentation: { enabled: true, strategy: grounded, vocabulary: open }   # vocabulary: closed for S2
  quality:      { enabled: true }
  speed:        { enabled: true, cuts: [0.3333, 0.6667] }
  subgoals:     { enabled: true, retrieval: true, retrieval_method: embedding }
  control:      { enabled: true, active_dof: true }                       # per-segment active-component set
  novelty:      { enabled: true, k: 5 }
  curation:     { enabled: true, compress: true,
                  weights: { quality: 0.5, novelty: 0.5 }, top_cut: null }
```

## Modules

| module | scope | default | requires | does |
|---|---|---|---|---|
| `segmentation` | episode | **on** | none | grounded `phase → target` subtasks. `vocabulary: open` (default) = `S2-open`; `closed` = `S2`; `strategy: baseline` = S0 |
| `quality` | episode | **on** | none | episode quality 1–5 (VLM). Near-degenerate on easy datasets; see `speed` |
| `speed` | episode→dataset | off | none | continuous, **motion-defined** `active_frames`/`active_seconds`/`active_fraction` (phase-agnostic) + raw `speed_norm`; a `fast`/`medium`/`slow` tier only when corpus-relative (else null) |
| `subgoals` | episode→dataset | off | `segmentation` | real end-of-sub-step keyframe (pointer); `retrieval: true` adds a same-phase keyframe from another **gate-passed** episode (pointer). No image files written |
| `control` | episode | off | `segmentation` | `control_modality` (joint vs end-effector coordinate frame, dataset-level); `active_dof: true` (default on) adds the per-segment **set of component groups that move** (`arm`/`gripper`/`arm+gripper`/`none`), from each dim's smoothed within-segment range |
| `novelty` | dataset | off | none | deterministic per-episode novelty (distance to nearest neighbours in a cheap frame embedding) |
| `curation` | dataset | off | `quality`, `novelty` | raw `curation_value = f(quality, novelty)`; tiers (`full`/`reduced`/`minimal`, or `keep`/`cut`) are **corpus-relative + guarded**: assigned only when ≥ `min_population` heterogeneous episodes exist (else null, "insufficient population to tier"). Overlay only; never deletes |

A module whose `requires` are not all enabled raises a clear error at validation. Everything is
additive in the output (schema v6); see [`SCHEMA.md`](SCHEMA.md). All deterministic modules
(`speed`, `control`, `novelty`, `curation`) cost **$0**; only `segmentation`/`quality` call the
VLM, and `robolabel run` reports per-module cost.

## Provider names

`model.provider` in a run config (and `--provider` on `robolabel annotate`) takes a registered
provider name: `gemini`, `openai`, `qwen`, `mock` or `openrouter`. The run config's default is
`gemini` with `gemini-2.5-flash`. `openrouter` reads its key from `OPENROUTER_API_KEY` and takes an
OpenRouter model id as `model.name` (or `--model`); always set it, since the default name is a Gemini
one. Built by name like this, it runs without a spend guard or response cache.

```yaml
run:
  model: { provider: openrouter, name: vendor/model-name }   # an OpenRouter model id
```

## V-lite (experimental)

V-lite adds no subcommands, run-config keys or options: the experimental V-lite pipeline (schema v7)
has no CLI subcommand yet. It is called from Python (`robolabel.vlite.run_episode`), with the spend
guard and response cache set up as in the README's
[Spend guard and cache](README.md#spend-guard-and-cache). Its output is described in
[SCHEMA.md](SCHEMA.md#v7-v-lite-output-experimental).

## v1.1 video first (experimental)

v1.1 adds no subcommand, run-config key, CLI option, dependency or extra: `robolabel --help` and the
run config are as before. Like V-lite it is called from Python, and its options are keyword
arguments. The pipeline itself runs on the core dependencies; the clip-folder adapter needs the
`video` extra (PyAV, decode only), and scoring needs the `eval` extra, as for V-lite. See
[How it works](README.md#how-it-works) in the README.

### `robolabel.vfirst.run_episode_v11(episode, *, camera, caller, context, ...)`

The full pipeline on one episode. It returns `view`, `rows` (write them with
`robolabel.schema_v7.write_v11(rows, out_dir)`), `calls`, `repairs` and the intermediate results.

| option | default | meaning |
|---|---|---|
| `camera` | required | the camera the model sees; one camera per run |
| `caller` | required | the model: any object with `.call(CallRequest) -> CallResult`, such as `OpenRouterProvider` or `MockProvider` |
| `context` | required | a dict sent with every call: `arm` (the run label written to every row, default `v1.1`), `episode_key`, `bucket` (the spend-guard bucket; the OpenRouter provider uses `sweep` when it is absent) and `model_key` (for a per-model cap) |
| `event_source` | `none` | `none`, `motion` or `gripper` |
| `l1` | None | the episode's L1 record (`robolabel.layers.signal.run_l1`): required with `gripper`, a ValueError with any other source, and its `num_frames` must equal the episode's |
| `coarse_mode` | `frames` | `frames` or `video` |
| `video` | None | a `robolabel.providers.base.VideoPart`, required with `coarse_mode="video"`: `VideoPart(data=<mp4 bytes>, mime="video/mp4", seconds=<clip duration>)`; `seconds` (default 0) sizes the spend guard's worst-case reservation, one image-equivalent per second of video (at least one) |
| `crawl` | True | False turns the crawl off |
| `crawl_caller` | None | the crawl's model; None uses `caller` |
| `crawl_context` | None | the crawl calls' context; None uses `context` |
| `reasoning` | None | sent with every call, for example `{"effort": "low"}` |
| `scene_max_frames` | 8 | the inventory's frames, and at most this many keyframes for the facts |
| `image_tokens_per_image` | 1500.0 | the worst-case input tokens per image that the spend guard reserves, for every step except the crawl |
| `crawl_image_tokens` | 1500.0 | the same for crawl calls |
| `start_mode` | `json_schema_strict` | the structured-output mode the OpenRouter provider tries first; when the model rejects a mode, it moves on to `json_object_with_schema_in_prompt`, then `plain_with_schema_in_prompt` |

### `robolabel.vfirst.run_timing(episode, *, camera, coarse_caller, coarse_context, crawl_context, ...)`

Only the event source, the coarse pass and the crawl (no inventory, facts or goal calls; targets in
plain words), for timing experiments. It returns the segments before and after the crawl, the crawl
log, the calls, the events, the cost and the prompt hashes.

| option | default | meaning |
|---|---|---|
| `crawl_caller` | None | the crawl's model; None runs no crawl (unlike `run_episode_v11`) |
| `event_source` | `none` | as above |
| `l1` | None | the L1 record the `gripper` source reads; the other sources ignore it |
| `coarse_mode`, `video` | `frames`, None | as above |
| `reasoning` | None | as above |
| `coarse_image_tokens`, `crawl_image_tokens` | 1500.0 | as `image_tokens_per_image` above |
| `start_mode` | `json_schema_strict` | as above |

### Fixed values

Both pipelines use these values. The keywords named in the last column change them in the
lower-level functions; the 448 px, the 8 images and the ±1.0 s are module constants
(`robolabel.layers.frames.MODEL_MAX_SIDE`, `robolabel.layers.crawl.N_IMAGES` and
`robolabel.layers.crawl.HALF_WINDOW_S`), not keywords.

| setting | value | function and keywords |
|---|---|---|
| coarse frames | 2 per second, first and last included, at most 48, long side at most 448 px (constant) | `robolabel.layers.coarse.coarse_request(fps_target=2.0, max_frames=48)` |
| crawl caps | at most 3 calls per boundary and 12 crawled boundaries per episode; 8 images per call and stage 1 over ±1.0 s (constants) | `robolabel.layers.crawl.crawl_boundaries(max_calls_per_boundary=3, max_boundaries=12, skip_types=frozenset())` |
| motion source | long side 128 px, 20th percentile, pauses of at least 0.3 s | `robolabel.events.get_source("motion", long_side=128, percentile=20, min_pause_s=0.3)` |
| gripper source | `arm_move` and recovery events included | `robolabel.events.get_source("gripper", calibration=None, include_arm_move=True, include_recovery=True)` |
| goal command | the four forms `inside`, `on_top_of`, `activated`, `state` | `robolabel.layers.goal.goal_command(requirements, names, categories=None, extra_forms=False)` |

### Clip folders

`robolabel.adapters.clip_folder.ClipFolderSource(root, clip_ids=None, *, keys=None, max_side=512,
jpeg_quality=92)` reads one short video per folder as one episode:

```
<root>/<clip id>/clip.mp4      the clip (required)
<root>/<clip id>/task.txt      the task text (optional; its first non-empty line)
<root>/<clip id>/source.json   optional; only its "task" string is read, when there is no task.txt
```

No other file in the folder is read. Each clip is an episode with key `C/<clip id>` (or the key
`keys` maps it to), one camera named `video`, and the frame rate, size and frame count of the file.
`clip_ids` picks clips and their order (default: every folder with a `clip.mp4`, sorted). Frames are
decoded once, on first access, and kept as JPEG with a long side of at most `max_side` px.
`video_part(clip_id, *, max_seconds=60.0)` returns the file's own bytes as a `VideoPart` when it is
an H.264 mp4 of at most `max_seconds`, else None; `release(clip_id)` drops the decoded frames.

### Scoring, baselines and the held-out guard

| function | option | default | meaning |
|---|---|---|---|
| `robolabel.eval.score.score_view` | `failure_convention` | `auto` | the rule of the failed-attempt metrics (S4 in `robolabel.eval.semantic`, and G2 part (i) in `robolabel.eval.goal`): `auto` reads `v11` when a predicted segment has `attempt_outcome`, else `v7` (older views score as before); `v7` or `v11` force one |
| `robolabel.eval.semantic.s4_episode`, `robolabel.eval.goal.g2_episode` | `convention` | `auto` | as above |
| `robolabel.eval.semantic.s4_episode` | `pred_attempts` | None | the predicted attempt records (the view's `attempts`, each a whole attempt), read under `v11`; without them the failed spans come from the segments' `attempt_idx` and `attempt_outcome`. `score_view` passes the view's `attempts` |
| `robolabel.baselines.sig_only_view`, `sig_only_segments`, `uniform5_view`, `legacy_view` | `failure_convention` | `v7` | `v7` gives the views as before; `v11` applies the per-phase failure convention and adds `attempt_outcome` |
| `robolabel.eval.heldout.HeldoutGuard` | `clip_keys` | None | the clip allowlist: `C/<clip id>` keys this guard accepts as dev (also `allow_clip_keys(keys)`). Any other `C/...` key is refused and logged; with no allowlist every clip key is refused |
| `robolabel.eval.heldout.require_explicit_episodes` | `allow_clip_keys` | False | True accepts `C/<clip id>` keys in the list (the guard still decides) |

`robolabel.eval.heldout.load_clip_keys(path)` reads the `C/...` keys of a YAML or JSON file with a
`clips` list of `{key: ...}` entries, in file order.
