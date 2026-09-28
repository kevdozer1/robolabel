# Schema

`robolabel` writes two artifacts: a VLM annotations file (`annotations.parquet`)
and, separately, a human gold file (`*.json`). They are never merged; VLM labels
and human labels live in different files so neither can silently overwrite the
other.

## `annotations.parquet` (VLM output)

Schema version: **`robolabel/annotations/v6`** (stored in every row's
`schema_version` column; bump it on any breaking change). Long format: one row
per record, three record types per episode. The experimental V-lite pipeline writes
**`robolabel/annotations/v7`** instead, with more record types; see
[v7 (V-lite output)](#v7-v-lite-output-experimental) below. The experimental v1.1 pipeline writes v7
with a few more optional columns; see [v1.1 additions](#v11-additions-video-first-experimental).

**v2** adds three columns for the annotation-strategy layer: `phase` and
`boundary_evidence` (per subtask) and `strategy` (per episode). **v3** adds one
more per-subtask column, `target`: the grounded object/destination slot, so a
label reads `phase → target` (e.g. `approach → red cube`). **v4** adds four
**deterministic, data-derived** conditioning columns (no VLM): `control_modality`
(per episode), `active_dof` (per subtask), and a `retrieved_subgoal_episode_id` /
`retrieved_subgoal_frame_idx` pair (per subgoal: a same-phase keyframe from a
*different* episode, stored **alongside** the real `subgoal_frame_idx`, never
replacing it). All of these are optional. The change is purely **additive**: **v1,
v2, and v3 files still read**; absent columns are treated as null. The v4 fields
are written by `robolabel enrich` (see `control.py` / `retrieve.py`), not by the
annotate pass.

`control_modality` and `active_dof` are **two independent axes**, easy to conflate:

- **`control_modality`** is the action **coordinate frame**: `joint` (joint-angle targets) vs
  `end-effector` (Cartesian poses), read from the action feature names. It is **dataset-level
  and constant** (an SO-101 is `joint` either way) and is pi0.7's control-modality field. It says
  nothing about motion.
- **`active_dof`** is the **set of component groups that actually move** in a segment (`arm`,
  `gripper`, `arm+gripper`, `none`). It is **per-segment** and is *not* a pi0.7 field. A group is
  active iff one of its dims moves through more than a threshold fraction of its full-episode range
  within the segment, measured as the **smoothed within-segment range** (its excursion). Using the
  range rather than net start-to-end displacement means a gripper that opens to release and then
  recloses still counts, while a light moving-average rejects single-frame jitter and a gripper
  that merely **holds** a position has near-zero range and is not counted. Groups are auto-derived
  from the action names and generalize to N groups, with members named for components (`arm`,
  `gripper`, and so on), never `joint`/`end-effector`.

**v5** adds five episode-level **curation** columns, all deterministic (no VLM):
`speed` (`fast`/`medium`/`slow` bin) + `speed_norm` (the underlying scalar),
`novelty`, and `curation_value` / `curation_tier`. Written by the `robolabel run`
modules (`speed.py`, `novelty.py`, `curation.py`). **v6** adds the continuous,
phase-agnostic motion descriptor `active_frames` / `active_seconds` /
`active_fraction` (the primary speed signal, motion-defined). Two semantics notes
that travel with these: (1) the **raw** continuous fields (`speed_norm`, `novelty`,
`curation_value`, `active_*`) are always emitted; (2) the **tier** fields (`speed`,
`curation_tier`) are **corpus-relative and guarded**: pooled across all
episodes/categories with global thresholds, and left **null** ("insufficient
population to tier") on a population too small or homogeneous to tier honestly,
rather than fabricating bands. Still purely additive: **v1..v5 files still read**.

| column | type | record types | meaning |
|---|---|---|---|
| `schema_version` | str | all | `robolabel/annotations/v6` |
| `source` | str | all | always `vlm` in this file |
| `episode_id` | str | all | stable id from the adapter |
| `task` | str? | all | task string if the dataset has one |
| `num_frames` | int | all | episode length in frames |
| `fps` | float | all | frames per second |
| `record_type` | str | all | `episode_metadata` \| `subtask` \| `subgoal` |
| `segment_idx` | int? | subtask, subgoal | 0-based subtask index |
| `start_frame` | int? | subtask | inclusive start frame |
| `end_frame` | int? | subtask | inclusive end frame |
| `subtask_text` | str? | subtask | short action phrase |
| `phase` | str? | subtask | **v2**; closed-vocabulary phase (S2+), e.g. `approach`/`grasp` |
| `target` | str? | subtask | **v3**; grounded object/destination this subtask acts on (S2+), e.g. `red cube`; null for `retract` |
| `boundary_evidence` | str? | subtask | **v2**; one-line visual evidence for the boundary (S1+) |
| `active_dof` | str? | subtask | **v4**; the **set of component groups that move** over the segment, `+`-joined and sorted (`arm`, `gripper`, `arm+gripper`, `none`). Deterministic, from each dim's **smoothed within-segment range** (a held gripper has ~zero range; a release open-and-reclose still counts). Groups auto-derived from action names; generalizes to N groups |
| `quality` | int? | episode_metadata | curation/training-usefulness, 1–5 |
| `task_success_quality` | int? | episode_metadata | task-completion score, 1–5 |
| `mistake` | bool? | episode_metadata | clear visible mistake |
| `boundary_clarity` | str? | episode_metadata | e.g. `clear`/`partial`/`weak` |
| `control_mode` | str? | episode_metadata | legacy strategy metadata if provided (superseded by `control_modality`) |
| `control_modality` | str? | episode_metadata | **v4**; `joint`/`end-effector`: the action **coordinate frame** (joint targets vs Cartesian poses), from the action feature names. NOT gripper involvement |
| `reason` | str? | episode_metadata | the VLM's stated evidence |
| `speed` | str? | episode_metadata | **v5**; `fast`/`medium`/`slow` tier, **corpus-relative** (pooled, guarded); **null** when the population is too small/homogeneous to tier |
| `speed_norm` | float? | episode_metadata | **v5**; raw scalar: mean per-step action velocity |
| `novelty` | float? | episode_metadata | **v5**; raw: mean distance to nearest neighbours in a frame embedding (corpus-pooled when rescored); higher = rarer |
| `curation_value` | float? | episode_metadata | **v5**; raw `f(quality, novelty)` in [0,1], weights from the run config; always emitted |
| `curation_tier` | str? | episode_metadata | **v5**; value-tiered overlay `full`/`reduced`/`minimal` (or `keep`/`cut`), **corpus-relative + guarded**; **null** = "insufficient population to tier". Never deletes data |
| `active_frames` | int? | episode_metadata | **v6**; frames from motion onset to offset: **motion-defined, phase-agnostic** (not tied to named phases) |
| `active_seconds` | float? | episode_metadata | **v6**; `active_frames / fps` |
| `active_fraction` | float? | episode_metadata | **v6**; `active_frames / num_frames` |
| `subgoal_frame_idx` | int? | subgoal | frame index of the **real** end-of-sub-step subgoal (ground truth) |
| `subgoal_image_path` | str? | subgoal | extracted PNG path (if `--no-images` not set) |
| `retrieved_subgoal_episode_id` | str? | subgoal | **v4**; episode the retrieved (same-phase) subgoal came from; null if none |
| `retrieved_subgoal_frame_idx` | int? | subgoal | **v4**; frame index of the retrieved subgoal in that other episode |
| `provider` | str | all | provider name (gemini/openai/qwen/mock) |
| `model` | str | all | model id |
| `strategy` | str? | all | **v2**; annotation strategy name (`S0`..`S4`); null == baseline |
| `cost_usd` | float? | episode_metadata | estimated cost for this episode's calls |
| `receipt_path` | str? | episode_metadata | directory of raw per-call receipts |

Subtasks for an episode are contiguous, non-overlapping, and cover `[0, num_frames-1]`.
Subgoal frames default to each subtask's `end_frame`.

Row order is deterministic (episode_id, then record type, then segment). Absolute
paths (`subgoal_image_path`, `receipt_path`) naturally depend on the output
directory; everything else is reproducible from the same inputs.

### Side files under the output directory

```
<out>/annotations.parquet
<out>/strategy.json                                                         # resolved strategy config (provenance)
<out>/raw_receipts/<episode_id>/{subtasks,metadata}_{observe,label}.json   # raw VLM responses
<out>/raw_receipts/<episode_id>/refine_b<k>.json                            # S3+ boundary-refinement calls
<out>/subgoal_frames/<episode_id>_seg<k>_f<frame>.png                       # extracted subgoals
```

Receipts never contain image bytes; they hold the request question, the raw
response JSON/text, status, latency, and (where the provider reports it) token
counts.

## v7 (V-lite output, experimental)

Schema version: **`robolabel/annotations/v7`**. Only the experimental pipelines write it:
V-lite (`robolabel.schema_v7.write_v7` on the rows that `robolabel.vlite.run_episode`
returns; see [CONFIG.md](CONFIG.md#v-lite-experimental)), and v1.1 with the additions described
[below](#v11-additions-video-first-experimental). `robolabel run`, `annotate` and `demo` still
write v6.

v7 is additive. It is the same long-format `annotations.parquet` with every v6 column,
65 new columns and five new record types: `coarse_subtask`, `attempt`, `scene_fact`,
`requirement` and `check`. **v1 to v6 files still read** (`robolabel.schema.read_annotations`
reads any version), and code that selects rows by `record_type` finds the v6 record types
where it expects them. Nested values (boxes, points, evidence, visibility) are stored as
JSON text, so every column has one type. A column that a record type does not use is null.

Every v7 row carries `schema_version`, `source` (`vlm`), `episode_id`, `task`, `num_frames`,
`fps`, `provider`, `model`, `strategy` (`v-lite`) and the new column `arm` (the run label
given to `run_episode`). Rows are ordered by episode, arm, record type (in the order of the
sections below), then segment. V-lite writes no `subgoal` rows. Objects are named by inventory
IDs (`o1`, `o2`, ...), or `none` / `unsure`; a goal reference the model gave as text that
matches no inventory name stays as that text.

### `episode_metadata` (one per episode)

| column | type | meaning |
|---|---|---|
| `goal_objective` | str? | the goal's objective sentence (L4); null without a goal |
| `primary_target` / `primary_destination` | str? | the goal's main object and destination |
| `goal_source` | str? | `single_episode` when there is a goal |
| `episode_outcome` | str | `success`, `failure`, `partial` or `unknown`, over the required goal items |
| `n_attempts` / `n_failed_attempts` | int | attempts among the segments, and those with a failed segment |
| `speed_steps` | int | episode length in frames |
| `label_risk` | float? | L5 risk: failed rules over applicable rules (null when no rule applies) |
| `review_status` | str | `routed` (L5 flagged the episode for review) or `auto` |
| `cameras_used` | str | the episode's cameras, comma-joined |
| `pipeline_version` | str | `v-lite` plus the prompt version |
| `layer_models_json` | JSON | what produced each layer: `{"L1": "signal", "L2": <model>, "L3": <model>, "L4": <model>, "L5": "rules"}` |
| `cost_usd` | float | (v6 column) the episode's model-call cost |

### `subtask` (one per fine segment)

The v6 columns are filled too, so older readers see the segments: `segment_idx`,
`start_frame`, `end_frame`, `subtask_text` (the phase text), `phase` (= `phase_class`) and
`target` (the target's inventory ID). Segments are contiguous and cover `[0, num_frames-1]`.

| column | type | meaning |
|---|---|---|
| `phase_class` | str | `approach`, `grasp`, `transport`, `release`, `retract`, `press`, `pour`, `insert`, `fold`, `wipe`, `push`, `pull`, `rotate`, `open`, `close` or `other` |
| `target_object_id` / `destination_object_id` | str? | what the segment acts on, and where it goes |
| `attempt_idx` | int | the grasp attempt the segment belongs to |
| `outcome` | str | `success`, `failed` or `aborted` |
| `failure_type` | str | `none`, `missed_grasp`, `slip`, `drop`, `wrong_object`, `misplace`, `press_no_effect`, `aborted` or `other` |
| `mistake` | bool | (v6 column, now per segment) true when `outcome` is `failed` |
| `boundary_source` | str | `signal` when the segment ends on the frame of an L1 candidate the model confirmed, else `vlm` |
| `boundary_confidence` | float | 0.9 for `signal`, 0.5 for `vlm` |
| `evidence_frame` / `evidence_camera` | int? / str? | the first evidence item |
| `evidence_json` | JSON | the evidence items, at most 3: `[{"frame", "camera", "statement"}]` |
| `coarse_idx` | int? | the coarse subtask that contains this segment |

### `coarse_subtask` (one per coarse subtask)

Coarse subtasks group the fine segments and are rendered from fixed templates, such as
"pick up the red cube" or "move the arm away". Columns: `coarse_idx`, `start_frame`,
`end_frame`, `coarse_text` (also in `subtask_text`), `target_object_id`,
`destination_object_id`, and `mistake` (true for a group of failed or aborted phases).

### `attempt` (one per grasp attempt, from L1)

Measured from the gripper signal, not from the model.

| column | type | meaning |
|---|---|---|
| `attempt_idx` | int | 1-based attempt number |
| `start_frame` / `end_frame` | int | from the closing onset to the attempt's end |
| `outcome` | str | `hold`, `empty`, `slip`, `released`, `aborted` or `unknown` |
| `failure_type` | str | `none`, `missed_grasp`, `slip` or `aborted` |
| `evident_frame` | int | the frame where the outcome shows |
| `attempt_source` | str | `signal` |
| `confidence` | float | 0.8 |

### `scene_fact` (what the model reported seeing, L2)

`frame_idx`, `camera` and `object_id` say where and what; `source_model` is the model that
reported it (there is no detector yet, so these are the model's own reports). The kind of fact
is in `predicate`:

| `predicate` | other columns |
|---|---|
| `inventory` | one row per object and camera where the inventory saw it, at `frame_idx` 0: `object_name`, `category`, `point_json` (`[x, y]`), `box_json` (`[x0, y0, x1, y1]`), both normalized to 0..1, `visibility` `visible` |
| `visible` | the object is visible in that image; `visibility` is `visible` or `partial`; `box_json` at the last frame |
| `in_gripper` | `object_id` is what the gripper holds: an ID, `none` or `unsure` |
| `inside`, `on_top_of`, `touching` | a relation from `object_id` to `ref_object_id`; `value` is `true`, `false` or `unsure` |

### `requirement` (one per goal item, L4)

The goal is a list of end-state requirements, not a command.

| column | type | meaning |
|---|---|---|
| `req_id` | str | `r1`, `r2`, ... |
| `kind` | str | `object_end_state` or `robot_end_state` |
| `object_id` / `ref_object_id` | str | the object (`none` for robot items) and the reference object |
| `predicate` | str | `inside`, `on_top_of`, `touching`, `in_gripper`, `lifted`, `at_location`, `unchanged`, `activated`, `state`, `holding`, `gripper_open`, `gripper_closed`, `withdrawn`, `at_home_pose`, `near_object`, `tool_lifted` or `other` |
| `value` | str | `true`, `false` or `unsure` (free text for `state`) |
| `status` | str | `required`, `incidental` or `unsure` |
| `unsure_kind` | str? | `perception` or `intent` when `status` is `unsure` |
| `basis` | str | `task_string`, `physical_necessity`, `observed`, or `signal` for a robot item added from L1 |
| `achieved` | str | `true`, `false` or `unknown` at the end of the episode |
| `deciding_frame` / `deciding_camera` | int / str | where it is decided |
| `visibility_json` | JSON | camera to `visible`, `partial` or `not_visible` |
| `reason` | str | (v6 column) the model's short reason |

### `check` (one per rule, L5)

Ten rules per episode, free (no model call). Columns: `check_id` (the rule, 1 to 10),
`verdict` (`pass`, `fail` or `na`), `check_note`, `target_record` (`episode`),
`rule_or_question` (`rule <n>`), `checker` (`rule`) and `cost_usd` (0).

### Reserved columns

`evidence_visibility`, `outcome_confidence`, `speed_bin_pi07`, `execution_quality`,
`mask_path`, `consensus_present`, `consensus_of` and `probability` are part of the v7 column
set, but V-lite does not fill them yet (always null).

### The view record

`run_episode` also returns a view record: one plain-JSON dict per episode, which robolabel
does not write to disk itself. Its keys: `arm`, `episode_key`, `family`, `fps`,
`num_frames`, `cameras`, `camera_sizes`, `task`, `objects`, `segments`, `coarse`,
`attempts`, `goal`, `episode_outcome`, `checks`, `risk`, `routed`, `route_reasons`,
`cost_usd`, `calls`, `wall_s`, `valid`, `repairs`, `no_output`, `cache_hits`,
`pipeline_version`, `pipeline_code` (a hash of the code that wrote it),
`candidate_verdicts` and `step_status`. Objects in the view are named, not only numbered.
The `episode_metadata` row dict that `run_episode` returns also has `pipeline_code`, but it is
not a v7 column, so `write_v7` leaves it out of the parquet.

## v1.1 additions (video first, experimental)

The experimental v1.1 pipeline (`robolabel.vfirst.run_episode_v11`; see the README) returns v7
rows with 11 more columns, and `robolabel.schema_v7.write_v11` writes them. `write_v7` and the
V-lite output are unchanged.

v1.1 is additive to v7. It adds no record type and keeps `schema_version`
**`robolabel/annotations/v7`**: every new column is optional, and a reader that does not know
them sees a v7 file. **v1 to v7 files still read**: `robolabel.schema_v7.read_v11` reads any
version and adds every v1.1 column (null where the file has none), `episode_record_v11` and
`subtask_records_v11` return typed rows, and `subtask_records_v11` derives `attempt_outcome` for
a file that has none (the old rule, below; `derive=False` leaves it null, as in the file).
`event_sources` is stored as comma-joined text and
read back as a list; `has_end_state` and `crawl_enabled` are nullable booleans.

v1.1 rows carry `strategy` `v1.1` and `pipeline_version` `v1.1` plus the prompt version (for
example `v1.1 v8-2026-09-27.1`). `layer_models_json` gains `crawl` (the crawl model, or `none`),
and its `L1` is `signal` with the gripper event source, else `none`.

### `subtask` (v1.1)

| column | type | meaning |
|---|---|---|
| `end_event` | str? | the type of the boundary at the segment's end: `close_start`, `open_start`, `contact_start`, `contact_end` or `other` (always `other` on the last segment) |
| `coarse_end_frame` | int? | the end frame the coarse pass proposed, before the crawl or a snap to the signal moved it |
| `crawl_calls` | int? | crawl calls made for the boundary at the segment's end (0 when it was not crawled) |
| `attempt_outcome` | str? | `success`, `failed` or `aborted`: the result of the whole attempt the phase belongs to, the same on each of its phases |
| `event_sources` | str? | the run's event sources, comma-joined: `none`, `motion` or `gripper` |

What v7 columns mean in v1.1 rows:

- `boundary_source` gains two values: `coarse` (the end the coarse pass proposed) and `crawl`
  (the onset the crawl found). `signal` is a `close_start` or `open_start` boundary that took the
  frame of an L1 event. `boundary_confidence` is 0.9 for `signal` and 0.5 otherwise.
- `outcome` is the phase's own result: in a missed grasp the approach is `success` and the grasp
  `failed`. At most one phase per attempt is `failed`, `failure_type` is `none` on every
  successful phase, and `mistake` is true only on the failed phase.
- `subtask_text` holds `phase_text`, the phase in the model's own words; `phase_class` is `other`
  where no class fits.
- `target_object_id`, `destination_object_id` and `target` are inventory IDs when the inventory
  lists objects, else the model's plain words (or `none` / `unsure`).

### `episode_metadata` (v1.1)

| column | type | meaning |
|---|---|---|
| `has_end_state` | bool? | false when the activity has no object end state (a dance, a wave, a gesture); null without a goal, or when the answer gave neither true nor false |
| `goal_command` | str? | the goal as a command for training prompts, rendered from the required object end states, such as "Put the pink brick in the transparent box"; empty when none renders; null without a goal |
| `event_sources` | str? | the run's event sources, comma-joined |
| `coarse_mode` | str? | `frames` or `video` |
| `coarse_fps` | float? | the rate of the frames the coarse pass saw: their count minus one, over the seconds from the first to the last; null in `video` mode |
| `crawl_enabled` | bool? | whether the crawl was on |
| `crawl_model` | str? | the crawl's model; null when the crawl was off |

`goal_objective` is kept a state sentence (null without a goal): an objective that the state check
reads as a command (it starts with a known imperative verb, a fixed list in
`robolabel.layers.goal.is_state_objective`), or an empty one, is replaced by one rendered from the
required object items ("No object has a required end state." when none renders). The check is a word
list, so a command that starts with a verb outside it is kept as written.

### `attempt` (v1.1)

Every v1.1 run writes one row per attempt of the segments, with `attempt_source` `vlm`:
`start_frame` and `end_frame` span the attempt from its first phase to its last, `outcome` is its
`attempt_outcome` (`success`, `failed` or `aborted`), `failure_type` is that of its failed or
aborted phase (else `none`), `evident_frame` is the last frame of that phase (null for a
successful attempt), and `confidence` is 0.5. With the gripper source, the v7 rows measured by L1
(`attempt_source` `signal`) are written as well.

### `requirement` (v1.1)

Without the gripper source, the L1 record is never read: no robot item is added from it, and the
robot items are the model's own. With it, a robot item that the model left `required` with
`achieved` `unknown`, or marked `unsure` / `perception`, takes `achieved` from L1 when L1 decides
it, with `basis` `signal` (an `unsure` item then becomes `required`).

### `check` (v1.1)

Thirteen rules per episode: rules 1 to 10 as in v7, then 11 (every crawl pick lies inside a window
the model saw), 12 (no boundary moved by the crawl crosses a neighbouring boundary) and 13
(`has_end_state` false and no `object_end_state` item). Without the gripper source, rules 1, 2, 6,
7 and 8 need the signal and are `na`. Rules 11 to 13 count in `label_risk` but never route. A call
that failed, was refused, was unavailable, was stopped by the spend guard or stayed invalid (and a
step not run after a stop), or a coarse pass with no output, sets `label_risk` to 1.0 and
`review_status` to `routed`.

### The view record (v1.1)

`run_episode_v11` returns a view record with the V-lite keys except `candidate_verdicts`, and with
the v1.1 `pipeline_version` and `pipeline_code`. `step_status` covers the crawl too. Each segment
gains `end_event`, `coarse_end_frame`, `crawl_calls` and `attempt_outcome`, and `attempts` holds
one record per attempt of the segments: `attempt_idx`, `start`, `end`, `outcome` (the attempt
outcome), `failure_type`, `evident_frame` and `source` (`vlm`). New keys:

- `attempts_signal`: L1's own attempts in the same shape (`source` `signal`, L1's outcome words),
  null without the gripper source;
- `has_end_state`, `goal_command`, `event_sources` (a list), `coarse_mode`, `coarse_fps`,
  `crawl_enabled`, `crawl_model` and `crawl_calls` (the episode's total);
- `events`, the event source's events (below);
- `crawl_log`, one entry per typed boundary: `boundary_index`, `event_type`, `object` (the object a
  contact question names), `coarse_frame`, `stage1_frames` and `stage1_answer`, `retry_frames` and
  `retry_answer`, `stage2_frames` and `stage2_answer`, `pick` (the frame of the image the deciding
  answer points at), `onset`, `flags`, `calls`, `usd` and `call_log` (per call: `stage`, `frames`,
  `answer`, `read_as`, `status`, `usd`, `wall_s`, `cache_hit`). The flags are explained
  [below](#crawl-log-flags-v11);
- `coarse_status`, `failed_calls`, `camera`, `inventory_frames`, `keyframes`, `keyframe_source`
  (`boundaries`, or `signal` with the gripper source), `prompt_version` and `prompt_hashes`.

Many view keys say how a label was made: the event and crawl keys (`events`, `event_sources`,
`crawl_log`, `crawl_enabled`, `crawl_model`, `crawl_calls`), `coarse_mode`, `coarse_fps`,
`step_status`, `failed_calls`, `keyframe_source`, `attempts_signal`, `checks`, `repairs`,
`prompt_hashes`, a segment's `boundary_source`, `coarse_end_frame` and `crawl_calls`, a requirement's
`basis` and `added_by`, and `layer_models_json` in the rows. A blind comparison of setups should
therefore show only the fields it needs, such as the segments' `start`, `end`, `phase_text`,
`phase_class`, `target_name`, `destination_name` and `outcome`, and the goal's `objective`, rather
than hide a list of fields.

### Crawl log flags (v1.1)

| flag | meaning |
|---|---|
| `crawl_edge` | stage 1 answered 0 or 9 and no image pick followed: the window could not move, the call cap came first, or the retry did not pick an image (or picked one that contradicts stage 1). The onset that the edge answer supports is kept (the retry's, when it gave the same edge), or the coarse frame with `crawl_none` when that onset lies outside the clip |
| `crawl_none` | stage 1 answered -1, or an edge answer put the onset at frame 0 or past the last frame (the event is not inside the clip); the coarse frame stays |
| `crawl_inconsistent` | a later answer contradicts stage 1 (a retry pick against its edge, a retry at the opposite edge, or a 0 or 9 in stage 2); the stage-1 result is kept |
| `crawl_cross` | the refined onset would cross a neighbouring boundary; the coarse frame stays |
| `crawl_failed` | a crawl call failed or its answer was outside the allowed set; the frame known at that point is kept |
| `crawl_call_cap` | the per-boundary call cap was reached before the retry or stage 2; the result so far is kept |
| `skipped_type` | not crawled: the boundary's type is in `skip_types` |
| `skipped_signal` | not crawled: the boundary took the L1 frame (`boundary_source` `signal`) |
| `skipped_stopped` | not crawled: the spend guard had stopped paid calls |
| `skipped_cap` | not crawled: the episode's 12 crawled boundaries were already used |
| `skipped_short` | not crawled: the clip has fewer than 3 frames |

### Events and the L1 record (v1.1)

An event is `{"type", "frame", "confidence", "source", "attempt_idx"}`: `frame` is the onset (the
first frame of what the event starts), `confidence` a float from 0 to 1 (4 decimals) and
`attempt_idx` an int or null. Types by source: `gripper` gives `close_start`, `open_start` and
`arm_move`; `gripper_recovery` (a label of the gripper source) gives `open_start` and `back_off`;
`motion` gives `pause_start` and `pause_end`; `none` gives none.

The L1 record (`robolabel.layers.signal.run_l1`) gains `recovery_candidates`, a list of
`{"type", "frame", "confidence", "attempt_idx"}`. After an `empty` or `aborted` close whose fingers
open again it holds an `open_start` (confidence `high`) at the reopening onset minus 1 and, when
the arm then starts moving away from its pose at the close, a `back_off` (confidence `low`), both
with the failed attempt's index. Frames follow the L1 candidate convention (a candidate frame ends
the earlier segment). `recovery_version` names the rule. Every other L1 field, and `code_version`,
is unchanged.

### `attempt_outcome` in older files and in gold

A file or gold record without `attempt_outcome` gets it derived by the old rule
(`robolabel.eval.derive_attempt_outcome`): the phases that share an `attempt_idx` are one attempt,
which is `failed` when any of them failed or has `mistake` true, else `aborted` when one was
aborted, else `success`; a segment without `attempt_idx` is an attempt of its own. Present values
are kept. Gold v2 segments (`robolabel.eval.gold_v2`) take an optional `attempt_outcome`; files
without it still validate, and `with_attempt_outcome` derives it.

## Gold file (human labels)

Schema version: **`robolabel/gold/v1`**. One JSON object with an `episodes` list.
Each episode has an `auto` block (a snapshot of the VLM labels) and a `gold` block
(what the human enters). `accept_auto` flags mean "the human confirms the VLM
value here".

```json
{
  "schema_version": "robolabel/gold/v1",
  "episodes": [{
    "episode_id": "0",
    "task": "pink lego brick into the transparent box",
    "num_frames": 303,
    "auto":  {"subtasks": [...], "metadata": {"quality": 4, ...}, "subgoals": [...]},
    "gold":  {"subtasks": [{"segment_idx": 0, "start_frame": null, "end_frame": null,
                            "subtask_text": null, "accept_auto": null}],
              "metadata": {"quality": null, "mistake": null, "reason": null, "accept_auto": null},
              "subgoals": [{"segment_idx": 0, "frame_idx": null, "accept_auto": null}]},
    "review_notes": ""
  }]
}
```

The reliability report compares `auto` vs `gold` per episode and aggregates:
subtask boundary temporal IoU, quality exact / within-one agreement, subgoal frame
agreement.

## LeRobot subtask-convention export

`export --format lerobot` writes our subtask segments into the **subtask convention
the pinned lerobot (0.4.x) actually reads back**, verified against the installed
source, not guessed. Two files under `<out>/meta/`:

| file | schema | matches |
|---|---|---|
| `meta/subtasks.parquet` | **string-indexed** table (index = subtask phrase), one column `subtask_index` (0..N-1) | mirrors `meta/tasks.parquet` exactly; `LeRobotDataset` resolves a frame's subtask via `meta.subtasks.iloc[subtask_index].name` |
| `meta/episodes_subtasks.parquet` | one row per episode: `subtask_indices`, `subtask_names`, `subtask_start_frames`, `subtask_end_frames`, `subtask_start_times`, `subtask_end_times` | the per-frame `subtask_index` column is reconstructable from this (we don't rewrite the binary `data/` parquet); the `subtask_*` columns are SARM-compatible |

Pick the subtask string with `--subtask-field {subtask_text,phase}` (default
`subtask_text`).

**What survives the export:** the subtask temporal boundaries and the subtask phrase.

**What stays only in `annotations.parquet`** (the LeRobot subtask convention has no slot for it): the
per-boundary `boundary_evidence`, the closed-vocabulary `phase` tag, the episode
`quality` / `task_success_quality` / `mistake` / `reason`, the provider receipts, and
`cost_usd`. `annotations.parquet` remains the full-fidelity record.

**Not emitted:** `meta/tasks_high_level.parquet` and `task_index_high_level` are part
of the separate **LeRobot Annotate** GUI (`huggingface/lerobot-annotate`) and a newer
lerobot than the pinned 0.4.x core; they are intentionally not written. The round-trip
test reloads `meta/subtasks.parquet` through lerobot's own `load_subtasks` and confirms
every frame's `subtask_index` resolves to the subtask segment it falls in.

`annotations.parquet` + `export --format jsonl` remain the portable, full-fidelity outputs.
