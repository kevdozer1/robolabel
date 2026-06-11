# Schema

`robolabel` writes two artifacts: a VLM annotations sidecar (`annotations.parquet`)
and, separately, a human gold file (`*.json`). They are never merged; VLM labels
and human labels live in different files so neither can silently overwrite the
other.

## `annotations.parquet` (VLM output)

Schema version: **`robolabel/annotations/v2`** (stored in every row's
`schema_version` column; bump it on any breaking change). Long format — one row
per record, three record types per episode.

**v2** adds three columns for the annotation-strategy layer: `phase` and
`boundary_evidence` (per subtask) and `strategy` (per episode). They are optional
and null under the baseline strategy (S0). **v1 files still read** — the new
columns are simply absent and treated as null.

| column | type | record types | meaning |
|---|---|---|---|
| `schema_version` | str | all | `robolabel/annotations/v2` |
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
| `boundary_evidence` | str? | subtask | **v2**; one-line visual evidence for the boundary (S1+) |
| `quality` | int? | episode_metadata | curation/training-usefulness, 1–5 |
| `task_success_quality` | int? | episode_metadata | task-completion score, 1–5 |
| `mistake` | bool? | episode_metadata | clear visible mistake |
| `boundary_clarity` | str? | episode_metadata | e.g. `clear`/`partial`/`weak` |
| `control_mode` | str? | episode_metadata | strategy metadata if provided |
| `reason` | str? | episode_metadata | the VLM's stated evidence |
| `subgoal_frame_idx` | int? | subgoal | frame index of the subgoal |
| `subgoal_image_path` | str? | subgoal | extracted PNG path (if `--no-images` not set) |
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
the pinned lerobot (0.4.x) actually reads back** — verified against the installed
source, not guessed. Two files under `<out>/meta/`:

| file | schema | matches |
|---|---|---|
| `meta/subtasks.parquet` | **string-indexed** table (index = subtask phrase), one column `subtask_index` (0..N-1) | mirrors `meta/tasks.parquet` exactly; `LeRobotDataset` resolves a frame's subtask via `meta.subtasks.iloc[subtask_index].name` |
| `meta/episodes_subtasks.parquet` | one row per episode: `subtask_indices`, `subtask_names`, `subtask_start_frames`, `subtask_end_frames`, `subtask_start_times`, `subtask_end_times` | the per-frame `subtask_index` column is reconstructable from this (we don't rewrite the binary `data/` parquet); the `subtask_*` columns are SARM-compatible |

Pick the subtask string with `--subtask-field {subtask_text,phase}` (default
`subtask_text`).

**What survives the export:** the subtask temporal boundaries and the subtask phrase.

**What stays sidecar-only** (the LeRobot subtask convention has no slot for it): the
per-boundary `boundary_evidence`, the closed-vocabulary `phase` tag, the episode
`quality` / `task_success_quality` / `mistake` / `reason`, the provider receipts, and
`cost_usd`. The `annotations.parquet` sidecar remains the full-fidelity record.

**Not emitted:** `meta/tasks_high_level.parquet` and `task_index_high_level` are part
of the separate **LeRobot Annotate** GUI (`huggingface/lerobot-annotate`) and a newer
lerobot than the pinned 0.4.x core; they are intentionally not written. The round-trip
test (`tests/test_export_lerobot.py`) reloads `meta/subtasks.parquet` through lerobot's
own `load_subtasks` and confirms every frame's `subtask_index` resolves to the subtask
segment it falls in.

The parquet sidecar + `export --format jsonl` remain the portable, full-fidelity
outputs. See `RELEASE_READINESS.md`.
