# Schema

`robovid_conditioner` writes two artifacts: a VLM annotations sidecar (`annotations.parquet`)
and, separately, a human gold file (`*.json`). They are never merged; VLM labels
and human labels live in different files so neither can silently overwrite the
other.

## `annotations.parquet` (VLM output)

Schema version: **`robovid_conditioner/annotations/v1`** (stored in every row's
`schema_version` column; bump it on any breaking change). Long format — one row
per record, three record types per episode.

| column | type | record types | meaning |
|---|---|---|---|
| `schema_version` | str | all | `robovid_conditioner/annotations/v1` |
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
<out>/raw_receipts/<episode_id>/{subtasks,metadata}_{observe,label}.json   # raw VLM responses
<out>/subgoal_frames/<episode_id>_seg<k>_f<frame>.png                       # extracted subgoals
```

Receipts never contain image bytes; they hold the request question, the raw
response JSON/text, status, latency, and (where the provider reports it) token
counts.

## Gold file (human labels)

Schema version: **`robovid_conditioner/gold/v1`**. One JSON object with an `episodes` list.
Each episode has an `auto` block (a snapshot of the VLM labels) and a `gold` block
(what the human enters). `accept_auto` flags mean "the human confirms the VLM
value here".

```json
{
  "schema_version": "robovid_conditioner/gold/v1",
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

## Optional LeRobot write-back

Writing annotations back into a LeRobot dataset's own metadata is a planned
capability, gated on the pinned LeRobot format supporting per-episode annotation
fields cleanly. Until then, the parquet sidecar + `robovid_conditioner export` (JSONL) are
the portable outputs. See `RELEASE_READINESS.md`.
