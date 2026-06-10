# Migration Map

`robovid_conditioner` is a deliberate min-cut extraction from a private research monorepo (`annotation_pipeline` / `bridgeengine`). This file records every monorepo module that was **ported**, **adapted**, or **deliberately left behind**, with a one-line reason. It is maintained throughout the extraction; it is the eulogy for the complexity that did not make the cut.

The monorepo is a read-only reference. No file was copied with its git history (that history contains machine-specific paths, personal documents, and API cost receipts). Everything here was re-authored or transcribed deliberately into a clean tree.

## Proposed package layout

```
src/robovid_conditioner/
  episode.py            # Episode dataclass + EpisodeSource ABC (the adapter contract)
  adapters/
    lerobot.py          # LeRobotAdapter (primary; HF hub id or local path)
    directory.py        # DirectoryAdapter (mp4s / frame dirs + optional jsonl)
  providers/
    base.py             # VLMProvider ABC, ProviderResponse, contact sheet, two-stage helper, registry
    gemini.py | openai.py | qwen.py | mock.py   # one provider per file
  rubric.py             # Rubric loader (dataclass over rubric.yaml)
  rubric.yaml           # default rubric: prompts, score defs, gate thresholds (tabletop pick-and-place)
  labelers/
    subtasks.py         # subtask temporal segmentation (two-stage observe -> label)
    metadata.py         # episode quality/strategy metadata (two-stage observe -> label)
    subgoals.py         # subgoal keyframe selection (subtask end frame) + image extraction
  annotate.py           # orchestrate labelers over a source -> AnnotationSet
  schema.py             # annotations.parquet schema (versioned) read/write
  reliability.py        # temporal IoU, exact/within-one agreement, subgoal agreement
  gate.py               # quality gate (collapsed scores, repeated text, contradictions)
  gold.py               # gold set create/merge/update — human labels kept separate from VLM
  review_app.py         # Streamlit review GUI
  cost.py               # cost aggregation from per-call receipts
  cli.py                # robovid_conditioner annotate|review|reliability|gate|export|cost|demo
  demo.py               # synthetic-episode generation + fully offline end-to-end demo
```

## Ported (transcribed and de-coupled)

| Monorepo source | robovid_conditioner target | One-line reason |
|---|---|---|
| `bridgeengine/labelers/backends.py` | `providers/base.py` + `providers/{gemini,openai,mock}.py` | The provider abstraction (raw receipts, per-call cost, retries) is the product's spine; split one-file-per-provider and drop the moondream coupling. |
| `bridgeengine/labelers/moondream_client.py` (`make_contact_sheet`, `_image_to_data_url`) | `providers/base.py` | Contact-sheet builder is provider-agnostic and reused by every provider; the Moondream HTTP client itself is left behind. |
| `bridgeengine/labelers/subtask_segmenter.py` | `labelers/subtasks.py` | The two-stage observe→label segmentation flow is core; rewritten to take an `Episode` (not a snapshot path) and to read prompts from `rubric.yaml`. |
| `bridgeengine/labelers/episode_metadata.py` | `labelers/metadata.py` | Same: two-stage quality/mistake/strategy labeling, de-snapshotted and rubric-driven. |
| `bridgeengine/derive_subgoals.py` | `labelers/subgoals.py` | Subgoal = end frame of each subtask segment; simple and portable. The gold-derivation variant is folded into `gold.py`. |
| `bridgeengine/goldset.py` (`reliability_report`, `_temporal_iou`, agreement means) | `reliability.py` | Temporal-IoU + exact/within-one + subgoal-agreement metrics are the measurement half of the thesis; re-sourced from the parquet sidecar instead of snapshot parquet. |
| `bridgeengine/goldset.py` (`write_gold_template`) + `bridgeengine/calibration.py` (`update_episode_review`, merge logic) | `gold.py` | Gold-set create/merge/update keeps human labels strictly separate from VLM labels — the product's non-overwrite rule. |
| `bridgeengine/quality_gate.py` (`_check_subtasks`, `_check_metadata`, score-dispersion, `_has_unnegated_word`) | `gate.py` | Collapsed-score / repeated-text / contradiction checks are the "gate"; thresholds moved into `rubric.yaml`. |
| `bridgeengine/review_gui.py` (Streamlit data flow only) | `review_app.py` | The calibration GUI is a required surface; re-authored small against adapter output (the 1232-line research GUI is not copied verbatim). |
| `bridgeengine/labelers/base.py` (`sha256_file`, JSON helpers, `LabelResult` idea) | `schema.py` / `providers/base.py` | Provenance helpers are reused; `LabelResult` is replaced by parquet rows. |
| `ANNOTATION_RUBRIC.md` (score rubric prose) | `rubric.yaml` + `SCHEMA.md` | The rubric becomes machine-readable config + documentation, not prose buried in a markdown file. |

## Adapted but heavily simplified

| Monorepo source | What changed | Reason |
|---|---|---|
| Snapshot system (`ingest/snapshot.py`, `snapshot_clone.py`, `snapshot_merge.py`, `LABELER_VERSIONS`) | Replaced wholesale by a single deterministic `annotations.parquet` sidecar (`schema.py`) plus optional LeRobot metadata write-back. | The heavyweight content-addressed snapshot store is research infrastructure; an independent researcher wants a sidecar file, not a database. |
| Deterministic fallback adapters in the labelers (`_segment_episode`, `_quality_and_mistake`, LeWM pilot proxy) | Dropped. The `Mock` provider is the only no-API path and is documented as meaningless. | The fallbacks existed to keep a research pipeline green without spending API budget; conflating "scaffolding" with "labels" is exactly what this tool argues against. |
| `bridgeengine/scoring.py` (`score_metadata_for_curation`, `boundary-usefulness-v3`) | Dropped. Keep only the `metadata_quality` / `task_success_quality` accessors. | A 200-line regex rescorer tuned to BridgeData pick-and-place "reasons" is the opposite of "rubric as config"; the VLM produces the score, the rubric defines it, the gate checks it. |

## Deliberately left behind (not in scope)

| Monorepo module | Reason |
|---|---|
| `bridgeengine/benchmark/` (LeWM eval, `leak_power`, `diagnostics`, `idm`, head-to-head, scale curves) | Research evaluation of whether labels help training — out of scope by mandate. The honest answer to that question lives in the linked technical report, not in this tool. |
| `bridgeengine/query/` (`duckdb_helpers`, `qdrant_helpers`) | The DuckDB/Qdrant query layer is dataset-warehouse infrastructure, not annotation. |
| `bridgeengine/export/` (`cut.py`, `webdataset_export.py`), `ingest/bridge_v2.py` | BridgeData-specific cuts and ingest; superseded by adapters + the parquet sidecar. |
| Curation/compression modules, `value.py`, `cost_probe.py`, `perceptive_status.py`, `system_check.py`, `orchestrate/` | Internal status/curation tooling for the research program. |
| `labelers/{depth,tracks,pose,masks,captions,subgoal_images,perceptive}.py` | CV signal extractors (depth/tracks/pose) belong to the LeWM aux-target experiment, not the pi0.7 conditioning-annotation product. `subgoal_images` is replaced by `labelers/subgoals.py`. |
| All `*.md` status/almanac/closeout/handoff/report files | Machine-specific narrative; the public README is written fresh. The technical report is *linked*, never bundled. |
| All `scripts/*.ps1` | Windows-only; Windows is explicitly not a target. |

## Provenance / license flags

- All ported code is re-authored from a single-author private repo owned by the same author publishing `robovid_conditioner`; no third-party copied code is carried over. Dependencies (`pillow`, `pandas`, `pyarrow`, `requests`, `pyyaml`, `numpy`, `streamlit`, `lerobot`) are used through their public APIs, not vendored.
- License: Apache-2.0 (see `LICENSE`).
- **Open flag:** `lerobot` is GPL-touching in some optional extras; `robovid_conditioner` depends on it only through the public `LeRobotDataset` read API and pins the version in the README. Confirm the pinned `lerobot` license is compatible with redistribution before tagging 1.0.
