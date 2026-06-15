# robolabel — Technical Overview

*A comprehensive engineering writeup of the whole project: what it is, why it exists, how every
layer works, what has actually been measured, and what is deliberately not claimed.*

> Scope note. This document describes the system as it stands at the `0.1.0` (unreleased) line,
> including the schema-v3 structured-label patch (`phase → target`). Numbers are quoted from the
> project's own reports (`STRATEGY_REPORT.md`, `FRESH_TRIAL_REPORT.md`, `CLAIMS.md`); code
> references point at `src/robolabel/`. Where a count drifts between docs (e.g. the test count),
> the current verified value is used and noted.

---

## 1. What it is, in one paragraph

**robolabel** reads a [LeRobot](https://github.com/huggingface/lerobot) robot-manipulation dataset,
uses a vision-language model (VLM) to **draft three kinds of annotation per episode** — *subtask
boundaries* (where one phase of a manipulation ends and the next begins), an *episode quality
score* (1–5, "is this demonstration worth training on?"), and *subgoal keyframes* — and then
**measures those drafts against a human gold set** with plain, reproducible numbers. The output is a
columnar sidecar (`annotations.parquet`) plus an optional export into LeRobot's own subtask
convention. The project's thesis is explicitly modest: it does not claim to produce *good* labels;
it claims to produce **drafts plus an honest measurement of how good they are**. The entire design —
the gold-file separation, the gate that flags but never drops, the claims audit, the blind trial — is
built around that thesis.

---

## 2. The problem and the core idea

Robot-learning datasets (LeRobot, Open-X, BridgeData, …) ship raw trajectories: video + proprioception
+ a one-line task string. Downstream methods (SARM-style subtask conditioning, goal-conditioned BC,
world-model finetuning) want *structure* on top of that: where the subtasks are, which frames are
subgoals, whether the episode is even worth keeping. Hand-labeling is expensive; a VLM can draft it
cheaply. The catch is that **a VLM draft that looks fluent can be quietly wrong**, and a labeling tool
that hides that failure mode is worse than no tool.

robolabel's answer has three pillars:

1. **Draft with a VLM, but ground the draft.** A naive "segment this video" prompt produces two
   characteristic failures (see §9): a single "complete the task" blob, or boundaries placed at
   arithmetically uniform fractions of the duration. The *annotation-strategy layer* (S0–S4) adds
   frame-indexed grounding, a closed phase vocabulary, per-boundary visual *evidence*, dense-window
   boundary refinement, and self-consistency to push the model off those failure modes.

2. **Measure everything against human gold, and keep them physically separate.** The VLM's draft and
   the human's correction live in different files (parquet sidecar vs. JSON gold), and even inside the
   gold file the `auto` (VLM) and `gold` (human) blocks never merge. Reliability is computed by
   comparing the two.

3. **Never overclaim.** A `CLAIMS.md` audit maps every public statement to the artifact that supports
   it and a status (`verified`, `verified-on-one-dataset`, `mechanical-only`, `untested`,
   `fixed-and-spot-checked-pending`). The headline result is a *negative* one (the strategy layer did
   not improve mean boundary IoU), reported as prominently as the positive ones.

---

## 3. System architecture (the pipeline in prose)

```
            ┌──────────────┐     Episode(frames, fps, task)
  dataset → │  EpisodeSource│ ───────────────────────────────►┐
 (LeRobot / │  (adapters)   │                                  │
 directory) └──────────────┘                                   ▼
                                              ┌──────────────────────────────┐
   StrategyConfig (S0..S4) ─────────────────► │      annotate_episode        │
   Rubric (rubric.yaml) ────────────────────► │  segment → metadata → subgoals│
   VLMProvider (gemini/openai/qwen/mock) ────► └──────────────────────────────┘
                                                          │  per-call receipts (no image bytes)
                                                          ▼
                                          annotations.parquet  +  strategy.json  +  raw_receipts/
                                                          │
        ┌─────────────────────────────────────────────────┼──────────────────────────────────┐
        ▼                         ▼                          ▼                                  ▼
  reliability (vs gold)      gate (flags)            inspect / review viewers          export (jsonl / lerobot)
  metrics: IoU, boundary     degenerate / uniform     multi-track timeline,             meta/subtasks.parquet,
  P/R@±5, MAE, quality       quality-outlier          evidence tab, blind trial         meta/episodes_subtasks.parquet
```

Everything to the left of `annotations.parquet` is **production** (spends API budget). Everything to the
right is **consumption** and runs **zero-API** — the analysis/reconstruction paths monkeypatch the
network call to raise, so they can only read cached receipts (a hard offline guarantee).

---

## 4. Data model and schema

robolabel writes two artifacts that are **never merged**: the VLM sidecar and the human gold file.

### 4.1 `annotations.parquet` (VLM output) — schema `robolabel/annotations/v3`

A *long-format* table: one row per record, three record types per episode (`episode_metadata`,
`subtask`, `subgoal`). Defined in `src/robolabel/schema.py`. The schema has evolved additively:

- **v1** — base: episode metadata + subtask boundaries + subgoals.
- **v2** — adds `phase` (closed-vocabulary phase label, per subtask), `boundary_evidence` (one-line
  visual reason for the boundary, per subtask), and `strategy` (S0–S4 name, per episode). Null under
  the baseline S0.
- **v3** — adds `target` (per subtask): the grounded object/destination the subtask acts on
  ("red cube"), so a label reads **`phase → target`**. Required for every phase except `retract`.

The change is purely additive: **v1 and v2 files still read** (absent columns are treated as null).
The full column list is in `SCHEMA.md`. The `SubtaskSegment` dataclass:

```python
@dataclass
class SubtaskSegment:
    segment_idx: int
    start_frame: int
    end_frame: int
    subtask_text: str
    phase: str | None = None       # v2: closed-vocabulary phase (S2+)
    evidence: str | None = None    # v2: one-line visual evidence for the boundary (S1+)
    target: str | None = None      # v3: grounded object/destination (S2+); None for retract
```

Subtasks for an episode are **contiguous, non-overlapping, and cover `[0, num_frames-1]`** with
inclusive endpoints. Row order is deterministic; the only non-reproducible fields are absolute paths
(`subgoal_image_path`, `receipt_path`).

### 4.2 Gold file — schema `robolabel/gold/v1`

A single JSON object with an `episodes` list. Each episode carries an `auto` block (a frozen snapshot
of the VLM labels) and a `gold` block (what the human enters). `accept_auto` flags mean "the human
confirms the VLM value here." Key functions (`src/robolabel/gold.py`):

- `build_gold_template(annotations_dir)` — pre-fills `auto`, nulls out `gold`.
- `load_or_sync_gold(...)` — idempotent: creates the file if missing, otherwise **re-syncs the `auto`
  side** (cheap, from cache) while **preserving every human `gold` edit**.
- `update_episode_review(...)` — writes a human review into the `gold` block (quality clipped to
  [1,5], boundary/subgoal edits indexed and merged).

The separation is the whole point: the VLM can be re-run and the `auto` snapshot refreshed without ever
touching the human's corrections.

---

## 5. The Episode abstraction and adapters

### 5.1 `Episode` (`src/robolabel/episode.py`)

A dataclass wrapping one demonstration: `episode_id`, `num_frames`, `fps`, `task`, a **lazy**
`get_frame` callable returning `(H,W,3)` uint8 RGB, optional `actions`, `camera_key`, and an `extra`
bag. Frame access (`Episode.frame(i)`) clamps the index and coerces whatever the adapter returns
(float [0,1], CHW, RGBA, grayscale) into RGB uint8 via `_as_rgb_uint8` — so adapters can stay native
and coercion happens once, at access time.

### 5.2 Adapters (`src/robolabel/adapters/`)

- **`LeRobotAdapter`** — reads a LeRobot **v3.0** dataset (HF hub id or local path) through the
  `LeRobotDataset` API. Per-episode frame ranges come from `meta.episodes[ep]["dataset_from_index" /
  "dataset_to_index"]`; `fps` from `meta.fps`; a `--camera-key` selects one stream (defaults to the
  first). Optional dependency: `pip install 'robolabel[lerobot]'`.
- **`DirectoryAdapter`** — reads a folder of per-episode **frame directories** (`.png/.jpg/...`) or
  **videos** (`.mp4/.mov/...` via `imageio`/ffmpeg, lazily). Per-episode `task`/`fps` overrides come
  from an optional `episodes.jsonl`.
- **`build_source(kind, target, **kwargs)`** — the registry dispatch used by the CLI.

---

## 6. Providers (the VLM boundary)

### 6.1 The interface (`src/robolabel/providers/base.py`)

Every provider implements one method:

```python
def ask(self, frames, frame_labels, question, receipt_path, *,
        frame_captions=None, temperature=None) -> ProviderResponse
```

It tiles `frames` into a **contact sheet** (`make_contact_sheet`, a 3-column grid; captions optionally
stamp the frame index + timestamp), sends it with the text `question`, and **always writes a receipt**
to `receipt_path`. A receipt holds the request question, the raw response JSON/text, status, latency,
and (where reported) token counts — but **never image bytes**. A concrete helper, `observe_then_label`,
implements the two-stage pattern used throughout: ask an *observe* question, extract JSON from the
answer, build a *label* question from those observations, ask again.

Shared utilities: a provider **registry** (`build_provider(name, model)`, resolved from `--provider`
or `$ROBOVID_PROVIDER`), tolerant JSON extraction (`try_extract_json` survives ```json fences and
prose), `image_to_data_url`, and `load_secret(env_vars, label)` — which checks env vars then a local
`.env` file and raises a `MissingCredentialError` naming the exact variable to set.

### 6.2 Concrete providers

| provider | endpoint / model | credential | cost | notes |
|---|---|---|---|---|
| **gemini** | `generativelanguage…/v1beta/{model}:generateContent`, default `gemini-2.5-flash` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | hard-coded per-model price table | retries on 429/5xx (≤8, exp backoff cap 20s); **receipt-level caching** (reuses a prior 200 response for the same model) |
| **openai** | `/v1/responses` (multimodal), default `gpt-4o` | `OPENAI_API_KEY` | none (token counts only) | request/response shape written but not dogfic at scale |
| **qwen** | local `Qwen2.5-VL-7B-Instruct` | none (local) | always $0 | needs `[qwen]` extra + GPU |
| **mock** | none | none | $0 | returns *structurally valid, semantically meaningless* JSON; routes by prompt markers; used for CI, demo, tests |

The **default per-call read timeout is 120 s**; one fresh-dataset episode genuinely exceeded that and
was backfilled at 300 s (`scripts/backfill_ep1.py`) — a real, documented robustness edge, not a silent
drop.

---

## 7. The annotation-strategy layer (S0–S4)

This is the technical heart of the project: a configurable stack of techniques between the adapter and
the provider, designed to move the VLM off its characteristic failure modes. It is **off by default** —
S0 is the cheap, reproducible baseline.

### 7.1 `StrategyConfig` (`src/robolabel/strategy.py`)

| field | default | meaning |
|---|---|---|
| `frame_count` | 6 | evenly-spaced keyframes in the contact sheet |
| `resolution` | 224 | thumbnail width |
| `caption_timestamps` | False | stamp frame index + timestamp on each thumbnail |
| `grounded` | False | require per-segment `end_frame` + `evidence` (S1+) |
| `closed_vocabulary` | False | enforce the phase vocabulary; coerce unknowns to `other` (S2+) |
| `enforce_min_segments` | False | apply the granularity floor (S2+) |
| `min_granularity_policy` | `"warn"` | `warn` = accept a below-floor answer and flag it; `reject` = re-prompt |
| `require_target` | False | **v3**: every non-retract segment must name a grounded target (S2+) |
| `max_label_attempts` | 1 | re-prompt budget on a schema-validation failure |
| `refine_boundaries` | False | dense-window per-boundary refinement (S3+) |
| `refine_window` / `refine_max_frames` | 15 / 25 | refinement window and frame cap |
| `self_consistency_k` | 1 | label samples; median-combined (S4: k=3) |
| `temperature` | 0.4 | decoding temperature when k>1 |

### 7.2 The presets (cumulative)

| | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| frame_count | 6 | 12 | 12 | 12 | 12 |
| caption_timestamps | – | ✓ | ✓ | ✓ | ✓ |
| grounded (frame index + evidence) | – | ✓ | ✓ | ✓ | ✓ |
| closed_vocabulary | – | – | ✓ | ✓ | ✓ |
| enforce_min_segments | – | – | ✓ | ✓ | ✓ |
| **require_target** (v3) | – | – | ✓ | ✓ | ✓ |
| refine_boundaries | – | – | – | ✓ | ✓ |
| self_consistency_k | 1 | 1 | 1 | 1 | 3 |

`load_strategy` resolves a preset name (case-insensitive), or a JSON file (`{"base":"S2", …overrides}`),
or `None` → S0. `provenance()` is what gets written to `strategy.json` next to the parquet, so every run
records exactly the config that produced it. **"Grounded" = S2** is the shipped/recommended strategy.

### 7.3 Two-stage grounded segmentation (`src/robolabel/labelers/segmentation.py`)

For S1+, `segment_episode` runs **observe → label**:

1. **Observe.** Sample `frame_count` evenly-spaced frames; ask the model for *visible physical events
   only* (gripper close, approach, release, retract) as frame indices. Stored once, reused across all
   self-consistency samples.
2. **Label.** Feed the observations back and ask for segments. `validate_grounded_segments` enforces the
   schema: integer `end_frame`; an `evidence` string when grounded; a `phase` in the closed vocabulary
   (unknowns coerced to `other`) when `closed_vocabulary`; and a **`target`** when `require_target`
   (rejected if empty on any non-retract phase). A capped re-prompt (`max_label_attempts`) appends a
   corrective suffix on failure.

Two small but important normalizers were added with the structured-label patch:

- `_clean_target(value)` maps vague fillers (`none`, `n/a`, `the scene`, `object`, …) and parquet `NaN`
  to `None`, and trims to 80 chars — so "target present" means an actual named object.
- `_dedupe_trailing_phases(segs)` collapses consecutive identical **trailing** phases into one segment
  spanning to the last frame (keeping the earlier target). This is the fix for the graded "two retract
  steps" error.

**Self-consistency (S4).** `self_consistency_k` samples are combined per-boundary by *median* (not
per-segment majority vote): find the modal segment count, take the median `end_frame` at each boundary
position across matching samples, then enforce strict monotonicity.

**Boundary refinement (S3+).** For each internal boundary, extract a dense ±`refine_window` window
(capped at `refine_max_frames`), and ask the model for the exact transition frame, clamped to stay
ordered between neighbours.

**Min-granularity policy.** The S2+ "reject a segmentation below the floor" rule is a *policy*, not a
hard rule. It defaults to `warn` (accept the below-floor answer, flag it `single_segment_candidate`)
because the human gold for at least one episode (SO-101 ep7) is a genuine single continuous segment that
a hard floor could never match. Set `reject` to reproduce the ablation's original numbers exactly.

### 7.4 Free baselines (honest floors)

- **`S_grip`** (`labelers/gripper_baseline.py`) — a *zero-API* proprioceptive segmenter: normalize the
  gripper channel, detect open/close transitions (debounced), find end-effector speed pauses before each
  gripper event, assign canonical phases by order. Thresholds live in `rubric.yaml`.
- **uniform-fifths** — five equal segments with canonical phases. The trivial "did the model beat
  arithmetic?" floor.

Both are scored against gold with the *same* reliability code as the VLM strategies.

### 7.5 The rubric (`rubric.yaml` + `rubric.py`)

Every prompt, the phase vocabulary `[approach, grasp, transport, release-place, retract, other]`, the
min/max segment counts, the quality scale, the gate thresholds, and the gripper-baseline constants are
**config, not code** — the Python contains no scoring or vocabulary logic. The closed vocabulary is
*hand-authored*, not derived from gold (confirmed: the gold/reliability code never touches `phase`),
which is what gives the labels their cross-dataset objectivity. Prompt templates are filled with manual
key substitution (not `str.format`) so literal JSON braces in the examples survive.

---

## 8. The measurement subsystem

### 8.1 Metrics (`src/robolabel/metrics.py`)

- **Temporal IoU** — inclusive-frame intersection-over-union of two segments: `inter / union` where
  `inter = max(0, min(a_end,g_end) − max(a_start,g_start) + 1)`. `None` if a boundary is missing.
- **Episode IoU** — index-aligned per-segment IoU, averaged over the episode (the reliability report's
  headline definition).
- **Boundaries** — the internal transition frames: the `end_frame` of every segment *except the last*.
- **Boundary precision / recall / MAE @ ±5** (`boundary_pr_mae`) — greedily match each predicted
  boundary to the nearest unused gold boundary within `tol=5` frames; precision = matched / predicted,
  recall = matched / gold, MAE = mean frame error on matches. **This is the metric the project argues
  actually matters** for subtask conditioning (landing on the real transition frame), as opposed to mean
  overlap.

### 8.2 Reliability (`src/robolabel/reliability.py`)

`reliability_report(gold_path)` compares `auto` vs `gold` per reviewed episode and aggregates:

- subtask boundary temporal IoU (mean),
- quality **exact** and **within-one** agreement (over episodes where both scores exist),
- subgoal frame agreement.

An episode counts as "reviewed" if it has *any* human edit. `accept_auto=true` on a subtask copies the
auto boundaries into gold before the IoU is computed.

### 8.3 The gate (`src/robolabel/gate.py`)

`run_gate` is advisory: it **flags failure modes but never drops episodes** — and reports
`dropped_episode_count` (always 0) to make that auditable. Detectors:

- **degenerate single segment** — `len(subtasks) <= 1`.
- **uniform split** — coefficient of variation of segment lengths `< 0.12` (with ≥3 segments): boundaries
  placed at near-arithmetic fractions.
- **quality-outlier `needs_review`** — a score ≥2 points below the dataset median.
- plus repeated subtask text, missing/vague target grounding, score↔reason contradiction, and a
  collapsed-quality-distribution flag.

### 8.4 Evidence factual-accuracy (the metric no other tool reports)

For each grounded boundary, the model emits a one-line *evidence* string ("the red cube lifts off the
table"). The blind trial asks a human, frame in hand: **is that statement actually true of the cited
frame?** The fraction that are true is *evidence factual-accuracy*. It is the difference between "the
boundary happened to land right" and "the model knew why." It is computed by `trial_report.tally` over
the blind-grading denominators (only non-final segments that carry evidence).

### 8.5 The frozen evaluation (`scripts/`)

- **`eval_strategies.py`** runs the full ablation: every (model × strategy) cell, scored against gold,
  with per-episode retry and receipt-based cost accounting; metadata (quality) is labeled once per
  (model, episode) and reused across strategies.
- **`compute_metrics.py`** is the **zero-API** second pass: it monkeypatches `gemini.requests.post` to
  raise, then reconstructs every cell's segments from cached receipts and adds the per-band IoU
  breakdown, the uniform-fifths baseline, and the boundary P/R/MAE numbers. Reconstruction pins
  `min_granularity_policy="reject"` and `require_target=False` so it stays faithful to the originally
  reported run (and so pre-v3 receipts reconstruct).
- **`score_gripper.py`** scores `S_grip` the same way.
- **`eval/so101_split.json`** is the frozen 30-tune / 20-test split (seeded). Strategies are iterated on
  tune; the chosen strategy is scored once on test.

---

## 9. The consumption / acceptance-review layer

This is the half of the project a reviewer touches, and it is unusually large for a labeling tool —
deliberately, because "trust the labels" is the thing being argued.

### 9.1 CLI (`src/robolabel/cli.py`)

`annotate`, `review`, `inspect`, `query`, `trial-report`, `reliability`, `gate`, `export`, `cost`,
`demo`. Each is a thin handler over a library function.

### 9.2 `robolabel inspect` — the verification viewer (`inspect_server.py`)

A dependency-free `http.server` SPA. Left: a queue sortable by worst-IoU / most-flags / id, filterable to
gate-flagged or ungraded. Center: the video frame with a **multi-track boundary timeline** — gold plus
every strategy/baseline on parallel color-coded lanes, segments labeled `phase → target` (via the
`segLabel` helper), a playhead, click-to-seek. Right, three tabs:

- **Metrics** — per-track IoU, boundary precision/recall@±5, MAE, segment count, gate flags, quality,
  cost.
- **Evidence** — each evidence string next to a thumbnail of the exact frame it cites, with true/false
  judge buttons. This is how evidence factual-accuracy is graded.
- **Grade** (blind mode only) — per boundary: within ±5? phase correct? evidence true? plus an overall
  *usable / touch-up / garbage* verdict, saved to a grades JSON.

Frames are served from a thread-safe LRU cache with background prefetch; the whole thing reconstructs
from cached receipts with no network.

### 9.3 `robolabel review` — the calibration GUI (`review_server.py`)

The human-gold authoring session: scrub the video, accept or correct each auto boundary/subgoal, set the
1–5 quality score, write a reason. Every save goes through `update_episode_review` into the gold file's
`gold` block — **the `auto` block is never touched**. Live reliability stats sit in the header.

### 9.4 `robolabel query` — the usefulness path (`query.py`)

`--phase grasp` retrieves every segment with that phase across the dataset and tiles their representative
frames into a contact-sheet PNG (captioned `ep{id} f{frame} → {target}`). `--needs-review` lists the
gate's quality outliers, worst first. The visceral "the labels mean something" proof.

### 9.5 `robolabel trial-report` — the blind tally (`trial_report.py`)

Tallies a blind-grading session into a markdown table. Two protocols:

- **`mark-failures-only`** (default) — the grader marked only the failures; every *unmarked* boundary /
  phase / evidence slot counts as a pass over the **known denominator** (carried in the unblind map). This
  matches how a human actually grades and is a first-class feature, not a workaround.
- **`mark-all`** — rate = mean of the marks entered.

Per strategy it reports boundary acceptance, phase accuracy, evidence factual-accuracy, failure-band
rate, and the usable/touch-up/garbage distribution.

### 9.6 Export and consumability

`export --format jsonl` is the portable, full-fidelity dump. `export --format lerobot` writes into the
**pinned LeRobot 0.4.x subtask convention** (verified against the installed source, not guessed):

- `meta/subtasks.parquet` — a string-indexed vocabulary table mirroring `meta/tasks.parquet`, so
  `LeRobotDataset` resolves a frame's subtask via `meta.subtasks.iloc[subtask_index].name`.
- `meta/episodes_subtasks.parquet` — per-episode `subtask_indices / names / start_frames / end_frames /
  start_times / end_times` (SARM-compatible).

**What survives export:** the temporal boundaries and the subtask phrase. **What stays sidecar-only:**
`boundary_evidence`, `phase`, the episode quality fields, receipts, and `cost_usd` — the convention has
no slot for them. `scripts/consumability_check.py` runs the full annotate → export → reload → resolve
chain and asserts every frame's `subtask_index` resolves to the segment it falls in (all 72 frames of a
3-episode synthetic set pass; same assertion in CI). The honest caveat: this is a non-destructive
*metadata overlay*; the per-frame `subtask_index` column is **not** written into the binary `data/`
parquet, so a full SARM dataloader would still need that materialization step.

---

## 10. What has actually been measured (the evidence)

### 10.1 SO-101 ablation (`STRATEGY_REPORT.md`)

Dataset `lerobot/svla_so101_pickplace`, 50 episodes, 30 tune / 20 held-out test, one annotator's gold
*built by correcting S0 drafts* (so the gold is S0-anchored — a quiet home-field advantage for the
baseline). Three failure bands motivate the whole strategy stack: **(a) degenerate** single blob,
**(b) uniform split**, **(c) drifted** (plausible but systematically off).

Key results:

- **Failure tail eliminated.** Tune: S0-Flash leaves 3 degenerate + 9 uniform; frame grounding (S1)
  empties the uniform band (9 → 0); S2 holds degenerate at 0. Held-out test: S0-Flash **5/20** in a
  failure band, grounded **0/20**. This is the most robust result.
- **Mean IoU did *not* improve** (the headline negative): chosen **Pro-S2 0.444** vs **S0-Flash 0.460**
  on test. The +0.05 tune edge (selected by a margin of **0.003**) was overfit.
- **But boundary placement improved where it matters:** grounded lands **36% more** gold boundaries
  within ±5 frames (recall **0.307 vs 0.226**). Mean overlap and exact-transition recall *disagree*, and
  the project argues the latter is what conditioning needs.
- **Quality metric is near-degenerate** (gold is 49/50 "score 5", so a constant-5 baseline scores
  0.97–1.00). The meaningful signal is the catastrophic false-negative rate: Flash **3/30** → Pro
  **1/30**.
- **Free baselines lose:** S_grip 0.18–0.20 IoU, uniform-fifths ~0.36; VLM cells sit ~0.10 above uniform
  and ~0.25 above S_grip.
- Total ablation spend: **$16.54 / $30** ceiling, tracked from on-disk receipts.

The report also keeps an honest counter-example (ep7, a genuine single-segment episode the min-granularity
floor *cannot* match) and documents a real bug it hit (a cached-receipt preflight made the cost projection
read $0; true spend was always correct).

### 10.2 Generalization, second dataset (`FRESH_TRIAL_REPORT.md`)

Dataset `lerobot/svla_so100_stacking` (Apache-2.0, never used to build any number, no S0-anchored gold),
task "put the red cube on the blue cube." Both strategies on the same 20 episodes:

- **Objective, gold-free:** failure-band rate **grounded 0/20 vs S0 8/20 (40%)**. So failure-band
  elimination is now verified on **two** datasets, paired (SO-101 5/20→0; fresh 8/20→0). The grounded
  evidence referenced the *new* scene, not parroted lego/brick — it is grounding to the actual video.
- **Blind grading (author, all 40 items, mark-failures-only):** grounded boundary acceptance **0.86 vs
  0.66**, evidence factual-accuracy **0.98**, verdict **14 usable vs 4** — but grounded **lost on phase
  accuracy (0.80 vs 0.89)**. That loss is precisely the underspecification flaw: "approach" doesn't say
  *which* cube.
- **The structured-label fix (schema v3):** the 20 episodes were re-annotated with the patched grounded
  strategy. Objective re-check: failure-band **still 0/20**, and **79/79 (100%)** non-retract segments now
  carry a grounded target (`approach → red cube`, `transport → blue cube`). The format fix is verified by
  tests and the live run; the phase-accuracy *re-grade* is the one open item.

### 10.3 The claims audit (`CLAIMS.md`)

Fifteen rows, each mapped to an artifact and a status. Verified outright: drafts-for-any-VLM, measures-
against-gold, gate-never-drops, LeRobot export resolves. Verified-on-one-dataset: failure-band
elimination, boundary-placement win, the mean-IoU *non*-improvement (a negative result, stated as such),
the Pro quality-FN reduction, the near-degenerate quality metric, S_grip losing. Explicitly *not* shown:
that a full SARM dataloader trains off the export (mechanical-only), and that the annotations **improve
downstream training** — row 14 is `untested (explicitly; on one careful test, negative)`. Row 15 (the
structured `phase → target` labels) is `fixed-and-spot-checked-pending`.

---

## 11. Engineering practices worth calling out

- **Resilience + resume.** `annotate_source` checkpoints the parquet after every episode and skips
  episodes already present on a re-run, so a transient provider error (429, timeout, one bad answer)
  loses at most the in-flight episode. This was exercised for real twice on the fresh dataset (a 429
  mid-run kept 20/22; a repeated 120 s timeout on one episode was backfilled). Tested by
  `test_resilience.py`.
- **Zero-API reconstruction.** Every analysis path (`compute_metrics`, `build_inspect_data`, the viewers)
  can rebuild segments from cached receipts with the network monkeypatched to raise — a hard offline
  guarantee, and the reason the ablation numbers are reproducible without spending a cent.
- **Receipts without secrets.** Receipts hold the prompt, response, status, latency, and token counts but
  never image bytes; the API key is never echoed into a receipt, log, or report. `.env` is git-ignored;
  every commit in this line passed a secret scan. Frozen eval artifacts (gold, splits, the report numbers)
  are treated as read-only.
- **Config over code.** Prompts, vocabulary, thresholds, prices — all in `rubric.yaml` / the strategy
  presets, so behavior changes are data changes with provenance (`strategy.json`).
- **Tests + packaging.** The current suite is **99 tests** passing (ruff clean), spanning schema,
  metrics, providers, adapters, gate, the S0–S4 strategy layer, resilience, export round-trip,
  calibration, and the blind-trial tally. Packaged as `robolabel` (Apache-2.0, Py≥3.10, hatchling) with
  optional `[lerobot]` / `[qwen]` / `[dev]` extras and a single `robolabel` console entry point.

---

## 12. Honest limitations (stated, not hidden)

- **Training utility is not demonstrated** — and on one careful, preregistered JEPA/LeWM head-to-head
  (`docs/why.md`) the apparent conditioning win was a latent-variance artifact that vanished under
  normalization; the one-step latent task was near-degenerate. Do not treat these annotations as a known
  training win.
- **One task family, one annotator's gold.** Everything `verified-on-one-dataset` carries that caveat; the
  fresh set extends only the *objective* failure-band result and (subjectively) the blind trial.
- **S0-anchored gold** gives the baseline a quiet boundary-metric advantage (which makes the grounding
  wins *stronger*, not weaker).
- **Provider coverage is uneven.** Gemini is dogfooded at ablation scale; the OpenAI and Qwen paths are
  written to shape but not exercised live. mp4 directory input is written but unexercised. Cost estimates
  are Gemini-specific.
- **Export is an overlay, not in-place mutation.** See §9.6.
- **One camera.** Labels come from a single view; multi-view reasoning is out of scope.

---

## 13. The structured-label patch, in context (this line of work)

The most recent change closes the one subjective gap the blind trial surfaced. It is a **label-format**
change, scoped tightly (no open-vocab work, no new strategies, no changes to the frozen ablation numbers,
S0 untouched):

1. **Schema v3** — additive `target` column; v1/v2 still read.
2. **Required, grounded target** on S2+ (`require_target`), with `none` allowed only for `retract`;
   validation rejects an empty target with the capped re-prompt; vague fillers normalized to `None`.
3. **Terminal-phase dedupe** — the "two retract steps" fix.
4. **`phase → target` everywhere** — inspect timeline/evidence/grade panels, query captions, the
   inspect-data payload.
5. **Reconstruction pinned** (`require_target=False`) so pre-v3 receipts and the frozen SO-101 numbers are
   untouched.
6. **Re-annotation** of the 20 fresh episodes: 20/20 (one backfilled past the 120 s timeout), failure-band
   still 0/20, 79/79 targets.
7. **Docs** — CHANGELOG (crediting the blind-grading session: "the acceptance kit caught this
   pre-launch"), CLAIMS row 15 (`fixed-and-spot-checked-pending`), README example, SCHEMA v3,
   FRESH_TRIAL_REPORT note.

The one remaining step is the author's spot-check of the re-graded phase accuracy from
`inspect_data/fresh_v3.json`, after which CLAIMS row 15 can move off `pending`.

---

## 14. File map (where to look)

| area | files |
|---|---|
| data model | `src/robolabel/schema.py`, `SCHEMA.md`, `src/robolabel/gold.py` |
| episode / adapters | `src/robolabel/episode.py`, `src/robolabel/adapters/` |
| providers | `src/robolabel/providers/{base,gemini,openai,qwen,mock}.py` |
| strategy layer | `src/robolabel/strategy.py`, `src/robolabel/labelers/segmentation.py`, `labelers/gripper_baseline.py`, `src/robolabel/rubric.py`, `rubric.yaml` |
| measurement | `src/robolabel/metrics.py`, `reliability.py`, `gate.py` |
| consumption | `src/robolabel/cli.py`, `inspect_server.py`, `inspect_data.py`, `review_server.py`, `query.py`, `trial_report.py`, export code |
| evaluation | `scripts/eval_strategies.py`, `compute_metrics.py`, `score_gripper.py`, `build_inspect_data.py`, `eval/so101_split.json` |
| evidence | `STRATEGY_REPORT.md`, `FRESH_TRIAL_REPORT.md`, `CLAIMS.md`, `docs/why.md`, `docs/consumability.md`, `REVIEW_GUIDE.md`, `RELEASE_READINESS.md` |
| tests / packaging | `tests/` (99 tests), `pyproject.toml` |

---

*Bottom line: robolabel is a VLM annotation tool whose real contribution is not the labels but the
**measurement discipline** around them — a grounded strategy layer that demonstrably eliminates the
catastrophic failure bands, a metric (boundary placement, evidence factual-accuracy) chosen because it
matters for conditioning rather than because it flatters the tool, and a claims/evidence apparatus that
reports the negative results as loudly as the positive ones.*
