# Outreach collateral — DRAFTS ONLY (nothing here has been posted; do not post without review)

Everything in this file is a **draft for the author to review and post manually**. No
account is connected; nothing has been sent. Honest voice, no marketing. Links resolve only
after the repo is public.

---

## 1. X / Twitter thread (≤ 6 posts)

**1/**
robolabel: VLM-drafted subtask / quality / subgoal labels for LeRobot datasets — then
*measured* against human gold instead of trusted. The headline result is a negative one I
kept anyway: a "smarter" prompt did **not** beat the baseline on mean boundary IoU. 🧵

**2/**
What the grounding layer *did* do, on a 20-episode held-out test: it erased the catastrophic
failure modes — single "do the task" blobs and boundaries placed at uniform fifths — **5/20 →
0/20** — and landed **36% more** transitions within ±5 frames (recall .307 vs .226). Mean IoU
was the wrong number to optimize.

**3/**
Does that hold off pick-and-place? Same gold-free check on stacking, then on **pour** and
**cloth-folding**: frame-grounding keeps the failure-band at **0/8 on both**. But the
hand-authored pick-place vocabulary breaks — on pour, 17.5% of segments have no word and fall
to "other." The open-vocab variant names them right: "pour water," "perform fold."

**4/**
What it is **not**: a training win. In one preregistered, controlled test, using a signal
like this as world-model conditioning was a latent-variance artifact that vanished under
normalization. These labels are for VLA subtask conditioning + dataset curation. Honest
writeup: <LeWM / why.md link>

**5/**
The measurement *is* the product: every public claim maps to an evidence artifact + a status
(verified / one-dataset / untested). Per-call cost and receipts. A blind-grading viewer that
puts each evidence string next to the frame it cites. Zero-API reconstruction of every number.

**6/**
My actual view: data *interfaces* — not model size — are the near-term bottleneck for robot
learning. A label you can't audit is worse than no label. Repo + full ablation writeup:
<repo link>

---

## 2. Repo description + topics

**One-line "About":**
> VLM-drafted subtask, quality & subgoal annotations for LeRobot datasets — measured against
> human gold, not trusted (grounded strategy ablation + reliability report + LeRobot export).

*(GitHub limits the About blurb to ~350 chars; the line above is ~180.)*

**Topics:**
`lerobot` · `robot-learning` · `imitation-learning` · `vla` · `vlm` · `dataset-annotation` ·
`data-curation` · `subtask-segmentation` · `video-annotation` · `huggingface` · `robotics` ·
`annotation-reliability`

---

## 3. Issue replies (finalized — verified targets; post only after the repo is public)

Norm in the repo: short, friendly, lead with thanks, link a concrete command/PR, offer to
help. ~2–3 sentences. Targets are real open/closed issues; engage the closest real thread
honestly rather than inventing a request.

### (a) Subtask annotation — [huggingface/lerobot#3407](https://github.com/huggingface/lerobot/issues/3407) (open)
> The friction here is that lerobot carries two subtask shapes — the native
> `meta/subtasks.parquet` + per-frame `subtask_index`, vs. the SARM script's per-episode
> `subtask_*` columns — and they don't line up. While building an exporter I mapped both
> field-for-field: `robolabel export --format lerobot` writes the `meta/subtasks.parquet`
> convention and reconstructs the per-frame `subtask_index`, round-trip-tested through
> lerobot's own `load_subtasks`. Happy to share the mapping notes if they'd help reconcile the
> two paths.

### (b) Task-metadata editing — [huggingface/lerobot#2326](https://github.com/huggingface/lerobot/issues/2326) (open, `enhancement`)
> For the "edit/extend a task description after the fact" item: I keep per-episode annotations
> (subtask boundaries + quality) in a sidecar and round-trip the subtask half into the
> `meta/subtasks.parquet` convention, so most of the read/edit/write plumbing already exists in
> `robolabel`. Glad to contribute a focused task-metadata `--writeback` or compare notes if it
> helps move #2326.

### (c) Quality filtering / curation — [huggingface/lerobot#630](https://github.com/huggingface/lerobot/issues/630) (closed; `remove_episodes` resolved deletion)
> `remove_episodes` covers the deletion; the harder half is deciding *which* episodes.
> `robolabel` scores per-episode quality with a VLM and flags low-score outliers as
> `needs_review` (it never auto-drops), all measured against a human gold set
> (`robolabel gate` / `robolabel reliability`). Happy to share that curation flow if the "which
> episodes?" question comes up again.

---

*See `docs/launch_checklist.md` for the PyPI/publish steps and the LeRobot Discord post draft.*
