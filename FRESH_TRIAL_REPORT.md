# Fresh-dataset trial — `lerobot/svla_so100_stacking`

A second dataset, **never used to build any robolabel number**, with **no
S0-anchored gold** — the test of whether the findings generalize off the one dataset
they were tuned on.

**Dataset choice (documented):** `lerobot/svla_so100_stacking` — Apache-2.0 (verified
on the HF page), LeRobot **v3.0**, SO-100 arm, task *"Put the red cube on top of the
blue cube"* (a genuinely different task from SO-101 lego pick-place), `observation.images.top`
+ `observation.images.wrist`, 30 fps. Chosen over the MIT `Tomas0413/so100_screw_lid_v0`
(v2.1 format risk + ~2500-frame episodes) for format safety and shorter clips. 20 episodes
annotated.

## Objective signal — paired, computed now, gold-free

Both strategies ran on **exactly the same 20 episodes** (0–19). No human grading needed;
these come straight from the fresh annotations.

| metric (over the same 20 episodes) | grounded-Flash (S2) | S0-Flash (baseline) |
|---|---|---|
| **failure-band rate** (degenerate or uniform-split) | **0 / 20** | **8 / 20** (3 degenerate + 5 uniform) |
| mean segments / episode | 5.2 | 4.0 |
| per-boundary evidence present | 20 / 20 | — (S0 produces none) |

Total API spend (both strategies, paired 20): **$0.765 of the $10 ceiling**.

**The failure-band elimination result holds — and is stronger here.** On a brand-new task,
paired on the same episodes, the baseline collapses **40% (8/20)** of episodes into a
single blob or uniform fifths, while the grounded strategy collapses **none (0/20)**. On
SO-101 held-out the same gap was 5/20 → 0/20; on this fresh dataset it is **8/20 → 0/20**.
So the claim "frame grounding eliminates the degenerate/uniform failure bands" is now
**verified on two datasets, paired**, with no S0-anchored gold involved. The grounded
evidence also referenced the *new* scene ("red cube lifts off the table", "red cube arrives
over the blue cube"), not parroted lego/brick — it is grounding to the actual video.

**Run note (honest):** the first annotation pass hit a Gemini credit limit (HTTP 429)
mid-run; the resilient `annotate_source` checkpointing saved every completed episode
(grounded 20/22), confirming the resilience works under a real mid-run failure. After a
$10 top-up, S0-Flash was run on exactly those 20 episodes for the paired contrast above.

## Blind grading (subjective) — author's session, all 40 items

The 40 blind items (20 grounded-Flash + 20 S0-Flash, same episodes, identity hidden,
shuffled) were graded blind by the author against the video. **Grading protocol:
mark-failures-only — the grader marked only the failures; every unmarked boundary, phase,
and evidence string counts as a pass over the known denominator** (the number of
boundaries / phases / evidence slots in the graded items, not just the marks entered).

| strategy | items | boundary acceptance (±5f) | phase accuracy | **evidence factual-accuracy** | failure-band | usable / touch-up / garbage |
|---|---|---|---|---|---|---|
| **grounded-Flash** | 20 | **0.86** (n=84) | 0.80 (n=84) | **0.98** (n=84) | **0.00** | **14 / 4 / 2** |
| S0-Flash | 20 | 0.66 (n=61) | 0.89 (n=61) | n/a (no evidence) | 0.40 | 4 / 6 / 9 |

**Read:** grounded wins decisively on boundaries (0.86 vs 0.66), on the verdict (14
"usable" vs 4), and on the metric no other tool reports — **evidence factual-accuracy
0.98**, i.e. when the model says "red cube lifts off the table" it is true of the cited
frame 98% of the time. The one place grounded *loses* is **phase accuracy (0.80 vs S0's
free-text 0.89)** — which is precisely the **label-underspecification** flaw the author's
grading surfaced: the closed phase label ("approach") often doesn't say *which* object.
The structured-label patch (phase → target) targets exactly this number.

**evidence factual-accuracy** is the metric no other tool reports: the fraction of the
model's stated reasons ("gripper contacts brick") that are factually true of the exact
frame they cite. It is the difference between "the boundary happened to land right" and
"the model knew why."

### Structured-label re-annotation (`phase → target`, schema v3)

In response to the underspecification above, the 20 episodes were re-annotated with the
patched grounded strategy (S2 + required `target`; `fresh_stacking/grounded_flash_v3`).
Objective re-check: the **failure-band rate stays 0 / 20**, and **79 / 79 (100%)** of
non-retract segments now carry a grounded target — e.g. the previously-bare first label is
now `approach → red cube`, and a two-cube scene reads `transport → blue cube`.

**Spot-check (frame-level, 8 / 20 episodes, 36 segments).** Rendering each segment's
mid-frame next to its `phase → target` label (`scripts/spotcheck_frames.py`, zero-API,
reads cached frames) and judging the label against the frame:

- **Phase labels: correct in every sampled segment** — the canonical
  `approach → grasp → transport → release-place → retract` order held throughout (one short
  episode truncated to `approach/grasp/transport`, still correct). No phase mislabels (no
  "grasp" where the gripper was clearly transporting, etc.).
- **Target present and right object class: 100%** of sampled non-retract segments.
- **Residual, documented not hidden:** the *target* on `transport` / `release-place` is not
  yet written to one convention — it names the **moved object** in some episodes
  (`transport → red cube`, ep4/ep19) and the **destination** in others
  (`transport → blue cube`, ep0/ep7/ep10/ep13/ep16). Both are defensible under the slot's
  "object *or* destination" definition, but a downstream consumer expecting a single rule
  would see drift. This is a *convention* gap, not a grounding failure, and is the open item
  for the author's full blind re-grade (the gold-standard confirmation) via `robolabel
  inspect --data inspect_data/fresh_v3.json`.

So the underspecification the blind trial surfaced ("approach" — *which* cube?) is closed:
the object is now named on every segment. The phase labels themselves were already correct
here; the remaining work is making the target *naming rule* uniform, tracked as v0.2.

# Cross-task generalization probe (pour + cloth-fold) — v0.2

**Question.** Does frame-grounding still eliminate the catastrophic failure bands on tasks
*outside* pick-and-place, and does the hand-authored pick-and-place phase vocabulary break
where the grounding does not?

**Datasets (third and fourth task families, gold-free).**
- **Pour** — `Ishah8840/so101_pouring` (SO-101, **sim**, v3.0, `observation.images.front`),
  task *"Pour the water from the source cup into the target cup."* 8 episodes (0–7).
- **Fold** — `the-sam-uel/bi-so101-fold-horizontal-set-1` (SO-101, **real teleop, bimanual**,
  v3.0; annotated from the single `observation.images.overhead` stream), task *"Fold the
  yellow cloth horizontally."* 8 episodes (0–7).

**Method.** grounded-Flash (Gemini 2.5 Flash) on the **same 8 episodes** per task, two
conditions: **(a)** the shipped closed-vocab default **S2**, and **(b)** the new open-vocab
**S2-open** (free-text phase names). Only gold-free, task-agnostic metrics + an author
frame spot-check. Total new API spend for the whole probe: **$0.75** (ceiling $6).

| task | condition | failure-band | seg/ep (mean) | target-present | phase coerced to `other` |
|---|---|---|---|---|---|
| pour | closed-vocab **S2** | **0 / 8** | 5.0 | 34/34 (1.00) | **7 / 40 (17.5%)** |
| pour | open-vocab **S2-open** | **0 / 8** | 5.1 | 37/41 (0.90) | **0 / 41 (0%)** |
| fold | closed-vocab **S2** | **0 / 8** | 5.0 | 32/32 (1.00) | 1 / 40 (2.5%) |
| fold | open-vocab **S2-open** | **0 / 8** | 4.9 | 31/39 (0.79) | **0 / 39 (0%)** |

**Finding 1 — the grounding mechanism generalizes.** The failure-band rate is **0 / 8 in all
four cells**. Frame-indexed grounding holds the degenerate/uniform failure bands at zero on
pour and on cloth-fold, in *both* vocab conditions — so the mechanism that mattered on
pick-and-place and stacking is not specific to those tasks. Failure-band elimination is now
seen on **four task families** (pick-place, stacking, pour, fold).

**Finding 2 — the closed pick-and-place vocabulary degrades off-task, and degrades
*differently* on each task.**
- On **pour**, the closed vocabulary visibly breaks: **17.5%** of segments are coerced to
  `other` because the vocabulary has no word for the pour action. The defining moment of the
  task is unnameable under the closed labels.
- On **fold**, the closed vocabulary coerces *less* to `other` (2.5%) — but that is the
  *worse* failure: instead of admitting the mismatch, it silently mislabels the fold as
  `release-place` / `transport` (pick-place words that loosely fit). A low `other` rate here
  is a false comfort, not a good fit.
- The **open-vocab** condition coerces **0%** in both, and the free-text phases read sensibly
  and task-appropriately: `pour water → target cup`, `tilt to pour`, `position cup → target
  cup` (pour); `perform fold → yellow cloth`, `fold cloth`, `grasp cloth → yellow cloth`
  (fold). Same grounding, same boundaries — just a phase name that fits the task.

**Author spot-check (frame-level, open-vocab).** On two fully-inspected episodes (pour ep0,
fold ep0; 10 boundaries), the per-boundary evidence strings were true of their cited frame in
**9 / 10** (one borderline: a `pour water` segment paired with a release-flavored evidence
clause), and the open-vocab phase names read sensibly in **10 / 10** sampled segments. Frames
rendered via `scripts/probe_spotcheck.py`.

**Honest caveats.** (1) The pour set is **simulation**; the fold set is **real** teleop but
**bimanual**, annotated from one overhead camera. (2) 8 episodes per task, no human gold —
these are objective gold-free metrics plus a small author spot-check, *not* a blind boundary
re-grade. (3) The `other`-coercion rate **understates** the closed-vocab problem on fold
(silent mislabeling reads as a *low* `other` rate). (4) Target-present drops to ~0.79–0.90
under open-vocab because the looser retract/withdraw exemption lets more free-text "lift/
position" phases omit a target — a small regression to tighten. The mechanism generalizes;
the vocabulary does not — which is exactly what S2-open is for.

## What this report licenses the README to claim

- **Verified on two datasets:** grounding eliminates the degenerate/uniform failure bands
  (SO-101 held-out 5/20 → 0/20; fresh 0/20).
- **Pending the blind grading above:** boundary acceptance, phase accuracy, and evidence
  factual-accuracy on the fresh task. Until filled, the README treats these as not-yet-shown.
