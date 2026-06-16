# Price & efficiency

Everything learned about cost. Numbers are sourced to on-disk receipts (the per-call token
counts and prices the run actually used) where possible; projections are labelled as such.

## Pricing actually used

The Gemini price table in `providers/gemini.py` (USD per 1M tokens):

| model | input | output | vs 2.5-flash |
|---|---|---|---|
| **gemini-2.5-flash** (default) | $0.30 | $2.50 | 1× |
| gemini-2.5-pro (≤200k ctx) | $1.25 | $10.00 | ~4× |
| gemini-2.5-flash-lite | $0.10 | $0.40 | **~3× cheaper** (input ⅓, output ~⅙) — **untested for grounded; recommend validating** |

Cost is computed per call from the provider-reported token counts and written into each
receipt, then summed; it is an estimate (prices drift), audited via the receipts.

## Measured cost per episode (receipts)

Grounded open-vocab, **Gemini 2.5 Flash**, from this project's runs:

| run | modules | eps | $/episode |
|---|---|---|---|
| pick-place minimal | segmentation + quality | 5 | ~$0.022 |
| pour minimal | segmentation + quality | 5 | ~$0.028 |
| fold minimal | segmentation + quality | 5 | ~$0.020 |
| pour **everything-on** | + speed/subgoals/control/novelty/curation | 5 | ~$0.026 |
| pick-place **everything-on** | all modules | 8 | ~$0.017 |
| fold **everything-on** | all modules | 8 | ~$0.021 |

**Key finding — the full stack is ~free over the minimal run.** Every module beyond
segmentation + quality (`speed`, `subgoals`, `control`, `novelty`, `curation`) is
**deterministic, no VLM**, so an everything-on run costs essentially the same as a minimal one
(~$0.02–0.03/ep). The only paid work is the two VLM modules.

From the frozen ablation (`STRATEGY_REPORT.md`, $16.54/$30 total, receipts): Flash grounded cells
~$0.018/ep; **Pro ~$0.068/ep (~4×)**; the S3 dense-window refinement pass adds ~$0.008/ep
(Flash S3 ≈ $0.026); S4 self-consistency (k=3) ≈ $0.040/ep.

## Flash vs Pro

On the held-out test, **Pro's 4× cost bought only better *quality judgment*, not better
*boundary placement*** (`STRATEGY_REPORT.md`): Pro cut catastrophic quality false-negatives 3→1,
but did **not** improve mean boundary IoU (Pro-S2 0.444 vs Flash-S0 0.460) or boundary recall. So
Pro is worth it only if episode-quality discrimination is the bottleneck; for the conditioning
boundaries, **Flash is the right default.**

## Recommended config

**Grounded, open-vocab, Gemini 2.5 Flash** (`segmentation: {strategy: grounded, vocabulary: open}`
+ `quality`). Enable the deterministic modules freely (they cost nothing). Reserve Pro for a
quality-discrimination pass on variable-quality corpora; reserve S3/S4 for when grasp/release
boundary precision is the bottleneck (see the grasp/release refinement note in
`FRESH_TRIAL_REPORT.md`).

## Scale levers

- **Batch API** — Gemini's asynchronous batch tier is ~**50% off** synchronous. Annotation is
  embarrassingly batchable (one independent call per episode), so a scale run should use it.
- **Context caching** — the rubric/prompt prefix (the long instruction block) is identical across
  every call; caching it is ~**90% off** on the cached prefix tokens. The prefix is a large share
  of the input, so this is a material total reduction on a big run.
- Both compound with Flash-Lite if it validates.

## Projected cost per 1,000 episodes

Projections from the measured ~$0.02–0.03/ep (Flash, grounded), **not** a 1k receipt run:

| configuration | est. $/1,000 eps |
|---|---|
| **Minimal** (segmentation + quality), Flash, sync | **~$20–28** |
| **Full-stack** (all modules), Flash, sync | **~$20–28** (deterministic modules add ~$0) |
| Full-stack + S3 contact-refinement, Flash | ~$28–36 |
| Full-stack, **Pro** | ~$80–110 (~4×) |
| Minimal, **Flash-Lite** (untested) | ~$7–10 |
| Minimal, Flash, **batch API (−50%)** | ~$10–14 |
| Minimal, Flash, batch + context caching | **~$6–10** |

So a full conditioning + curation pass over a 1,000-episode dataset is **tens of dollars**, not
hundreds — and the curation/metadata that lets you then train on a high-value subset is free on
top of the segmentation you were already paying for.
