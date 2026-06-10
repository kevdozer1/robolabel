# Do these annotations actually improve training?

Short answer: **not demonstrated, and on at least one careful test, no.** Read this
before assuming labelkit's annotations help a downstream model.

labelkit produces *measured* conditioning annotations. Whether feeding those
annotations into a world model or VLA actually lowers downstream error is a
separate question, and it is easy to fool yourself on. The author ran a
preregistered, controlled head-to-head on exactly this question (a small JEPA
world model, LeWM, finetuned on 100 BridgeData V2 episodes, pi0.7 conditioning
annotations vs. CV auxiliary targets, three seeds, one fixed held-out split). The
honest findings:

- The apparent win from the full pi0.7 conditioning stack at one scale was largely
  a **latent-variance artifact**: every condition finetunes the encoder, so each
  condition's error is measured in its own latent geometry. A ~3% contraction of
  the target-latent variance manufactured a ~3% "win" that **vanished under
  variance normalization** and did **not** separate from zero in action space.
- The underlying one-step latent-prediction task was close to **degenerate**: a
  trivial copy baseline beat the trained predictor, and ~60% of held-out windows
  were near-static, so the metric had little headroom for any annotation to move.
- The conditioning channel itself was **small and low-cardinality** (a few percent
  of the context-embedding norm; ~28 distinct conditioning vectors across the
  held-out set), and a content-free adapter-null control perturbed the model as
  much as the content-bearing one.

In other words: on that instrument and that task, the conditioning annotations
were not shown to improve prediction once the confounds were controlled, and the
effect sizes were small for reasons that had little to do with annotation quality.

What this means for you:

1. Do not treat labelkit's annotations as a known training win. They are a
   measured first pass, not a result.
2. If you use them for VLA finetuning or data filtering, **run your own controlled
   evaluation** — ideally in a metric that is not in the model's own latent space
   (e.g. action-space error), with a trivial-baseline floor and a variance control.
3. The calibration loop and reliability report exist precisely because label
   quality is the first thing to rule in or out before you go looking for a
   training effect.

The full preregistration, paired-bootstrap analysis, variance normalization,
inverse-dynamics probe, and trivial-baseline diagnostics live in the author's
research repository's technical report. This page is the public summary; the claim
labelkit makes is "labels + measurement + a fixing loop," not "labels that improve
training."
