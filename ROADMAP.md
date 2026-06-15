# Roadmap

Honest, small-scope roadmap. `v0.1` is the measured first release (subtask/quality/subgoal
drafts + the reliability/acceptance apparatus); everything below is explicitly *next*, not
done. Each line says what would move it and what evidence would close it.

## v0.1 (current) — measured first release

- VLM-drafted subtask boundaries, episode quality (1–5), subgoal keyframes for LeRobot.
- Grounded strategy layer (S0–S4); failure-band elimination verified on two pick-and-place /
  stacking datasets; mean-IoU non-improvement reported as prominently (`STRATEGY_REPORT.md`,
  `FRESH_TRIAL_REPORT.md`).
- Schema v3 structured labels (`phase → target`); reliability report; gate (flags, never
  drops); blind-grading viewer; LeRobot subtask export (round-trip tested).
- Claims audit (`CLAIMS.md`); per-call cost/receipts; zero-API reconstruction.

## v0.2 — cross-task generalization + open vocabulary (in progress)

- **Open-vocabulary grounded variant (`S2-open`)** — shipped behind the closed-vocab S2
  default: frame-grounding + per-boundary evidence + target slot, but free-text phase names so
  the labels fit tasks outside the hand-authored pick-and-place vocabulary. *(done; 6 tests.)*
- **Cross-task probe (pour + cloth-fold)** — gold-free check of whether grounding still
  eliminates the failure bands off pick-and-place, and where the closed pick-place vocabulary
  degrades. Finding and exact numbers: `FRESH_TRIAL_REPORT.md` → "Cross-task generalization
  probe" and `CLAIMS.md`. *(done for pour + fold; more task families would strengthen it.)*
- **Target-naming convention** — the v3 spot-check found the `target` slot drifts between the
  *moved object* and the *destination* on transport/place phases. Pick one rule (likely:
  destination for place/transport, manipuland otherwise) and enforce it in validation + prompt.
  *Closes `CLAIMS.md` row 15's residual.*
- **discover-vocab helper (optional)** — cluster the free-text open-vocab phases over ~10
  episodes into a per-dataset closed vocabulary, so a new task family can graduate from
  open-vocab to a measured closed vocab without hand-authoring.

## Beyond v0.2 (not scheduled)

- **Provider coverage** — exercise OpenAI + Qwen end to end with recorded fixtures replayed in
  CI (Gemini is the only dogfooded provider today).
- **LeRobot per-frame write-back** — materialize `subtask_index` into the binary `data/`
  parquet (currently a metadata overlay only), so a full SARM/VLA dataloader can train off it
  without an extra step. Decide whether 0.4.x can carry it cleanly or formally defer.
- **Multi-view** — labels come from one camera today; multi-view reasoning is out of scope.
- **Quality discrimination** — the quality metric can't be evaluated on a dataset that is 49/50
  one score; needs a set with real quality variance.
- **Training-utility evidence** — still unshown, and negative on one careful test
  (`docs/why.md`). Any future claim needs a controlled, non-latent-space evaluation.
