# Run summary

The full S0–S4 × {Gemini 2.5 Flash, Pro} strategy ablation (plus the free S_grip
proprioceptive baseline) ran end to end on `lerobot/svla_so101_pickplace` — 10 tune
cells × 30 episodes, mechanical selection, and one held-out 20-episode test cell — for
**$16.54 of the $30 ceiling**, with the four positioning/interop workstreams
(LeRobot export, S_grip, README, launch collateral) all built, tested (77 pass, ruff
clean), and committed on branch `strategy-layer`. The mechanically-selected winner was
**Gemini 2.5 Pro, strategy S2** (tune boundary IoU 0.453, clearing the +0.05 bar by
0.003) — but **on the held-out test it scored 0.444 while the S0-Flash baseline scored
0.460**, so the strategy layer did **not** improve mean boundary IoU on unseen data.
What it *did* do, confirmed on test, is eliminate the catastrophic failure bands
(S0-Flash: 5/20 episodes degenerate-or-uniform; the grounded winner: 0/20) and, via the
stronger model, cut catastrophic quality false-negatives (3→1 on tune) — so grounding
buys **robustness and data hygiene, not a higher average**, on this easy near-saturated
dataset. **Read [STRATEGY_REPORT.md](STRATEGY_REPORT.md) first** — its verdict and the
three failure-band exhibits (esp. episode 7, where the human label is a *single* segment
that S2's min-granularity floor forbids) are the whole story; one bug was found and
fixed mid-run (eval loader mis-indexed non-contiguous episode subsets) and one wrinkle
documented (preflight projection read $0 off a cached probe, so the budget gate was inert
— true spend stayed under ceiling regardless).
