# Run summary

The full annotation-strategy system (S0–S4 grounding → vocabulary → refinement →
self-consistency), the gate failure-band detectors, schema v2, the eval harness,
and the autonomous budget-capped runner are **built, committed on branch
`strategy-layer`, and green** (68 tests, ruff clean, offline dry-run of the whole
Phase 1–4 orchestration passing). **The live ablation did not run and $0 was spent**,
because no `GEMINI_API_KEY` is reachable — it is not in `labelkit/.env` nor in any
environment scope (a `$env:` set in your interactive shell does not reach the
process the run spawns), so there are no win, test number, or measured spend to
report yet. Everything that costs nothing is verified: the split is intact (30/20,
seeded), the SO-101 dataset + `observation.images.side` camera resolve and decode
from the local cache, and the cost projection (~$36.7 for the full sweep) already
tells us the run will hit the $30 ceiling and degrade gracefully via the priority
order. **Look first at `STRATEGY_REPORT.md` → "Run status"**, then put the key in
`labelkit/.env` and launch the one command there (`python scripts/run_ablation.py
… --budget 30`) — it is fully autonomous, resumable, and stops itself at the
ceiling, so it is safe to start and walk away.
