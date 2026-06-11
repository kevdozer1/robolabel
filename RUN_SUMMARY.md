# Run summary

**What this is.** `robolabel` (renamed from the working name `robovid_conditioner`) — a
LeRobot-first tool that uses a VLM to draft subtask boundaries + episode quality + subgoal
frames, then measures those drafts against a human gold set. Built, ablated, written up,
and packaged to **v0.1.0** on branch `main` (local; **not pushed, not published**).

**What ran.** The full S0–S4 × {Gemini 2.5 Flash, Pro} strategy ablation + free baselines
(`S_grip`, uniform-fifths) on `lerobot/svla_so101_pickplace`, 30 tune / 20 held-out test,
**$16.54 of the $30 ceiling**. Then a zero-API completion pass adding boundary-placement
and distributional metrics reconstructed from the cached receipts.

**What won, honestly.** Mechanical selection picked **Gemini 2.5 Pro, S2** (tune IoU
0.453) — but on the held-out test it scored **0.444 vs the S0-Flash baseline's 0.460**, so
the strategy layer did **not** improve mean boundary IoU on unseen data. Mean IoU was the
wrong number: on the same held-out test the grounded strategy hit **36% more gold
transitions within ±5 frames** (recall 0.307 vs 0.226), eliminated every degenerate/uniform
failure episode (5/20 → 0/20), and (with Pro) cut catastrophic quality false-negatives 3→1.
The free baselines (S_grip 0.184, uniform-fifths 0.359 IoU on test) are floors the VLM
clears by ~0.10–0.25. **Read [STRATEGY_REPORT.md](STRATEGY_REPORT.md) first.**

**State.** 80 tests pass, ruff clean, secret/grep audit clean, wheel builds and installs
clean in a fresh venv, `robolabel` CLI + offline demo work. Bugs found+fixed across the
arc: non-contiguous episode-subset indexing (eval loader), and the cost-projection probe
reading $0 off a cached receipt (now bypasses cache / falls back to the price table).
ep7 (human gold = a single segment) motivated demoting the min-granularity hard-reject to
a `warn` default. The four positioning workstreams shipped: `export --format lerobot`
(round-trip-tested), the README repositioning, and `docs/launch_checklist.md` +
`docs/blog_post.md` (all DRAFT, unposted).

## Remaining human-only steps (nothing automated past this)

1. **Reserve `robolabel`** on PyPI and GitHub (was free when chosen — confirm).
2. **Review the drafts** before posting: `docs/blog_post.md`, the three issue replies and
   the Discord post in `docs/launch_checklist.md`. Nothing has been posted.
3. **Push `main` + the `v0.1.0` tag**, then publish (TestPyPI → PyPI per the checklist).
4. Decide the launch message: "robustness/placement win, not a mean-IoU win" — the report
   lays out both sides; the honest framing is the differentiator.
