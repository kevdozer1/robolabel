# Run summary (v0.2 close-out + cross-task probe)

This run brought `robolabel` to a clean launchable state: it re-graded the v3 `phase → target`
labels (phases correct, targets 100%, one target-naming residual → `CLAIMS.md` row 15 now
`fixed-and-spot-checked`), scoped the tool in the README and `docs/why.md` to VLA
subtask-conditioning + dataset curation rather than world-model training, reconciled the
honest-state docs, and verified the wheel builds, passes `twine check`, and installs into a
fresh venv to run `robolabel demo` offline. It also shipped a first-class open-vocabulary
grounded variant (`S2-open`; the closed-vocab `S2` default is untouched) and spent **$0.75 of
the $6 ceiling** on a gold-free cross-task probe over pour and cloth-fold. **Finding:**
frame-grounding still eliminates the catastrophic failure bands off pick-and-place (**0/8** on
both pour and fold), while the closed pick-and-place phase vocabulary degrades — 17.5% of pour
segments coerced to `other` — exactly where the open-vocab phases (`pour water`, `perform
fold`) read correctly. State: **105 tests pass**, ruff clean, secret scan clean, `.env`
git-ignored, frozen SO-101 ablation numbers untouched, and nothing pushed or published.
**Publish** when ready with `python -m twine upload dist/*` (after reserving the `robolabel`
PyPI/GitHub name); **the one decision left to you** is whether to push `main` to GitHub and
publish to PyPI.
