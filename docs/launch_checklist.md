# Launch checklist (drafts — do not post anything yet)

Everything here is a **draft to review before posting**. Nothing in this file has
been sent. The issue-reply drafts target *real, verified* issues; two of the three
categories (subtask support, quality filtering) have **no clean open feature
request** because the capability already partly exists in lerobot, so those drafts
engage the closest real thread honestly rather than inventing a request.

---

## 0. Pre-publish blockers

- [x] **Package rename `robovid_conditioner` → `robolabel` — DONE.** Import package
  (`src/robolabel/`), `[project] name`, console-script, schema-version strings, and all
  docs/imports are `robolabel`. Fresh-venv install + offline demo + full test suite +
  secret/grep audit verified green post-rename.
- [ ] **Reserve the name.** Confirm `robolabel` is free on PyPI and as a GitHub repo
  before first publish (the earlier `labelkit` name was already taken — check).
- [ ] **Fill `STRATEGY_REPORT.md`** from `eval_out/results_{tune,test}.json` (+ merge
  the `S_grip` row from `eval_out_grip/`) once the sweep finishes; apply
  `docs/strategy_report_writeup_todo.md` (constant-5 quality baseline + catastrophic
  false-negative reframing).
- [ ] Tag `0.1.0` only after the report has real numbers.

## 1. PyPI publish steps

```bash
python -m pip install --upgrade build twine
python -m build                      # sdist + wheel into dist/
twine check dist/*
twine upload --repository testpypi dist/*    # smoke test on TestPyPI first
pip install -i https://test.pypi.org/simple/ robolabel   # verify install + `robolabel --help`
twine upload dist/*                  # real PyPI, once verified
```

Pin the supported `lerobot` range in `pyproject.toml` and the README "Compatibility"
note before upload (verified against lerobot 0.4.4).

## 2. Issue replies (drafts — verified targets)

> Norm in the repo: short, friendly, lead with thanks, link a concrete command/PR,
> offer to help or PR. Keep to ~2–3 sentences. Post these only after `robolabel` is
> public (so the links resolve).

### (a) Subtask annotation — target: [#3407](https://github.com/huggingface/lerobot/issues/3407) (open)
*("Why does SARM use a custom subtask format instead of the dataset's native subtask field?")*

> The friction is that lerobot has two subtask shapes — the native
> `meta/subtasks.parquet` + per-frame `subtask_index`, versus the SARM script's
> per-episode `subtask_*` columns — and they don't line up. I mapped both field-for-field
> while writing an exporter (`robolabel export --format lerobot` writes the
> `meta/subtasks.parquet` convention and reconstructs the per-frame `subtask_index`);
> happy to share the mapping notes if they help reconcile the two paths.

### (b) Task-metadata editing — target: [#2326](https://github.com/huggingface/lerobot/issues/2326) (open, `enhancement`)
*(maintainer call for "Develop LeRobotDataset tools"; first item is editing task descriptions.)*

> For the "edit/extend a task description after the fact" item: I keep per-episode
> annotations (subtask boundaries + quality) in a sidecar and round-trip the subtask
> half into the `meta/subtasks.parquet` convention, so most of the read/edit/write
> plumbing already exists in `robolabel`. Glad to contribute a focused
> task-metadata `--writeback` or compare notes if that helps move #2326.

### (c) Quality filtering / curation — target: [#630](https://github.com/huggingface/lerobot/issues/630) (closed; `remove_episodes` resolved deletion)
*(No open issue requests quality scoring; #630 is the closest — deleting episodes.)*

> `remove_episodes` covers the deletion; the harder half is deciding *which*
> episodes. `robolabel` scores per-episode quality with a VLM and flags low-score
> outliers as `needs_review` (it never auto-drops), all measured against a human
> gold set (`robolabel gate` / `robolabel reliability`). Happy to share that curation
> flow if the "which episodes?" question comes up again.

## 3. LeRobot Discord post (draft — lead with the reliability table)

> **robolabel** — VLM-drafted subtask + quality annotations for LeRobot datasets,
> *measured* against human calibration instead of trusted. On
> `lerobot/svla_so101_pickplace`, against a 50-episode human gold set:
>
> | _fill from STRATEGY_REPORT.md after the sweep_ | boundary IoU | quality (vs trivial) | subgoal |
> |---|---|---|---|
> | out-of-the-box (S0) | _tbd_ | _tbd_ | _tbd_ |
> | best strategy (test) | _tbd_ | _tbd_ | _tbd_ |
> | free proprioceptive baseline (S_grip) | 0.204 | — | _tbd_ |
>
> It exports to the native `meta/subtasks.parquet` subtask convention, records every
> VLM call's cost/receipt, and ships the reliability report + the full strategy
> ablation. Repo + writeup: <link>, `STRATEGY_REPORT.md`. Feedback welcome.

Keep it to that table + two sentences; no hype. Fill the numbers from the report —
do not post placeholders.

## 4. Links

- Strategy ablation writeup: [`STRATEGY_REPORT.md`](../STRATEGY_REPORT.md)
- Output schema + LeRobot export: [`SCHEMA.md`](../SCHEMA.md)
- Does this help training? [`docs/why.md`](why.md)
