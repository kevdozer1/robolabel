# robovid_conditioner

**Generate π0.7-style conditioning annotations — subtask temporal boundaries, episode quality/strategy metadata, subgoal keyframes — for LeRobot datasets, using the VLM of your choice, with a built-in human calibration loop and measured reliability.**

VLM labels are wrong often enough that you need to know how wrong; this tool measures it. The honest claim here is not "good labels." It is **labels + measurement + a fixing loop**.

![What robovid_conditioner adds to one raw LeRobot episode: subtask boundaries, an episode quality/mistake judgment, and subgoal keyframes.](docs/figures/annotation_overview.png)

*One real SO-101 episode, labeled by Gemini 2.5 Flash: the filmstrip is the raw video; the numbered bar is the subtask segmentation; ▼ marks each subgoal keyframe; the chip is the quality/mistake judgment. That is everything the tool adds — and then it measures how much a human disagrees with it.*

![Pipeline: LeRobot episode → annotate with your VLM → subtasks · quality · subgoals → review/calibrate → measured reliability.](docs/figures/pipeline.png)

---

## 60-second quickstart

Install and run the offline demo (no API key, finishes in seconds):

```bash
pip install -e .          # or: uv pip install -e .
robovid_conditioner demo --out demo_out
```

That generates tiny synthetic episodes, annotates them with the (meaningless)
mock provider, and writes a valid `annotations.parquet` — proving the pipeline
end to end without spending a cent.

Now do it for real on a LeRobot dataset with a real VLM:

```bash
pip install -e '.[lerobot]'
export GEMINI_API_KEY=...          # the error names the exact var if you forget

robovid_conditioner annotate \
  --source lerobot \
  --target lerobot/svla_so101_pickplace \
  --provider gemini --limit 5 \
  --out so101_annotations

robovid_conditioner gate        --annotations so101_annotations          # automatic red flags
robovid_conditioner review      --annotations so101_annotations --gold so101_gold.json \
                     --source lerobot --target lerobot/svla_so101_pickplace
robovid_conditioner reliability --gold so101_gold.json                    # VLM-vs-you agreement
```

Providers: `gemini`, `openai`, local `qwen` (`pip install '.[qwen]'`), and `mock`.
Adding another is one new file in `src/robovid_conditioner/providers/`. Full real walkthrough:
[`docs/lerobot_walkthrough.md`](docs/lerobot_walkthrough.md). Output schema:
[`SCHEMA.md`](SCHEMA.md).

---

## Why this exists

This tool was built *because of* the following numbers, not in spite of them. On
100 BridgeData V2 episodes labeled with **Gemini 2.5 Flash**, a human reviewer:

- changed **58 / 100** quality scores,
- agreed exactly with the VLM quality score only **0.42** of the time (**0.77** within one point),
- matched the VLM's subtask boundaries at a temporal IoU of **0.683**,
- and picked the same subgoal frame the VLM did only **0.347** of the time.

A VLM that disagrees with a human on more than half the quality scores, and picks
the "right" subgoal frame about a third of the time, is not a labeling oracle —
it is a fast first pass that you must measure and correct. So robovid_conditioner ships the
two things that turn a fast first pass into usable data: a **calibration loop**
(`robovid_conditioner review` — a browser GUI where you watch each clip, scrub
frame by frame, and set a boundary or subgoal from the current frame), and a **reliability report**
(`robovid_conditioner reliability`) that tells you how far the VLM was from you on *your*
data, in the same units (boundary IoU, score agreement, subgoal agreement).

If those numbers are bad enough on your dataset, the honest output of this tool is
"don't trust these labels yet" — and it will tell you that.

---

## Scope honesty

The default rubric (`src/robovid_conditioner/rubric.yaml`) was tuned on **tabletop
pick-and-place** teleoperation (the BridgeData V2 / SO-10x family). It has **not**
been validated on:

- long-horizon or multi-stage tasks,
- deformable-object manipulation,
- mobile manipulation,
- multi-view / multi-camera reasoning (robovid_conditioner currently labels from one camera).

The rubric is config, not code: copy `rubric.yaml`, edit the prompts and the
quality scale, and pass `--rubric your_rubric.yaml`. Expect to re-tune the prompts
and re-measure reliability for a new task family.

---

## What this is *not*

- **Not a format converter.** It does not convert between dataset formats. If you
  need that, use a format tool (e.g. a dataset "forge"); robovid_conditioner reads LeRobot
  and writes a sidecar.
- **Not a dataset standard.** It does not define how robot data should be stored.
  [LeRobot](https://github.com/huggingface/lerobot) is the standard; robovid_conditioner
  annotates it.
- **Not a labeling vendor.** There is no service, no account, no data leaving your
  machine except the VLM API calls you choose to make. You bring the VLM key.
- **Not evidence that these annotations improve training.** Whether pi0.7-style
  conditioning annotations actually help VLA finetuning is a separate, hard
  question. The honest current answer — and the methodology behind it — is written
  up in [`docs/why.md`](docs/why.md); read it before assuming these labels help a
  downstream model. robovid_conditioner gives you measured labels, not a training result.

---

## Status

Beta, single-author. The annotations and gold schemas are versioned but **may
change before 1.0**. Linux and macOS are the supported platforms (Windows is not a
target). Issues and PRs welcome.

License: [Apache-2.0](LICENSE).
