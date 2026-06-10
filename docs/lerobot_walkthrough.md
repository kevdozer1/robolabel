# Walkthrough: annotate a real LeRobot dataset

This is the real-provider path, against a small public LeRobot dataset. The
offline `robovid_conditioner demo` (mock provider) proves the plumbing; this proves the
product on real robot video.

## Dataset

We use [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace),
an official LeRobot SO-101 teleoperation dataset (**Apache-2.0**, so redistribution
of derived annotations is fine). Verified with this tool:

- 50 episodes, 30 fps
- cameras: `observation.images.up`, `observation.images.side`
- frames decode to 480×640 RGB
- per-episode task strings, e.g. *"pink lego brick into the transparent box"*

## Compatibility

- Written and tested against **lerobot 0.4.4** (`lerobot.datasets.lerobot_dataset.LeRobotDataset`)
  and the **v3.0** LeRobot dataset metadata layout (episode rows carry
  `dataset_from_index` / `dataset_to_index`, and a per-episode `tasks` list).
- Install the extra: `pip install 'robovid_conditioner[lerobot]'`.
- The adapter reads frames lazily through the public dataset API. If a newer
  LeRobot changes the episode-index field names, the adapter is the one place to
  update (`src/robovid_conditioner/adapters/lerobot.py`).

## Annotate

Set the credential for your provider (the error message names the exact variable
if you forget):

```bash
export GEMINI_API_KEY=...        # or OPENAI_API_KEY for --provider openai
```

Annotate the first 5 episodes with Gemini, using the `side` camera:

```bash
robovid_conditioner annotate \
  --source lerobot \
  --target lerobot/svla_so101_pickplace \
  --provider gemini \
  --camera-key observation.images.side \
  --limit 5 \
  --out so101_annotations
```

This writes:

```
so101_annotations/
  annotations.parquet            # the sidecar (episode metadata, subtasks, subgoals)
  raw_receipts/<episode>/*.json   # every VLM call's raw response, for provenance
  subgoal_frames/*.png            # extracted subgoal keyframes
```

Inspect what you got, and what it cost:

```bash
robovid_conditioner gate   --annotations so101_annotations     # automatic red flags
robovid_conditioner cost   --annotations so101_annotations     # estimated $ + raw receipt count
robovid_conditioner export --annotations so101_annotations --out so101.jsonl
```

## Calibrate (the point of the tool)

Build a gold file and review a handful of episodes by hand, then measure how far
the VLM was from you:

```bash
robovid_conditioner review \
  --annotations so101_annotations \
  --gold so101_gold.json \
  --source lerobot --target lerobot/svla_so101_pickplace      # shows clip frames

robovid_conditioner reliability --gold so101_gold.json --json so101_reliability.json
```

`reliability` prints subtask-boundary temporal IoU, quality exact / within-one
agreement, and subgoal frame agreement over the episodes you reviewed. Your VLM
labels (`auto`) and your corrections (`gold`) are stored in separate blocks of the
gold file; neither overwrites the other.

## Custom task family

The default rubric was tuned on tabletop pick-and-place. For a different task
family, copy the bundled rubric and edit the prompts and the quality scale:

```bash
python -c "import robovid_conditioner, shutil, pathlib; \
  shutil.copy(pathlib.Path(robovid_conditioner.__file__).parent / 'rubric.yaml', 'my_rubric.yaml')"
# edit my_rubric.yaml, then:
robovid_conditioner annotate ... --rubric my_rubric.yaml
```
