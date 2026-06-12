# Consumability: can a LeRobot consumer actually read these annotations?

**Verified.** The full chain runs end to end with one command and asserts correctness:
annotate → `export --format lerobot` → reload `meta/subtasks.parquet` through lerobot's
*own* `lerobot.datasets.utils.load_subtasks` → for every frame of every episode, assert
that lerobot's resolution rule `meta.subtasks.iloc[subtask_index].name` returns exactly
the subtask segment that frame falls in. On a 3-episode synthetic set this checks **all 72
frames** and passes; the same assertion is exercised in CI by
`tests/test_export_lerobot.py`. So a LeRobot consumer that reads the standard subtask
convention gets back our boundaries with the indices resolving to the correct frames.

```bash
# the whole chain, zero-API, reproducible:
python scripts/consumability_check.py
# or step by step on real annotations:
robolabel annotate --source lerobot --target <dataset> --provider gemini --strategy S2 --out ann
robolabel export   --annotations ann --format lerobot --out ann_lerobot
python -c "from lerobot.datasets.utils import load_subtasks; print(load_subtasks('ann_lerobot'))"
```

**Not yet shown (stated, not hidden).** We write a non-destructive *metadata overlay*
(`meta/subtasks.parquet` + a per-episode `episodes_subtasks.parquet`); we do **not** write
a per-frame `subtask_index` column into the dataset's binary `data/` parquet. A full
SARM/VLA-JEPA dataloader that expects `subtask_index` as a per-frame *feature* of the
dataset would need that write, so we did not instantiate one against a live dataset — that
would have been a mechanical "it ran" check on data we'd have had to mutate. The honest
status is: **the subtask convention round-trips and resolves correctly (verified); driving
a training dataloader off it is a documented next step, not a demonstrated capability.**
