"""Part 4 — the consumability chain, run end to end (zero-API).

annotate (mock S2, so segments carry phases) -> export --format lerobot ->
reload meta/subtasks.parquet via lerobot's own loader -> assert every frame's
subtask_index resolves to the subtask segment it falls in.

Prints PASS/FAIL + the exact assertion. Honest about the one step that is NOT done:
writing the per-frame subtask_index into the dataset's binary data/ parquet (we keep a
non-destructive metadata overlay), which a full SARM dataloader would need.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robolabel.annotate import annotate_source  # noqa: E402
from robolabel.demo import synthetic_source  # noqa: E402
from robolabel.export_lerobot import (  # noqa: E402
    SUBTASKS_REL_PATH,
    export_lerobot_subtasks,
    frame_subtask_indices,
)
from robolabel.providers import build_provider  # noqa: E402
from robolabel.schema import episode_records, list_episode_ids, read_annotations  # noqa: E402
from robolabel.strategy import load_strategy  # noqa: E402


def run() -> int:
    work = Path(tempfile.mkdtemp())
    ann_dir = work / "annotations"
    export_dir = work / "lerobot_export"

    # 1) annotate (mock S2 -> grounded segments with phases)
    annotate_source(synthetic_source(3), ann_dir, provider=build_provider("mock"),
                    strategy=load_strategy("S2"))
    print("1. annotated 3 synthetic episodes with mock S2 ->", ann_dir / "annotations.parquet")

    # 2) export to the LeRobot subtask convention
    manifest = export_lerobot_subtasks(ann_dir, export_dir)
    print("2. exported:", manifest["files"], "| subtask vocab:", manifest["n_subtasks_vocab"])

    # 3) reload meta/subtasks.parquet via lerobot's OWN loader if available, else pandas
    try:
        from lerobot.datasets.utils import load_subtasks
        subtasks = load_subtasks(export_dir)
        loader = "lerobot.datasets.utils.load_subtasks"
    except Exception:  # noqa: BLE001 - lerobot extra not installed
        import pandas as pd
        subtasks = pd.read_parquet(export_dir / SUBTASKS_REL_PATH)
        loader = "pandas (lerobot extra not installed; same parquet)"
    print(f"3. reloaded meta/subtasks.parquet via {loader}: {len(subtasks)} subtasks")

    # 4) assert: every frame's subtask_index resolves to the right subtask
    df = read_annotations(ann_dir)
    checked = 0
    for eid in list_episode_ids(df):
        rec = episode_records(df, eid)
        per_frame = frame_subtask_indices(rec["subtasks"], rec["num_frames"], subtasks)
        # the expected subtask phrase per frame
        seg_for = {}
        for s in rec["subtasks"]:
            for f in range(int(s["start_frame"]), int(s["end_frame"]) + 1):
                seg_for[f] = str(s["subtask_text"])
        for f, idx in enumerate(per_frame):
            # lerobot's resolution rule: meta.subtasks.iloc[idx].name is the subtask string
            assert subtasks.iloc[idx].name == seg_for[f], (eid, f, idx)
            checked += 1
    print(f"4. ASSERT PASS: all {checked} frames across 3 episodes resolve "
          "subtasks.iloc[subtask_index].name == the segment they fall in.")
    print("\nNot done (honest): the per-frame subtask_index is materialized from the overlay,"
          " not written into the dataset's binary data/ parquet. A full SARM/VLA dataloader that"
          " expects a per-frame `subtask_index` feature would need that write (deliberately not done"
          " to keep the export non-destructive). See SCHEMA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
