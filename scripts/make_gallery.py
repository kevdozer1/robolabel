"""Assemble the unified evaluation gallery across the task families.

For each task it builds a per-task inspect_data set from that task's `robolabel run`
everything-on output (so the payload carries the module fields — quality/speed/novelty/
curation/control + subgoal pointers), then writes gallery.json. Launch with:

    python scripts/make_gallery.py
    robolabel gallery --config gallery.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# task -> (run-output dir, lerobot target, camera key [first/auto], n episodes)
TASKS = [
    ("pick-place", "run_out/full_pp", "lerobot/svla_so101_pickplace", "observation.images.up", 8),
    ("pour", "run_out/full", "Ishah8840/so101_pouring", "observation.images.front", 5),
    ("fold", "run_out/full_fold", "the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead", 8),
]


def main() -> int:
    specs = []
    for task, run_dir, target, cam, n in TASKS:
        if not (ROOT / run_dir / "annotations.parquet").exists():
            print(f"[skip] {task}: {run_dir} missing (run the everything-on config first)")
            continue
        out = f"inspect_data/gallery_{task.replace('-', '')}.json"
        subprocess.run(
            [sys.executable, "scripts/build_inspect_data.py", "from-annotations",
             "--track", f"grounded={run_dir}", "--dataset", target, "--out", out],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        specs.append({"task": task, "data": out, "source": "lerobot",
                      "target": target, "camera_key": cam, "episodes": f"0-{n - 1}"})
        print(f"[ok] {task}: {n} eps -> {out}")
    (ROOT / "gallery.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"\nwrote gallery.json ({len(specs)} tasks). Launch:\n  robolabel gallery --config gallery.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
