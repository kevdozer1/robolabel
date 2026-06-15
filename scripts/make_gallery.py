"""Assemble a `robolabel gallery` config across all four task families.

Reuses the existing per-task inspect_data JSONs, slices each to the first N episodes
(episode ids 0..N-1, a contiguous prefix so served frames stay aligned), writes the
balanced slices to inspect_data/gallery_<task>.json, and emits gallery.json.

    python scripts/make_gallery.py
    robolabel gallery --config gallery.json
"""
from __future__ import annotations

import json
from pathlib import Path

N = 8  # episodes per task in the gallery

# task -> (source inspect_data json, lerobot target, camera key)
TASKS = [
    ("pick-place", "inspect_data/so101.json", "lerobot/svla_so101_pickplace", "observation.images.side"),
    ("stacking", "inspect_data/fresh_v3.json", "lerobot/svla_so100_stacking", "observation.images.top"),
    ("pour", "inspect_data/pour.json", "Ishah8840/so101_pouring", "observation.images.front"),
    ("fold", "inspect_data/fold.json", "the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead"),
]


def main() -> int:
    specs = []
    for task, src, target, cam in TASKS:
        p = Path(src)
        if not p.exists():
            print(f"[skip] {task}: {src} missing (build it first)")
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        keep = {str(i) for i in range(N)}
        payload["episodes"] = [e for e in payload["episodes"] if str(e["episode_id"]) in keep]
        out = Path("inspect_data") / f"gallery_{task.replace('-', '')}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        specs.append({"task": task, "data": out.as_posix(), "source": "lerobot",
                      "target": target, "camera_key": cam, "episodes": f"0-{N - 1}"})
        print(f"[ok] {task}: {len(payload['episodes'])} eps -> {out}")
    Path("gallery.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"\nwrote gallery.json ({len(specs)} tasks). Run:\n  robolabel gallery --config gallery.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
