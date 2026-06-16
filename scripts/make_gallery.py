"""Assemble the unified evaluation gallery across the task families, with a CORPUS-LEVEL
rescore of the population-relative fields.

For each task it builds a per-task inspect_data set from that task's `robolabel run`
everything-on output, then re-references novelty / curation value+tier / speed tier to the
POOLED distribution across all tasks (global thresholds + a population guard) so the tiers are
meaningful rather than normalized within one small same-y run. The per-run parquets are left
untouched (they record what `robolabel run` produced); only the gallery JSONs carry the corpus
scores. Launch:

    python scripts/make_gallery.py
    robolabel gallery --config gallery.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robolabel.adapters import build_source  # noqa: E402
from robolabel.corpus import rescore_corpus  # noqa: E402
from robolabel.novelty import episode_embedding  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402
from robolabel.schema import episode_records, list_episode_ids, read_annotations  # noqa: E402

# task -> (run-output dir, lerobot target, camera key [first/auto], n episodes)
TASKS = [
    ("pick-place", "run_out/full_pp", "lerobot/svla_so101_pickplace", "observation.images.up", 8),
    ("pour", "run_out/full", "Ishah8840/so101_pouring", "observation.images.front", 8),
    ("fold", "run_out/full_fold", "the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead", 8),
]


def _present():
    return [t for t in TASKS if (ROOT / t[1] / "annotations.parquet").exists()]


def corpus_scores(tasks) -> dict:
    """Pool frame embeddings + quality + speed_norm across all tasks; rescore globally."""
    pool = {}
    for task, run_dir, target, cam, n in tasks:
        df = read_annotations(ROOT / run_dir)
        src = {e.episode_id: e for e in build_source("lerobot", target, camera_key=cam,
                                                     episodes=list(range(n)))}
        for eid in list_episode_ids(df):
            meta = episode_records(df, eid)["metadata"]
            ep = src.get(eid)
            pool[(task, eid)] = {
                "emb": episode_embedding(ep.frame, ep.num_frames) if ep else None,
                "quality": meta.get("quality"), "speed_norm": meta.get("speed_norm"),
            }
    # drop episodes whose frames couldn't be embedded
    pool = {k: v for k, v in pool.items() if v["emb"] is not None}
    return rescore_corpus(pool, min_population=load_rubric().curation_min_population)


def main() -> int:
    tasks = _present()
    if not tasks:
        print("no run outputs found; run the everything-on config first.")
        return 1
    scores = corpus_scores(tasks)
    tiered = next(iter(scores.values()))["tiered"] if scores else False
    specs = []
    for task, run_dir, target, cam, n in tasks:
        out = ROOT / f"inspect_data/gallery_{task.replace('-', '')}.json"
        subprocess.run(
            [sys.executable, "scripts/build_inspect_data.py", "from-annotations",
             "--track", f"grounded={run_dir}", "--dataset", target, "--out", str(out.relative_to(ROOT))],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # patch corpus-relative fields into the gallery payload (leave the run parquet as-is)
        payload = json.loads(out.read_text(encoding="utf-8"))
        for e in payload["episodes"]:
            s = scores.get((task, str(e["episode_id"])))
            if s:
                e.setdefault("modules", {}).update({
                    "novelty": s["novelty"], "curation_value": s["curation_value"],
                    "curation_tier": s["curation_tier"], "speed": s["speed"]})
                e["corpus_tiered"] = s["tiered"]
        payload["corpus_tiered"] = tiered
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        specs.append({"task": task, "data": str(out.relative_to(ROOT)), "source": "lerobot",
                      "target": target, "camera_key": cam, "episodes": f"0-{n - 1}"})
        print(f"[ok] {task}: {n} eps -> {out.name}")
    (ROOT / "gallery.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print(f"\ncorpus-relative tiers: {'assigned' if tiered else 'insufficient population (raw scores only)'}")
    print(f"wrote gallery.json ({len(specs)} tasks). Launch:\n  robolabel gallery --config gallery.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
