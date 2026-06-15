"""Discover small SO-100/SO-101 pour + fold LeRobot datasets on the HF Hub.

Recipe: search the Hub for candidates, read each dataset's meta/info.json for
total_episodes / fps / robot_type / camera keys, pull a task string from
meta/tasks.(jsonl|parquet), and keep the ones whose task text matches
pour/fold/towel/cloth/wipe with a small episode count. No API spend; metadata only.

    python scripts/discover_datasets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SEARCHES = [
    "so100 pour", "so101 pour", "lerobot pour", "pour water",
    "so100 fold", "so101 fold", "lerobot fold", "fold towel", "fold cloth", "fold shirt",
]
TASK_RE = ("pour", "fold", "towel", "cloth", "wipe", "shirt", "napkin")
api = HfApi()


def _read_info(repo: str) -> dict | None:
    try:
        p = hf_hub_download(repo, "meta/info.json", repo_type="dataset")
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_task(repo: str) -> str:
    for fn in ("meta/tasks.jsonl", "meta/tasks.parquet"):
        try:
            p = hf_hub_download(repo, fn, repo_type="dataset")
            if fn.endswith(".jsonl"):
                line = Path(p).read_text(encoding="utf-8").splitlines()[0]
                return str(json.loads(line).get("task", ""))[:80]
            import pandas as pd
            df = pd.read_parquet(p)
            col = "task" if "task" in df.columns else df.columns[0]
            return str(df[col].iloc[0])[:80]
        except Exception:
            continue
    return ""


def _camera_keys(info: dict) -> list[str]:
    feats = info.get("features", {}) or {}
    return [k for k in feats if k.startswith("observation.images")]


def main() -> int:
    candidates: dict[str, object] = {}
    for q in SEARCHES:
        try:
            for d in api.list_datasets(search=q, limit=25):
                candidates[d.id] = d
        except Exception as e:  # noqa: BLE001
            print(f"search '{q}' failed: {type(e).__name__}", file=sys.stderr)
    print(f"{len(candidates)} unique candidates; probing metadata...\n", file=sys.stderr)

    rows = []
    for repo in sorted(candidates):
        info = _read_info(repo)
        if not info:
            continue
        robot = str(info.get("robot_type", "")).lower()
        n_ep = int(info.get("total_episodes", 0) or 0)
        cams = _camera_keys(info)
        task = _read_task(repo).lower()
        idl = repo.lower()
        is_so = ("so100" in robot or "so101" in robot or "so100" in idl or "so101" in idl)
        matches = any(t in task or t in idl for t in TASK_RE)
        if not (is_so and matches and cams):
            continue
        kind = "pour" if ("pour" in task or "pour" in idl) else "fold/cloth"
        rows.append((kind, n_ep, repo, robot, info.get("codebase_version", "?"),
                     info.get("fps", "?"), cams, task))

    rows.sort(key=lambda r: (r[0], abs(r[1] - 10)))  # prefer ~10 episodes
    print(f"\n{'KIND':10} {'EPS':>4} {'REPO':52} {'VER':5} {'FPS':>4}  CAMERAS | task")
    for kind, n_ep, repo, _robot, ver, fps, cams, task in rows:
        print(f"{kind:10} {n_ep:>4} {repo:52} {ver:5} {fps:>4}  {','.join(c.split('.')[-1] for c in cams)} | {task[:46]}")
    if not rows:
        print("(no SO-100/SO-101 pour/fold candidates with cameras found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
