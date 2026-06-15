"""Confirm pour/fold candidates load through the installed lerobot and render a frame.

Tries a ranked list per task, downloading only a couple of episodes, and reports the
first repo that instantiates + renders frame 0. No API spend. Saves a sample frame to
probe_loadtest/<kind>.png for eyeballing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# lerobot 0.4.4 hard-rejects v2.0/v2.1 — these are all codebase_version v3.0.
CANDIDATES = {
    "pour": [
        ("RajatDandekar/so101_pour_chocolates", "observation.images.webcam"),
        ("SurajCreation/so101_pour_v1", "observation.images.overhead"),
        ("feiyu05/so101_grab_pour_dataset", "observation.images.front"),
    ],
    "fold": [
        ("the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead"),
        ("Integer003/fold-towel-so101-clean", "observation.images.front"),
        ("ppprock11/foldcloth2", "observation.images.front"),
    ],
}
OUT = Path("probe_loadtest")


def try_one(repo: str, cam: str) -> tuple[bool, str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robolabel.adapters.lerobot import LeRobotAdapter
    try:
        # Download only episodes 0-1 to keep it cheap.
        LeRobotDataset(repo, episodes=[0, 1])
    except Exception as e:  # noqa: BLE001
        return False, f"load(eps=[0,1]) failed: {type(e).__name__}: {str(e)[:120]}"
    try:
        src = LeRobotAdapter(repo, camera_key=cam, episodes=[0, 1])
        eps = list(src)
        ep = eps[0]
        arr = np.asarray(ep.frame(min(5, ep.num_frames - 1))).astype("uint8")
        OUT.mkdir(exist_ok=True)
        kind = "pour" if "pour" in repo.lower() else "fold"
        Image.fromarray(arr).convert("RGB").save(OUT / f"{kind}.png")
        return True, f"OK n_eps_loaded={len(eps)} ep0.num_frames={ep.num_frames} task={ep.task!r}"
    except Exception as e:  # noqa: BLE001
        return False, f"adapter/render failed: {type(e).__name__}: {str(e)[:120]}"


def main() -> int:
    chosen = {}
    for kind, lst in CANDIDATES.items():
        for repo, cam in lst:
            ok, msg = try_one(repo, cam)
            print(f"[{kind}] {repo} ({cam.split('.')[-1]}): {msg}", file=sys.stderr)
            if ok:
                chosen[kind] = (repo, cam)
                break
    print("\nCHOSEN:")
    for kind, (repo, cam) in chosen.items():
        print(f"  {kind}: {repo}  camera={cam}")
    return 0 if len(chosen) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
