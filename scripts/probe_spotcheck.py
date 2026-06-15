"""Render labeled segment montages for the cross-task probe, for an author/automated
spot-check of evidence factual-accuracy and whether open-vocab phase names read sensibly.

    python scripts/probe_spotcheck.py pour s2_open 0 1 2
    python scripts/probe_spotcheck.py fold s2_open 0 1 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from robolabel.adapters import build_source  # noqa: E402
from robolabel.inspect_data import segments_from_records  # noqa: E402
from robolabel.schema import episode_records, read_annotations  # noqa: E402

DATASET = {
    "pour": ("Ishah8840/so101_pouring", "observation.images.front"),
    "fold": ("the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead"),
}
OUT = Path("probe_spotcheck")
THUMB = 240


def main() -> int:
    task = sys.argv[1]
    cond = sys.argv[2]
    want = [int(x) for x in sys.argv[3:]] or [0, 1, 2]
    repo, cam = DATASET[task]
    df = read_annotations(f"probe_{task}/{cond}")
    # Load the SAME contiguous range the annotation used (range(8)); a non-contiguous
    # subset would re-index the frames and break ep.frame() (known adapter limitation).
    src = build_source("lerobot", repo, camera_key=cam, episodes=list(range(8)))
    eps = {ep.episode_id: ep for ep in src}
    OUT.mkdir(exist_ok=True)
    for eid in [str(w) for w in want]:
        if eid not in eps:
            continue
        segs = segments_from_records(episode_records(df, eid)["subtasks"])
        ep = eps[eid]
        tiles = []
        for i, s in enumerate(segs):
            mid = (int(s["start"]) + int(s["end"])) // 2
            arr = np.asarray(ep.frame(min(mid, ep.num_frames - 1))).astype("uint8")
            img = Image.fromarray(arr).convert("RGB")
            r = THUMB / img.width
            img = img.resize((THUMB, max(1, int(img.height * r))))
            cap = f"{i}: {s.get('phase') or '-'}" + (f" -> {s['target']}" if s.get("target") else "")
            canvas = Image.new("RGB", (img.width, img.height + 42), "white")
            canvas.paste(img, (0, 42))
            d = ImageDraw.Draw(canvas)
            d.text((3, 2), f"{task} {cond} ep{eid} seg{i} f{mid}", fill=(0, 0, 0))
            d.text((3, 14), cap[:50], fill=(180, 0, 0))
            d.text((3, 28), (s.get("evidence") or "")[:50], fill=(0, 90, 0))
            tiles.append(canvas)
        if not tiles:
            continue
        h = max(t.height for t in tiles)
        strip = Image.new("RGB", (sum(t.width for t in tiles), h), "white")
        x = 0
        for t in tiles:
            strip.paste(t, (x, 0))
            x += t.width
        path = OUT / f"{task}_{cond}_ep{eid}.png"
        strip.save(path)
        labels = " | ".join(f"{i}:{s.get('phase') or '-'}>{s.get('target') or 'none'}" for i, s in enumerate(segs))
        print(f"{path}  ({len(segs)} segs): {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
