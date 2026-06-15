"""One-off: render a labeled montage per episode for the v3 phase/target spot-check.

For each sampled episode in inspect_data/fresh_v3.json, tile the MID-frame of every
grounded segment into a horizontal strip, captioned "i: phase -> target". The author
(or an automated frame inspection) then judges whether each phase+target matches what
the frame shows. Zero API — reads cached LeRobot frames only.

    python scripts/spotcheck_frames.py 0 1 3 6 9 12 15 18
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from robolabel.adapters import build_source  # noqa: E402

DATA = "inspect_data/fresh_v3.json"
DATASET = "lerobot/svla_so100_stacking"
CAMERA = "observation.images.top"
OUT = Path("spotcheck")
THUMB = 240


def main() -> int:
    want = set(sys.argv[1:]) or {"0", "1", "3", "6", "9", "12", "15", "18"}
    payload = json.loads(Path(DATA).read_text(encoding="utf-8"))
    by_id = {str(e["episode_id"]): e for e in payload["episodes"]}
    source = build_source("lerobot", DATASET, camera_key=CAMERA)
    eps = {ep.episode_id: ep for ep in source}
    OUT.mkdir(exist_ok=True)

    for eid in sorted(want, key=int):
        e = by_id.get(eid)
        if e is None:
            print(f"ep {eid}: not in {DATA}", file=sys.stderr)
            continue
        segs = e["tracks"]["grounded"]["segments"]
        ep = eps.get(eid)
        if ep is None:
            print(f"ep {eid}: not in dataset", file=sys.stderr)
            continue
        tiles = []
        for i, s in enumerate(segs):
            mid = (int(s["start"]) + int(s["end"])) // 2
            arr = np.asarray(ep.frame(mid)).astype("uint8")
            img = Image.fromarray(arr).convert("RGB")
            ratio = THUMB / img.width
            img = img.resize((THUMB, max(1, int(img.height * ratio))))
            cap = f"{i}: {s.get('phase') or '-'}" + (f" -> {s['target']}" if s.get("target") else "")
            canvas = Image.new("RGB", (img.width, img.height + 30), "white")
            canvas.paste(img, (0, 30))
            d = ImageDraw.Draw(canvas)
            d.text((3, 3), f"ep{eid} seg{i} f{mid}", fill=(0, 0, 0))
            d.text((3, 15), cap[:46], fill=(180, 0, 0))
            tiles.append(canvas)
        if not tiles:
            continue
        h = max(t.height for t in tiles)
        strip = Image.new("RGB", (sum(t.width for t in tiles), h), "white")
        x = 0
        for t in tiles:
            strip.paste(t, (x, 0))
            x += t.width
        path = OUT / f"ep{eid}.png"
        strip.save(path)
        labels = " | ".join(f"{i}:{s.get('phase') or '-'}>{s.get('target') or 'none'}"
                            for i, s in enumerate(segs))
        print(f"ep{eid} ({len(segs)} segs) -> {path}")
        print(f"   labels: {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
