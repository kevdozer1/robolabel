"""Render docs/figures/grounded_annotations.gif — a clean eval GIF, three task panels.

Three episodes (re-annotated with the cleanup-round fixes), grounded strategy only, synced to
the playing video. Per panel: the current `phase -> target` sub-step label + a segment timeline
with a playhead; the episode quality; the continuous, motion-defined speed descriptor
(active_duration); the same-episode subgoal keyframes (labeled "selected — not generated"); and
the control modality (labeled "action coordinate frame"). Zero API (cached frames + parquet).

Honesty: every image is a real frame from the dataset; subgoals are SELECTED, never generated;
control is read from the action stream; the caption says exactly that.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from robolabel.adapters import build_source  # noqa: E402
from robolabel.schema import episode_records, read_annotations  # noqa: E402

# label, run-output dir, episode id, lerobot target, camera key
PANELS = [
    ("Pick-place", "run_out/full_pp", "7", "lerobot/svla_so101_pickplace", "observation.images.up"),
    ("Pour", "run_out/full", "3", "Ishah8840/so101_pouring", "observation.images.front"),
    ("Fold", "run_out/full_fold", "4", "the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead"),
]
OUT = Path("docs/figures/grounded_annotations.gif")
STEPS = 36
PANEL_W, VID_W, VID_H = 900, 244, 192
THUMB_W, THUMB_H = 74, 56
SEG_COLORS = ["#2563eb", "#e8752a", "#59a14f", "#b4536b", "#8c6bb1", "#3aa0a0", "#9aa5b1"]
INK, MUTED, GREEN = (24, 28, 36), (110, 118, 130), (40, 120, 60)


def _font(size: int, bold: bool = False):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_SM, F_MD, F_XL = _font(12), _font(14), _font(20, True)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _thumb(arr, w=THUMB_W, h=THUMB_H) -> Image.Image:
    return Image.fromarray(np.asarray(arr).astype("uint8")).convert("RGB").resize((w, h))


def load_panel(label, d, eid, repo, cam):
    rec = episode_records(read_annotations(d), eid)
    m = rec["metadata"]
    ep = {e.episode_id: e for e in build_source("lerobot", repo, camera_key=cam, episodes=list(range(8)))}[eid]
    segs = [{"start": int(s["start_frame"]), "end": int(s["end_frame"]),
             "phase": s.get("phase"), "target": s.get("target")} for s in rec["subtasks"]]
    keyframes = sorted(int(s["subgoal_frame_idx"]) for s in rec["subgoals"])
    return {"label": label, "ep": ep, "segs": segs, "keyframes": keyframes, "nf": ep.num_frames,
            "quality": _num(m.get("quality")), "modality": m.get("control_modality"),
            "active_seconds": _num(m.get("active_seconds")), "active_fraction": _num(m.get("active_fraction"))}


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def cur_segment(segs, frame):
    for i, s in enumerate(segs):
        if s["start"] <= frame <= s["end"]:
            return i
    return len(segs) - 1


def render_panel(P, frame_idx) -> Image.Image:
    segs, nf = P["segs"], P["nf"]
    ph = 212
    panel = Image.new("RGB", (PANEL_W, ph), "white")
    d = ImageDraw.Draw(panel)
    panel.paste(_thumb(P["ep"].frame(frame_idx), VID_W, VID_H), (10, 10))
    d.rectangle([10, 10, 10 + VID_W, 10 + VID_H], outline=(200, 205, 212))
    x = VID_W + 26
    si = cur_segment(segs, frame_idx)
    s = segs[si]
    d.text((x, 10), P["label"].upper(), font=F_SM, fill=MUTED)
    lab = f"{s['phase']} → {s['target']}" if s.get("target") else (s.get("phase") or "")
    d.text((x, 26), lab, font=F_XL, fill=_hex(SEG_COLORS[si % len(SEG_COLORS)]))
    q = f"{int(P['quality'])}/5" if P["quality"] is not None else "–"
    act = (f"{P['active_seconds']:.1f}s ({int(round((P['active_fraction'] or 0) * 100))}% active)"
           if P["active_seconds"] is not None else "–")
    d.text((x, 54), f"quality {q}    speed: active {act}", font=F_MD, fill=INK)
    d.text((x, 72), f"action coordinate frame: {P['modality'] or '–'}", font=F_MD, fill=INK)
    # timeline + playhead
    tl_x, tl_y, tl_w, tl_h = x, 98, PANEL_W - x - 20, 16
    for i, sg in enumerate(segs):
        a = tl_x + int(sg["start"] / max(1, nf - 1) * tl_w)
        b = tl_x + int(sg["end"] / max(1, nf - 1) * tl_w)
        d.rectangle([a, tl_y, b, tl_y + tl_h], fill=_hex(SEG_COLORS[i % len(SEG_COLORS)]))
    px = tl_x + int(frame_idx / max(1, nf - 1) * tl_w)
    d.line([px, tl_y - 3, px, tl_y + tl_h + 3], fill=(15, 18, 24), width=2)
    # real subgoal keyframes row
    ky = 126
    d.text((x, ky), "subgoal keyframes  (selected end-of-sub-step frames — not generated)", font=F_SM, fill=GREEN)
    kx = x
    for i, fr in enumerate(P["keyframes"]):
        panel.paste(_thumb(P["ep"].frame(min(fr, nf - 1))), (kx, ky + 16))
        d.rectangle([kx, ky + 16, kx + THUMB_W, ky + 16 + THUMB_H],
                    outline=_hex(SEG_COLORS[i % len(SEG_COLORS)]))
        kx += THUMB_W + 6
    return panel


def main() -> int:
    panels = [load_panel(*p) for p in PANELS]
    total_h = sum(render_panel(P, 0).height for P in panels)
    frames = []
    for t in range(STEPS):
        canvas = Image.new("RGB", (PANEL_W, total_h), "white")
        y = 0
        for P in panels:
            fi = int(round(t / (STEPS - 1) * (P["nf"] - 1)))
            pim = render_panel(P, fi)
            canvas.paste(pim, (0, y))
            y += pim.height
        frames.append(canvas)
        print(f"  step {t + 1}/{STEPS}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
