"""Offline demo: replay the three bundled sample episodes with their grounded annotations.

Run from the repo root after `pip install -e '.[lerobot]'` (or just `pip install -e .` plus
`pip install 'imageio[ffmpeg]'`):

    python demo/demo.py

It needs no API key and no dataset download. For each bundled episode it prints the grounded
annotation (phase -> target sub-steps, the deterministic per-segment active components, episode
quality and motion-defined speed, and the selected subgoal frames) and regenerates the annotated
figure at demo/grounded_annotations.gif from the bundled clips.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Use the installed robolabel package for the (deterministic) active-component computation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robolabel.control import component_groups, segment_active_groups  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = ["pickplace", "pour", "fold"]
OUT = HERE / "grounded_annotations.gif"
CM = load_rubric().control_motion

PANEL_W, VID_W, VID_H, THUMB_W, THUMB_H = 900, 244, 192, 74, 56
STEPS = 36
SEG_COLORS = ["#2563eb", "#e8752a", "#59a14f", "#b4536b", "#8c6bb1", "#3aa0a0", "#9aa5b1"]
INK, MUTED, GREEN = (24, 28, 36), (110, 118, 130), (40, 120, 60)


def _font(size, bold=False):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_SM, F_MD, F_XL = _font(12), _font(14), _font(20, True)


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _thumb(arr, w=THUMB_W, h=THUMB_H):
    return Image.fromarray(np.asarray(arr).astype("uint8")).convert("RGB").resize((w, h))


def load_panel(name):
    meta = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    frames = np.stack([np.asarray(f) for f in iio.imiter(HERE / "clips" / f"{name}.mp4", plugin="pyav")])
    a = np.asarray(meta["actions"], dtype="float64")
    er = a.max(0) - a.min(0)
    groups = component_groups(meta["action_names"], a.shape[1], meta.get("gripper_groups") or CM["groups"],
                              CM["default_group"])
    active = [segment_active_groups(a, s["start"], min(s["end"], len(a) - 1), groups, er,
                                    CM["threshold"], CM["edge"]) for s in meta["segments"]]
    return {**meta, "frames": frames, "nf": len(frames), "active": active}


def summarize(P):
    print(f"\n{P['label'].upper()}  ({P['repo']} ep{P['episode']}, {P['nf']} frames)")
    q = f"{int(P['quality'])}/5" if P["quality"] is not None else "-"
    sec = P["active_seconds"]
    spd = f"{sec:.1f}s active ({int(round((P['active_fraction'] or 0) * 100))}%)" if sec is not None else "-"
    print(f"  quality {q}   speed {spd}   subgoal frames {P['subgoals']}")
    for s, act in zip(P["segments"], P["active"]):
        tgt = f" -> {s['target']}" if s.get("target") else ""
        disp = act.replace("+", ", ") if act != "none" else "none"
        print(f"    f{s['start']:>3}-{s['end']:<3}  {(s['phase'] or '') + tgt:34s} active: {disp}")


def cur_segment(segs, frame):
    for i, s in enumerate(segs):
        if s["start"] <= frame <= s["end"]:
            return i
    return len(segs) - 1


def render_panel(P, frame_idx):
    segs, nf = P["segments"], P["nf"]
    panel = Image.new("RGB", (PANEL_W, 212), "white")
    d = ImageDraw.Draw(panel)
    panel.paste(_thumb(P["frames"][min(frame_idx, nf - 1)], VID_W, VID_H), (10, 10))
    d.rectangle([10, 10, 10 + VID_W, 10 + VID_H], outline=(200, 205, 212))
    x = VID_W + 26
    si = cur_segment(segs, frame_idx)
    s = segs[si]
    d.text((x, 10), P["label"].upper(), font=F_SM, fill=MUTED)
    lab = f"{s['phase']} → {s['target']}" if s.get("target") else (s.get("phase") or "")
    d.text((x, 26), lab, font=F_XL, fill=_hex(SEG_COLORS[si % len(SEG_COLORS)]))
    q = f"{int(P['quality'])}/5" if P["quality"] is not None else "-"
    act = (f"{P['active_seconds']:.1f}s ({int(round((P['active_fraction'] or 0) * 100))}% active)"
           if P["active_seconds"] is not None else "-")
    d.text((x, 54), f"quality {q}    speed: active {act}", font=F_MD, fill=INK)
    aset = P["active"][si] if si < len(P["active"]) else None
    adisp = aset.replace("+", ", ") if aset and aset != "none" else "none"
    d.text((x, 72), f"active components: {adisp}", font=F_MD, fill=INK)
    tl_x, tl_y, tl_w, tl_h = x, 98, PANEL_W - x - 20, 16
    for i, sg in enumerate(segs):
        a = tl_x + int(sg["start"] / max(1, nf - 1) * tl_w)
        b = tl_x + int(sg["end"] / max(1, nf - 1) * tl_w)
        d.rectangle([a, tl_y, b, tl_y + tl_h], fill=_hex(SEG_COLORS[i % len(SEG_COLORS)]))
    px = tl_x + int(frame_idx / max(1, nf - 1) * tl_w)
    d.line([px, tl_y - 3, px, tl_y + tl_h + 3], fill=(15, 18, 24), width=2)
    ky = 126
    d.text((x, ky), "subgoal keyframes  (selected end-of-sub-step frames, not generated)", font=F_SM, fill=GREEN)
    kx = x
    for i, fr in enumerate(P["subgoals"]):
        panel.paste(_thumb(P["frames"][min(fr, nf - 1)]), (kx, ky + 16))
        d.rectangle([kx, ky + 16, kx + THUMB_W, ky + 16 + THUMB_H], outline=_hex(SEG_COLORS[i % len(SEG_COLORS)]))
        kx += THUMB_W + 6
    return panel


def main():
    panels = [load_panel(n) for n in NAMES]
    for P in panels:
        summarize(P)
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
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True)
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
