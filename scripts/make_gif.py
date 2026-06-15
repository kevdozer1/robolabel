"""Render docs/figures/grounded_annotations.gif — one looped GIF, three task panels.

Per panel, synced to the playing video: the current `phase -> target` sub-step label, a
segment timeline with a playhead, the episode quality score, the control line
(control_modality + the current segment's active_dof), and a row of the REAL end-of-sub-step
subgoal keyframes (labeled "selected keyframes — not generated") plus, where available, the
retrieved same-phase subgoals from other episodes (labeled as such).

Honesty: every image shown is a real frame from the dataset; subgoals are SELECTED, never
generated; control is read from the action stream. Zero API (cached frames + parquet only).
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

# task -> (label, annotations dir, episode id, repo, camera)
PANELS = [
    ("Pick-place", "probe_pickplace/s2", "7", "lerobot/svla_so101_pickplace", "observation.images.side"),
    ("Pour", "probe_pour/s2_open", "5", "Ishah8840/so101_pouring", "observation.images.front"),
    ("Fold", "probe_fold/s2_open", "4", "the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead"),
]
OUT = Path("docs/figures/grounded_annotations.gif")
STEPS = 36                       # GIF frames (each episode plays over this many steps)
PANEL_W, VID_W, VID_H = 900, 190, 150
THUMB_W, THUMB_H = 66, 50
SEG_COLORS = ["#2563eb", "#e8752a", "#59a14f", "#b4536b", "#8c6bb1", "#3aa0a0", "#9aa5b1"]
INK, MUTED, GREEN, BLUE = (24, 28, 36), (110, 118, 130), (40, 120, 60), (37, 99, 235)


def _font(size: int, bold: bool = False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_SM, F_MD, F_LG, F_XL = _font(12), _font(14), _font(16, True), _font(20, True)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _thumb(arr, w=THUMB_W, h=THUMB_H) -> Image.Image:
    img = Image.fromarray(np.asarray(arr).astype("uint8")).convert("RGB")
    return img.resize((w, h))


def load_panel(label, d, eid, repo, cam):
    rec = episode_records(read_annotations(d), eid)
    src = build_source("lerobot", repo, camera_key=cam, episodes=list(range(8)))
    epmap = {e.episode_id: e for e in src}
    ep = epmap[eid]
    segs = [{"start": int(s["start_frame"]), "end": int(s["end_frame"]),
             "phase": s.get("phase"), "target": s.get("target"), "dof": s.get("active_dof")}
            for s in rec["subtasks"]]
    sgs = {}
    for s in rec["subgoals"]:
        rid = s.get("retrieved_subgoal_episode_id")
        rf = s.get("retrieved_subgoal_frame_idx")
        rid = None if (rid is None or isinstance(rid, float)) else str(rid)
        rf = None if (rf is None or (isinstance(rf, float) and math.isnan(rf))) else int(rf)
        sgs[int(s["segment_idx"])] = {"real": int(s["subgoal_frame_idx"]), "rep": rid, "rf": rf}
    return {"label": label, "ep": ep, "epmap": epmap, "segs": segs, "subgoals": sgs,
            "quality": rec["metadata"].get("quality"), "modality": rec["metadata"].get("control_modality"),
            "nf": ep.num_frames}


def cur_segment(segs, frame):
    for i, s in enumerate(segs):
        if s["start"] <= frame <= s["end"]:
            return i
    return len(segs) - 1


def render_panel(P, frame_idx) -> Image.Image:
    segs, nf = P["segs"], P["nf"]
    has_ret = any(v.get("rf") is not None for v in P["subgoals"].values())
    ph = 250 if has_ret else 196
    panel = Image.new("RGB", (PANEL_W, ph), "white")
    d = ImageDraw.Draw(panel)
    # video
    panel.paste(_thumb(P["ep"].frame(frame_idx), VID_W, VID_H), (10, 10))
    d.rectangle([10, 10, 10 + VID_W, 10 + VID_H], outline=(200, 205, 212))
    x = VID_W + 26
    si = cur_segment(segs, frame_idx)
    s = segs[si]
    d.text((x, 10), P["label"].upper(), font=F_SM, fill=MUTED)
    lab = f"{s['phase']} → {s['target']}" if s.get("target") else (s.get("phase") or "")
    d.text((x, 26), lab, font=F_XL, fill=_hex(SEG_COLORS[si % len(SEG_COLORS)]))
    q = P["quality"]
    qs = f"{int(q)}/5" if q is not None and not (isinstance(q, float) and math.isnan(q)) else "–"
    d.text((x, 54), f"quality {qs}    control: {P['modality'] or '–'}    active_dof: {s.get('dof') or '–'}",
           font=F_MD, fill=INK)
    # timeline + playhead
    tl_x, tl_y, tl_w, tl_h = x, 80, PANEL_W - x - 20, 16
    for i, sg in enumerate(segs):
        a = tl_x + int(sg["start"] / max(1, nf - 1) * tl_w)
        b = tl_x + int(sg["end"] / max(1, nf - 1) * tl_w)
        d.rectangle([a, tl_y, b, tl_y + tl_h], fill=_hex(SEG_COLORS[i % len(SEG_COLORS)]))
    px = tl_x + int(frame_idx / max(1, nf - 1) * tl_w)
    d.line([px, tl_y - 3, px, tl_y + tl_h + 3], fill=(15, 18, 24), width=2)
    # real keyframes row
    ky = 108
    d.text((x, ky), "selected keyframes  (real end-of-sub-step frames — not generated)", font=F_SM, fill=GREEN)
    kx = x
    for i, sg in enumerate(segs):
        fr = P["subgoals"].get(i, {}).get("real", sg["end"])
        panel.paste(_thumb(P["ep"].frame(fr)), (kx, ky + 16))
        d.rectangle([kx, ky + 16, kx + THUMB_W, ky + 16 + THUMB_H],
                    outline=_hex(SEG_COLORS[i % len(SEG_COLORS)]))
        kx += THUMB_W + 6
    # retrieved row (where available)
    if has_ret:
        ry = ky + 16 + THUMB_H + 8
        d.text((x, ry), "retrieved subgoals  (same phase, other episodes — selected, not generated)",
               font=F_SM, fill=BLUE)
        rx = x
        for i in range(len(segs)):
            info = P["subgoals"].get(i, {})
            if info.get("rf") is not None and info.get("rep") in P["epmap"]:
                panel.paste(_thumb(P["epmap"][info["rep"]].frame(info["rf"])), (rx, ry + 16))
                d.rectangle([rx, ry + 16, rx + THUMB_W, ry + 16 + THUMB_H], outline=BLUE)
                d.text((rx + 2, ry + 16 + THUMB_H - 12), f"ep{info['rep']}", font=F_SM, fill=(255, 255, 255))
            else:
                d.rectangle([rx, ry + 16, rx + THUMB_W, ry + 16 + THUMB_H], outline=(210, 214, 220))
                d.text((rx + 6, ry + 30), "n/a", font=F_SM, fill=MUTED)
            rx += THUMB_W + 6
    return panel


def main() -> int:
    print("loading panels (cached frames)...", file=sys.stderr)
    panels = [load_panel(*p) for p in PANELS]
    cap_h = 26
    panel_imgs0 = [render_panel(P, 0) for P in panels]
    total_h = sum(im.height for im in panel_imgs0) + cap_h
    frames = []
    for t in range(STEPS):
        canvas = Image.new("RGB", (PANEL_W, total_h), "white")
        y = 0
        for P in panels:
            fi = int(round(t / (STEPS - 1) * (P["nf"] - 1)))
            pim = render_panel(P, fi)
            canvas.paste(pim, (0, y))
            y += pim.height
        d = ImageDraw.Draw(canvas)
        d.text((10, y + 6), "robolabel grounded annotations — subgoals are real selected frames (never generated); "
               "control read from the action stream.", font=F_SM, fill=MUTED)
        frames.append(canvas)
        print(f"  step {t + 1}/{STEPS}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True)
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB, {len(frames)} frames, {total_h}x{PANEL_W})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
