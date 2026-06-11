"""Generate the README figures from a real annotated episode.

Produces two clean, low-clutter figures:

1. ``annotation_overview.png`` — a filmstrip of one real episode with the three
   things robolabel adds drawn directly on it: the subtask timeline,
   the quality/mistake badge, and the subgoal keyframes.
2. ``pipeline.png`` — a one-line pipeline strip.

Usage:
    python scripts/make_readme_figures.py \
        --annotations C:/path/to/so101_gemini --episode 0 \
        --target lerobot/svla_so101_pickplace --camera observation.images.side \
        --out docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from robolabel.adapters.lerobot import LeRobotAdapter  # noqa: E402
from robolabel.schema import episode_records, read_annotations  # noqa: E402

# A calm, high-contrast palette (color-blind friendly-ish), one per subtask.
SEGMENT_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
INK = "#222222"


def _episode_frames(target: str, camera: str, episode: int, k: int) -> tuple[list[np.ndarray], int]:
    src = LeRobotAdapter(target, camera_key=camera, episodes=[episode])
    ep = next(iter(src))
    idxs = [int(round(x)) for x in np.linspace(0, ep.num_frames - 1, k)]
    return [ep.frame(i) for i in idxs], ep.num_frames, idxs


def make_overview(annotations: str, target: str, camera: str, episode: int, out_dir: Path, k: int = 6) -> Path:
    df = read_annotations(annotations)
    rec = episode_records(df, str(episode))
    subtasks = rec["subtasks"]
    subgoals = sorted(int(s["subgoal_frame_idx"]) for s in rec["subgoals"])
    meta = rec["metadata"]
    frames, num_frames, frame_idxs = _episode_frames(target, camera, episode, k)
    last = num_frames - 1

    fig = plt.figure(figsize=(12, 4.6), dpi=170)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.55], hspace=0.10)

    # ---- filmstrip ----
    ax_film = fig.add_subplot(gs[0])
    montage = _montage(frames)
    ax_film.imshow(montage)
    ax_film.set_xticks([])
    ax_film.set_yticks([])
    for spine in ax_film.spines.values():
        spine.set_visible(False)
    # frame-index captions under each tile
    tile_w = montage.shape[1] / len(frames)
    for j, fi in enumerate(frame_idxs):
        ax_film.text((j + 0.5) * tile_w, montage.shape[0] + 10, f"frame {fi}",
                     ha="center", va="top", fontsize=8, color="#888888")

    # quality badge (top-right, on the film axes)
    q = _to_int(meta.get("quality"))
    mistake = bool(meta.get("mistake"))
    badge = f"quality {q}/5   {'✗ mistake' if mistake else '✓ no mistake'}"
    ax_film.text(0.995, 1.06, badge, transform=ax_film.transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#C44E52" if mistake else "#55A868", ec="none"))

    # ---- subtask timeline (numbered blocks + legend, never overlapping) ----
    ax_tl = fig.add_subplot(gs[1])
    ax_tl.set_xlim(0, last)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_yticks([])
    for spine in ax_tl.spines.values():
        spine.set_visible(False)
    handles = []
    for i, s in enumerate(subtasks):
        s0, s1 = int(s["start_frame"]), int(s["end_frame"])
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        ax_tl.add_patch(mpatches.FancyBboxPatch(
            (s0, 0.42), max(1, s1 - s0), 0.34, boxstyle="round,pad=0.0,rounding_size=2",
            fc=color, ec="white", lw=1.5))
        ax_tl.text((s0 + s1) / 2, 0.59, str(i + 1), ha="center", va="center",
                   fontsize=10, color="white", fontweight="bold")
        handles.append(mpatches.Patch(color=color, label=f"{i + 1}  {_short(str(s['subtask_text']), 26)}"))
    # subgoal markers sit just ABOVE the bar, clear of any text
    for sg in subgoals:
        ax_tl.plot([sg], [0.86], marker="v", markersize=8, color=INK, clip_on=False)
    ax_tl.set_xlabel("frame index   ( ▼ = subgoal keyframe )", fontsize=9, color="#666666")
    ax_tl.tick_params(labelsize=8, colors="#666666")
    ax_tl.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.45),
                 ncol=min(len(handles), 5), frameon=False, fontsize=8.5, handlelength=1.1,
                 columnspacing=1.2, title="subtasks", title_fontsize=8.5)

    fig.suptitle("robolabel: one raw LeRobot episode → subtask boundaries · quality/mistake · subgoal keyframes",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, f"task: “{rec['task']}”   ·   VLM: Gemini 2.5 Flash   ·   measure agreement with `robolabel reliability`",
             ha="center", fontsize=9, color="#666666")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "annotation_overview.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_pipeline(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 1.5), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # (label, facecolor, textcolor) — dark fill only where text is white.
    boxes = [
        ("LeRobot\nepisode", "#E8EEF6", INK),
        ("annotate\nyour VLM", "#4C72B0", "white"),
        ("subtasks ·\nquality · subgoals", "#E8EEF6", INK),
        ("review\ncalibrate", "#55A868", "white"),
        ("reliability\nmeasured", "#E8EEF6", INK),
    ]
    n = len(boxes)
    centers = [(i + 0.5) / n for i in range(n)]
    for i, ((label, fc, tc), cx) in enumerate(zip(boxes, centers, strict=False)):
        ax.text(cx, 0.5, label, ha="center", va="center", fontsize=10, color=tc, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.6", fc=fc, ec="#B8C4D0", lw=1.2))
        if i < n - 1:
            ax.annotate("", xy=(centers[i + 1] - 0.075, 0.5), xytext=(cx + 0.075, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pipeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _montage(frames: list[np.ndarray], sep: int = 4) -> np.ndarray:
    h = min(f.shape[0] for f in frames)
    tiles = []
    for f in frames:
        scale = h / f.shape[0]
        from PIL import Image
        im = Image.fromarray(f).resize((int(f.shape[1] * scale), h))
        tiles.append(np.asarray(im))
    white = np.full((h, sep, 3), 255, dtype=np.uint8)
    out = []
    for i, t in enumerate(tiles):
        out.append(t)
        if i < len(tiles) - 1:
            out.append(white)
    return np.concatenate(out, axis=1)


def _short(text: str, n: int = 28) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--camera", default="observation.images.side")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--keyframes", type=int, default=6)
    args = p.parse_args()
    out = Path(args.out)
    print("overview:", make_overview(args.annotations, args.target, args.camera, args.episode, out, args.keyframes))
    print("pipeline:", make_pipeline(out))


if __name__ == "__main__":
    main()
