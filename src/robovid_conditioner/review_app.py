"""Streamlit calibration GUI (``robovid_conditioner review``).

Play a clip, see the VLM labels, edit the quality score / mistake / reason,
accept or correct subtask boundaries, and mark subgoal frames. Edits are written
to the gold file's ``gold`` block via :func:`robovid_conditioner.gold.update_episode_review`;
the VLM ``auto`` block is never touched. A live reliability readout sits in the
sidebar so you can watch agreement as you review.

Run it with ``robovid_conditioner review --annotations <dir> --gold <file>`` (optionally
``--source/--target`` to show real clip frames). Requires the ``review`` extra.
"""

from __future__ import annotations

import argparse
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    known, _ = parser.parse_known_args()
    return known


def _load_episode_frames(source: str | None, target: str | None):
    """Return a dict episode_id -> Episode for showing frames, or {} if no source."""
    if not source or not target:
        return {}
    from robovid_conditioner.adapters import build_source

    try:
        src = build_source(source, target)
        return {ep.episode_id: ep for ep in src}
    except Exception as exc:  # noqa: BLE001 - frames are a convenience; degrade gracefully
        import streamlit as st

        st.warning(f"Could not load frames from source ({exc}). Reviewing labels without clip preview.")
        return {}


def _keyframe_strip(episode, n: int = 6) -> list:
    idxs = [int(round(x)) for x in _linspace(0, episode.num_frames - 1, min(n, episode.num_frames))]
    return [episode.frame(i) for i in idxs], idxs


def _linspace(a: int, b: int, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def run() -> None:  # pragma: no cover - exercised interactively, not in CI
    import streamlit as st

    from robovid_conditioner.gold import load_or_sync_gold, update_episode_review
    from robovid_conditioner.reliability import reliability_report

    args = _parse_args()
    st.set_page_config(page_title="robovid_conditioner review", layout="wide")
    st.title("robovid_conditioner · human calibration")

    gold = load_or_sync_gold(args.annotations, args.gold)
    episodes = gold["episodes"]
    frames_by_id = _cached_frames(args.source, args.target)

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    idx = max(0, min(st.session_state.idx, len(episodes) - 1))
    entry = episodes[idx]
    auto = entry["auto"]
    gold_block = entry["gold"]

    # ----- sidebar: navigation + live reliability ----- #
    with st.sidebar:
        st.header(f"Episode {idx + 1} / {len(episodes)}")
        st.caption(str(entry.get("episode_id")))
        cols = st.columns(2)
        if cols[0].button("◀ Prev", disabled=idx == 0):
            st.session_state.idx = idx - 1
            st.rerun()
        if cols[1].button("Next ▶", disabled=idx >= len(episodes) - 1):
            st.session_state.idx = idx + 1
            st.rerun()
        st.divider()
        report = reliability_report(args.gold)
        st.metric("Reviewed", f"{report['reviewed_episode_count']} / {report['episode_count']}")
        st.write({
            "quality exact": report["quality_exact_agreement"],
            "quality ±1": report["quality_within_one_agreement"],
            "boundary IoU": report["subtask_boundary_temporal_iou_mean"],
            "subgoal match": report["subgoal_frame_agreement"],
        })

    st.subheader(entry.get("task") or "(no task string)")

    # ----- clip preview ----- #
    episode = frames_by_id.get(str(entry.get("episode_id")))
    if episode is not None:
        strip, idxs = _keyframe_strip(episode)
        st.image(strip, caption=[f"frame {i}" for i in idxs], width=140)

    left, right = st.columns(2)

    # ----- metadata review ----- #
    with left:
        st.markdown("**Episode quality**")
        st.caption(f"VLM: quality={auto['metadata'].get('quality')} "
                   f"mistake={auto['metadata'].get('mistake')} — {auto['metadata'].get('reason')}")
        default_q = gold_block["metadata"].get("quality") or auto["metadata"].get("quality") or 3
        quality = st.slider("Your quality (1-5)", 1, 5, int(default_q))
        mistake = st.checkbox("Mistake", value=bool(gold_block["metadata"].get("mistake")
                                                    if gold_block["metadata"].get("mistake") is not None
                                                    else auto["metadata"].get("mistake")))
        reason = st.text_input("Reason", value=gold_block["metadata"].get("reason") or "")
        accept_meta = st.checkbox("Accept VLM metadata as-is", value=False)

    # ----- subtask + subgoal review ----- #
    with right:
        st.markdown("**Subtasks** (uncheck Accept to correct boundaries)")
        gold_subtasks: list[dict[str, Any]] = []
        for s in auto["subtasks"]:
            seg = s["segment_idx"]
            st.caption(f"[{seg}] {s['subtask_text']}  (frames {s['start_frame']}–{s['end_frame']})")
            c = st.columns([1, 1, 1])
            accept = c[0].checkbox("Accept", value=True, key=f"acc_{idx}_{seg}")
            start = c[1].number_input("start", value=int(s["start_frame"]), key=f"st_{idx}_{seg}", step=1)
            end = c[2].number_input("end", value=int(s["end_frame"]), key=f"en_{idx}_{seg}", step=1)
            gold_subtasks.append({"segment_idx": seg, "accept_auto": accept,
                                  "start_frame": int(start), "end_frame": int(end)})

        st.markdown("**Subgoal frames**")
        gold_subgoals: list[dict[str, Any]] = []
        for s in auto["subgoals"]:
            seg = s["segment_idx"]
            frame = st.number_input(f"subgoal seg {seg} frame", value=int(s.get("frame_idx") or 0),
                                    key=f"sg_{idx}_{seg}", step=1)
            gold_subgoals.append({"segment_idx": seg, "frame_idx": int(frame)})

    notes = st.text_area("Review notes", value=entry.get("review_notes", ""))

    if st.button("💾 Save review and next", type="primary"):
        update_episode_review(
            args.gold, entry["episode_id"],
            quality=quality, mistake=mistake, reason=reason, accept_auto_metadata=accept_meta,
            gold_subtasks=gold_subtasks, gold_subgoals=gold_subgoals, review_notes=notes,
        )
        st.session_state.idx = min(idx + 1, len(episodes) - 1)
        st.rerun()


def _cached_frames(source: str | None, target: str | None):
    import streamlit as st

    @st.cache_resource(show_spinner="Loading clip frames…")
    def _load(src, tgt):
        return _load_episode_frames(src, tgt)

    return _load(source, target)


if __name__ == "__main__":  # pragma: no cover
    run()
