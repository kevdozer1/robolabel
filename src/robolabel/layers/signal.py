"""L1 signal layer (PLAN 4.2 L1, V_LITE L1): deterministic, CPU only, no model call.

From ``observation.state`` and ``action`` alone it finds gripper closing and opening events, groups
them into grasp attempts with an outcome (hold, empty, slip, released), reads the robot's end state,
proposes candidate boundaries with IDs ``c1, c2, ...``, and plans keyframes and video windows for the
model layers.

Two gripper layouts are supported:

* ``so101`` (F1, F3, F4): ``action[5]`` is the commanded gripper position and ``observation.state[5]``
  the measured one, in the same units; larger means more open. The outcome uses the gap between
  them (PLAN 3.1): after a closing offset, a measured position that stays more open than the command
  means something blocks the fingers.
* ``libero`` (F2): ``action[6]`` is a binary command (-1 open, +1 close) and ``observation.state[6:8]``
  the two finger joints; the finger width is ``state[6] - state[7]``. The outcome uses where the
  width stalls: above the closed width plus a margin is a hold, at the closed width is empty.

Calibration comes from the dataset's ``meta/stats.json`` plus dev episodes only (never per episode),
see :func:`calibrate`. The output record is plain JSON with rounded floats, so identical input gives
byte-identical output (:func:`dumps_record`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CODE_VERSION = "l1-2026-09-27.1"


@dataclass(frozen=True)
class GripperLayout:
    name: str
    cmd_index: int
    meas_indices: tuple[int, ...]
    arm_indices: tuple[int, ...]
    arm_units: str
    close_sign: int  # sign of the command's change while closing
    outcome_rule: str  # "gap" or "stall"


LAYOUTS: dict[str, GripperLayout] = {
    "so101": GripperLayout("so101", 5, (5,), (0, 1, 2, 3, 4), "deg", -1, "gap"),
    "libero": GripperLayout("libero", 6, (6, 7), (0, 1, 2), "m", +1, "stall"),
}

FAMILY_LAYOUT = {"F1": "so101", "F3": "so101", "F4": "so101", "F2": "libero"}


@dataclass
class Calibration:
    """Per-dataset constants. Starting values from V_LITE L1; tuned on dev episodes only."""

    layout: str
    fps: float
    cmd_open: float
    cmd_closed: float
    meas_open: float
    meas_closed: float
    gap_threshold: float = 0.05  # share of the range (hold vs empty, slip collapse)
    velocity_threshold: float = 0.01  # share of the command range per frame (event runs)
    empty_window_s: float = 0.2  # empty if the gap falls below the threshold this soon after the offset
    early_collapse_s: float = 0.5  # a collapse this soon after the offset counts as empty, not slip
    stall_window_s: float = 0.5  # libero: width change below stall_eps over this window is a stall
    stall_eps: float = 0.01  # libero: share of the width range
    hold_margin: float = 0.05  # libero: hold if the stall is this share of the range above closed
    merge_gap_s: float = 0.25  # same-direction runs closer than this are one motion
    pause_speed: float = 0.0  # arm speed (units per second) below which the arm is pausing
    withdraw_threshold: float = 0.0  # arm displacement (units) after the last opening that counts as withdrawn
    end_state_band: float = 0.2  # gripper_open / gripper_closed within this share of an extreme
    source: str = ""

    @property
    def cmd_range(self) -> float:
        return abs(self.cmd_open - self.cmd_closed)

    @property
    def meas_range(self) -> float:
        return abs(self.meas_open - self.meas_closed)

    def frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))

    def min_run(self) -> int:
        """At least 3 frames at 30 fps, scaled with fps (at least 2)."""
        return max(2, int(round(3 * self.fps / 30.0)))


# ------------------------------------------------------------------------------------------------ inputs
def gripper_signals(state: np.ndarray, action: np.ndarray, layout: GripperLayout) -> dict[str, np.ndarray]:
    """Command, measured position (or finger width) and arm coordinates as float arrays."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    cmd = action[:, layout.cmd_index]
    if layout.name == "libero":
        meas = state[:, layout.meas_indices[0]] - state[:, layout.meas_indices[1]]
    else:
        meas = state[:, layout.meas_indices[0]]
    arm = state[:, list(layout.arm_indices)]
    return {"cmd": cmd, "meas": meas, "arm": arm}


def calibrate(stats: dict[str, Any], layout_name: str, fps: float,
              dev_arrays: list[tuple[np.ndarray, np.ndarray]] | None = None, source: str = "") -> Calibration:
    """Dataset calibration from ``meta/stats.json`` and dev episodes (state, action) only."""
    layout = LAYOUTS[layout_name]
    amin, amax = stats["action"]["min"], stats["action"]["max"]
    smin, smax = stats["observation.state"]["min"], stats["observation.state"]["max"]
    if layout.name == "libero":
        i, j = layout.meas_indices
        cal = Calibration(layout=layout.name, fps=fps, cmd_open=float(amin[layout.cmd_index]),
                          cmd_closed=float(amax[layout.cmd_index]),
                          meas_open=float(smax[i]) - float(smin[j]), meas_closed=float(smin[i]) - float(smax[j]))
    else:
        k = layout.cmd_index
        cal = Calibration(layout=layout.name, fps=fps, cmd_open=float(amax[k]), cmd_closed=float(amin[k]),
                          meas_open=float(smax[layout.meas_indices[0]]),
                          meas_closed=float(smin[layout.meas_indices[0]]))
    speeds: list[np.ndarray] = []
    maxima: list[tuple[float, float, float, float]] = []
    for state, action in dev_arrays or []:
        sig = gripper_signals(state, action, layout)
        if len(sig["arm"]) > 1:
            speeds.append(arm_speed(sig["arm"], fps))
            maxima.append((float(sig["cmd"].max()), float(sig["cmd"].min()),
                           float(sig["meas"].max()), float(sig["meas"].min())))
    if maxima:
        # Typical extremes from dev episodes (stats.json min/max carry outliers, for example F3's 52).
        cmax, cmin, mmax, mmin = (np.array(v) for v in zip(*maxima, strict=True))
        if layout.name == "libero":
            cal.meas_open = float(np.median(mmax))
            cal.meas_closed = float(mmin.min())  # fully closed on nothing (holds stall above it)
            cal.hold_margin = 0.015  # tuned on F2 dev: bowl rims stall 0.0033 and wider, empty closes creep to 0.0012
        else:
            cal.cmd_open, cal.cmd_closed = float(np.median(cmax)), float(np.median(cmin))
            cal.meas_open, cal.meas_closed = float(np.median(mmax)), float(np.median(mmin))
    if speeds:
        allv = np.concatenate(speeds)
        p90 = float(np.percentile(allv, 90))
        cal.pause_speed = round(0.2 * p90, 6)
        # withdrawn: after the last opening the arm moves at least as far as 1 s at the pause speed
        cal.withdraw_threshold = round(cal.pause_speed * 1.0, 6)
    else:
        cal.pause_speed = 20.0 if layout.arm_units == "deg" else 0.03
        cal.withdraw_threshold = 20.0 if layout.arm_units == "deg" else 0.03
    n = len(dev_arrays or [])
    cal.source = source or (f"extremes: median per-episode max and min over {n} dev episodes; "
                            "pause and withdraw from the same episodes" if n else "meta/stats.json min/max")
    return cal


def arm_speed(arm: np.ndarray, fps: float, smooth: int = 5) -> np.ndarray:
    """Smoothed arm speed (units per second), same length as ``arm`` (first value repeated)."""
    if len(arm) < 2:
        return np.zeros(len(arm))
    step = np.linalg.norm(np.diff(arm, axis=0), axis=1) * fps
    step = np.concatenate([step[:1], step])
    k = max(1, int(smooth))
    return np.convolve(step, np.ones(k) / k, mode="same")


# ------------------------------------------------------------------------------------------------ events
def gripper_runs(cmd: np.ndarray, cal: Calibration, close_sign: int) -> list[dict[str, Any]]:
    """Closing and opening runs of the command (probe P5 method).

    Velocity = the 3-frame centered moving average of the first difference. A run is at least
    ``cal.min_run()`` consecutive frames whose speed exceeds ``velocity_threshold`` times the command
    range, split where the direction changes. Velocity index k is the change from frame k to k+1, so a
    run over indices i..j-1 has onset frame i and offset frame j. Same-direction runs separated by at
    most ``merge_gap_s`` are merged.
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    if len(cmd) < 4 or cal.cmd_range <= 0:
        return []
    v = np.convolve(np.diff(cmd), np.ones(3) / 3.0, mode="same")
    thr = cal.velocity_threshold * cal.cmd_range
    direction = np.where(v * close_sign > thr, 1, np.where(v * close_sign < -thr, -1, 0))
    runs: list[dict[str, Any]] = []
    i, n = 0, len(direction)
    while i < n:
        d = direction[i]
        if d == 0:
            i += 1
            continue
        j = i
        while j < n and direction[j] == d:
            j += 1
        if j - i >= cal.min_run():
            runs.append({"type": "closing" if d > 0 else "opening", "onset": int(i), "offset": int(j)})
        i = j
    merged: list[dict[str, Any]] = []
    gap = cal.frames(cal.merge_gap_s)
    for r in runs:
        if merged and merged[-1]["type"] == r["type"] and r["onset"] - merged[-1]["offset"] <= gap:
            merged[-1]["offset"] = r["offset"]
        else:
            merged.append(dict(r))
    return merged


def _gap(sig: dict[str, np.ndarray], cal: Calibration, layout: GripperLayout) -> np.ndarray:
    """How much more open the measured gripper is than the command, as a share of the range."""
    if layout.name == "libero":
        return (sig["meas"] - cal.meas_closed) / max(cal.meas_range, 1e-9)
    sign = 1.0 if cal.cmd_open >= cal.cmd_closed else -1.0
    return sign * (sig["meas"] - sig["cmd"]) / max(cal.cmd_range, 1e-9)


def _first_sustained_below(values: np.ndarray, start: int, stop: int, thr: float, k: int) -> int | None:
    run = 0
    for f in range(max(start, 0), min(stop, len(values))):
        run = run + 1 if values[f] < thr else 0
        if run >= k:
            return f - k + 1
    return None


def _stall_frame(width: np.ndarray, start: int, stop: int, cal: Calibration) -> int | None:
    """First frame from which the width changes by less than stall_eps over stall_window_s."""
    w = cal.frames(cal.stall_window_s)
    eps = cal.stall_eps * cal.meas_range
    for f in range(max(start, 0), min(stop, len(width)) - w):
        seg = width[f : f + w + 1]
        if float(seg.max() - seg.min()) < eps:
            return f
    return None


def attempts_from_events(events: list[dict[str, Any]], sig: dict[str, np.ndarray], cal: Calibration,
                         layout: GripperLayout) -> list[dict[str, Any]]:
    """Group closing runs into attempts and give each an outcome.

    A closing run starts a new attempt when it is the first one or an opening run happened since the
    previous attempt started; otherwise it is a re-close inside the current attempt. The attempt ends
    at the next opening onset (or the last frame).
    """
    n = len(sig["cmd"])
    gap = _gap(sig, cal, layout)
    thr = cal.gap_threshold
    deb = max(2, cal.min_run())
    attempts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    opened_since = True
    for ev in events:
        if ev["type"] == "opening":
            if current is not None and current.get("opening_onset") is None:
                current["opening_onset"] = ev["onset"]
                current["opening_offset"] = ev["offset"]
            opened_since = True
            continue
        if current is None or opened_since:
            current = {"attempt_idx": len(attempts) + 1, "closing_onset": ev["onset"],
                       "closing_offset": ev["offset"], "recloses": [], "opening_onset": None,
                       "opening_offset": None}
            attempts.append(current)
            opened_since = False
        else:
            current["recloses"].append({"onset": ev["onset"], "offset": ev["offset"]})
    early = cal.frames(cal.early_collapse_s)
    for a in attempts:
        end = a["opening_onset"] if a["opening_onset"] is not None else n
        offsets = [a["closing_offset"]] + [r["offset"] for r in a["recloses"]]
        hold_frame = None
        empty_frame = None
        evaluated = False
        if layout.outcome_rule == "stall":
            width = sig["meas"]
            stall = _stall_frame(width, a["closing_onset"], end, cal)
            evaluated = stall is not None or end < n
            if stall is not None and gap[stall] >= cal.hold_margin:
                hold_frame = stall
            elif evaluated:
                empty_frame = stall if stall is not None else min(end, n) - 1
        else:
            for off in offsets:
                ideal_end = off + cal.frames(cal.empty_window_s) + 1
                below = _first_sustained_below(gap, off, min(ideal_end, end), thr, 1)
                if below is not None:
                    evaluated = True
                    empty_frame = below if empty_frame is None else empty_frame
                    continue
                if ideal_end <= min(end, n):  # the whole window ran without the fingers reaching the command
                    evaluated = True
                    hold_frame = off
                    break
                # window cut short by an opening or the episode end: this close cannot be judged
        slip_frame = None
        if hold_frame is not None:
            # libero: a slip means the width reaches the closed value (V_LITE L1), not just the hold margin
            margin = 0.005 if layout.outcome_rule == "stall" else thr
            slip_frame = _first_sustained_below(gap, hold_frame, end, margin, deb)
            if slip_frame is not None and layout.outcome_rule == "gap" and slip_frame - hold_frame <= early:
                empty_frame, hold_frame, slip_frame = slip_frame, None, None
        a["note"] = ""
        if hold_frame is not None and slip_frame is None and a["opening_onset"] is not None \
                and a["opening_onset"] - hold_frame < early:
            a["outcome"], a["failure_type"] = "aborted", "aborted"
            a["event_frame"] = int(a["opening_onset"])
            a["note"] = "the fingers opened again within 0.5 s of closing"
            hold_frame = None
        elif hold_frame is None and not evaluated and a["opening_onset"] is not None:
            a["outcome"], a["failure_type"] = "aborted", "aborted"
            a["event_frame"] = int(a["opening_onset"])
            a["note"] = "the fingers opened again before the close could hold anything"
        elif hold_frame is None and not evaluated:
            a["outcome"], a["failure_type"] = "unknown", "none"
            a["event_frame"] = int(a["closing_offset"])
            a["note"] = "the close was cut short by the episode end"
        elif hold_frame is None:
            a["outcome"], a["failure_type"] = "empty", "missed_grasp"
            a["event_frame"] = int(empty_frame) if empty_frame is not None else int(a["closing_offset"])
        elif slip_frame is not None:
            a["outcome"], a["failure_type"] = "slip", "slip"
            a["event_frame"] = int(slip_frame)
        elif a["opening_onset"] is not None:
            a["outcome"], a["failure_type"] = "released", "none"
            a["event_frame"] = int(a["opening_onset"])
        else:
            a["outcome"], a["failure_type"] = "hold", "none"
            a["event_frame"] = int(hold_frame)
        a["hold_frame"] = int(hold_frame) if hold_frame is not None else None
        a["end_frame"] = int(end - 1)
        a["gap_after_offset"] = round(float(gap[min(a["closing_offset"], n - 1)]), 4)
    return attempts


def split_rest_close(attempts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A last close on nothing after a release, with no opening after it, is the gripper going to rest
    (teleoperators close the gripper once the object is placed); it is not a grasp attempt."""
    if len(attempts) >= 2:
        last = attempts[-1]
        released_before = any(a["outcome"] == "released" for a in attempts[:-1])
        if last["outcome"] in ("empty", "unknown", "slip") and last["opening_onset"] is None and released_before:
            rest = dict(last)
            rest["note"] = "close after the last release with no opening after it: gripper going to rest"
            return attempts[:-1], [rest]
    return attempts, []


# ------------------------------------------------------------------------------------------------ outputs
def _first_moving(speed: np.ndarray, start: int, stop: int, thr: float, k: int) -> int | None:
    run = 0
    for f in range(max(start, 0), min(stop, len(speed))):
        run = run + 1 if speed[f] > thr else 0
        if run >= k:
            return f - k + 1
    return None


def end_state(attempts: list[dict[str, Any]], events: list[dict[str, Any]], sig: dict[str, np.ndarray],
              cal: Calibration) -> list[dict[str, Any]]:
    n = len(sig["meas"])
    last = n - 1
    holding = bool(attempts) and attempts[-1]["outcome"] == "hold"
    items = [{"predicate": "holding", "ref_object": "none", "value": holding, "basis": "signal",
              "confidence": "high", "frame": last}]
    open_share = (float(sig["meas"][-1]) - cal.meas_closed) / max(cal.meas_open - cal.meas_closed, 1e-9)
    band = cal.end_state_band
    if open_share >= 1.0 - band:
        pred, conf = "gripper_open", "high"
    elif open_share <= band:
        pred, conf = "gripper_closed", "high"
    else:
        pred, conf = ("gripper_open", "low") if open_share >= 0.5 else ("gripper_closed", "low")
    items.append({"predicate": pred, "ref_object": "none", "value": True, "basis": "signal",
                  "confidence": conf, "frame": last, "open_share": round(open_share, 4)})
    openings = [e for e in events if e["type"] == "opening"]
    if openings:
        start = min(int(openings[-1]["offset"]), last)
        disp = float(np.linalg.norm(sig["arm"][last] - sig["arm"][start]))
        withdrawn = disp > cal.withdraw_threshold
        note = f"arm moved {disp:.3f} between frame {start} and {last}"
    else:
        disp, withdrawn, note = 0.0, False, "no opening event, so no withdrawal after a release"
    items.append({"predicate": "withdrawn", "ref_object": "none", "value": bool(withdrawn), "basis": "signal",
                  "confidence": "low", "frame": last, "displacement": round(disp, 4), "note": note})
    return items


def candidates(attempts: list[dict[str, Any]], speed: np.ndarray, cal: Calibration) -> list[dict[str, Any]]:
    """Candidate boundaries (a boundary is the end frame of the earlier segment)."""
    n = len(speed)
    k = max(2, cal.min_run())
    out: list[dict[str, Any]] = []
    for a in attempts:
        out.append({"frame": max(0, a["closing_onset"] - 1), "transition": "approach->grasp",
                    "confidence": "high", "attempt_idx": a["attempt_idx"], "event": "closing onset"})
        if a["hold_frame"] is not None:
            stop = a["opening_onset"] if a["opening_onset"] is not None else n
            mv = _first_moving(speed, a["closing_offset"] + 1, stop, cal.pause_speed, k)
            if mv is not None and mv - 1 > a["closing_onset"]:
                out.append({"frame": mv - 1, "transition": "grasp->transport", "confidence": "low",
                            "attempt_idx": a["attempt_idx"], "event": "arm starts moving after the close"})
        if a["outcome"] == "released":
            out.append({"frame": max(0, a["opening_onset"] - 1), "transition": "transport->release",
                        "confidence": "high", "attempt_idx": a["attempt_idx"], "event": "opening onset"})
            mv = _first_moving(speed, a["opening_offset"] + 1, n, cal.pause_speed, k)
            if mv is not None and mv - 1 >= a["opening_onset"]:
                out.append({"frame": mv - 1, "transition": "release->retract", "confidence": "low",
                            "attempt_idx": a["attempt_idx"], "event": "arm starts moving after the opening"})
    out = [c for c in out if 0 <= c["frame"] < n - 1]
    out.sort(key=lambda c: (c["frame"], c["transition"]))
    dedup: list[dict[str, Any]] = []
    for c in out:
        if dedup and dedup[-1]["frame"] == c["frame"]:
            continue
        dedup.append(c)
    for i, c in enumerate(dedup, 1):
        c["candidate_id"] = f"c{i}"
    return dedup


def keyframe_plan(attempts: list[dict[str, Any]], n: int, cal: Calibration, max_frames: int = 8) -> list[int]:
    """Frame 0, closing onsets, closing offset + 0.25 s, opening onsets, opening offset + 0.25 s, last.

    Frames closer than 0.25 s are merged, keeping the more important one (first and last frame, then
    event onsets, then the settled frames 0.25 s after an offset). If more than ``max_frames`` remain,
    the first and last frames and the frames of the first and last attempts are kept, then the rest
    fill in time order.
    """
    last = n - 1
    q = cal.frames(0.25)
    cands: dict[int, tuple[int, int]] = {}  # frame -> (priority, attempt index)

    def add(frame: int, prio: int, owner: int) -> None:
        frame = int(min(max(frame, 0), last))
        old = cands.get(frame)
        if old is None or prio > old[0]:
            cands[frame] = (prio, owner)

    add(0, 3, 0)
    add(last, 3, 0)
    for a in attempts:
        idx = a["attempt_idx"]
        add(a["closing_onset"], 2, idx)
        add(a["closing_offset"] + q, 1, idx)
        if a.get("opening_onset") is not None:
            add(a["opening_onset"], 2, idx)
            add(a["opening_offset"] + q, 1, idx)
    kept: list[int] = []
    for f in sorted(cands):
        if kept and f - kept[-1] < q:
            if cands[f][0] > cands[kept[-1]][0] and cands[kept[-1]][0] < 3:
                kept[-1] = f
            continue
        kept.append(f)
    if len(kept) <= max_frames:
        return kept
    first_idx = attempts[0]["attempt_idx"] if attempts else 0
    last_idx = attempts[-1]["attempt_idx"] if attempts else 0
    must = [f for f in kept if cands[f][0] == 3 or cands[f][1] in (first_idx, last_idx)]
    out = sorted(must)[:max_frames]
    for f in kept:
        if len(out) >= max_frames:
            break
        if f not in out:
            out.append(f)
    return sorted(out)


def video_windows(events: list[dict[str, Any]], n: int, fps: float) -> list[list[int]]:
    w = max(1, int(round(fps)))
    return [[max(0, e["onset"] - w), min(n - 1, e["onset"] + w)] for e in events]


def run_l1(state: np.ndarray, action: np.ndarray, cal: Calibration, *, episode_key: str = "",
           family: str = "") -> dict[str, Any]:
    """The full L1 record for one episode."""
    layout = LAYOUTS[cal.layout]
    sig = gripper_signals(state, action, layout)
    n = len(sig["cmd"])
    events = gripper_runs(sig["cmd"], cal, layout.close_sign)
    attempts, rest_closes = split_rest_close(attempts_from_events(events, sig, cal, layout))
    speed = arm_speed(sig["arm"], cal.fps)
    cands = candidates(attempts, speed, cal)
    return {
        "code_version": CODE_VERSION,
        "episode_key": episode_key,
        "family": family,
        "layout": layout.name,
        "fps": cal.fps,
        "num_frames": n,
        "calibration": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(cal).items()},
        "events": events,
        "attempts": attempts,
        "rest_closes": rest_closes,
        "end_state": end_state(attempts, events, sig, cal),
        "candidates": cands,
        "keyframes": keyframe_plan(attempts, n, cal),
        "windows": video_windows(events, n, cal.fps),
    }


def dumps_record(record: dict[str, Any]) -> str:
    """Canonical JSON line for an L1 record (sorted keys, no spaces), byte-identical on identical input."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ text for prompts
def attempt_summary(record: dict[str, Any]) -> str:
    """One line, for example 'attempt 1: missed grasp at frame 88; attempt 2: held from 120, released at 216'."""
    parts = []
    for a in record.get("attempts", []):
        i = a["attempt_idx"]
        if a["outcome"] == "empty":
            parts.append(f"attempt {i}: missed grasp at frame {a['event_frame']}")
        elif a["outcome"] == "aborted":
            parts.append(f"attempt {i}: closed at frame {a['closing_onset']} and opened again at {a['event_frame']}")
        elif a["outcome"] == "unknown":
            parts.append(f"attempt {i}: closing at frame {a['closing_onset']}, outcome not measurable")
        elif a["outcome"] == "slip":
            parts.append(f"attempt {i}: held from {a['hold_frame']}, lost the grip at frame {a['event_frame']}")
        elif a["outcome"] == "released":
            parts.append(f"attempt {i}: held from {a['hold_frame']}, released at {a['opening_onset']}")
        else:
            parts.append(f"attempt {i}: held from {a['hold_frame']} to the end")
    return "; ".join(parts) if parts else "no grasp attempt detected by the gripper signal"


def candidate_lines(record: dict[str, Any]) -> list[str]:
    fps = float(record.get("fps") or 1.0)
    lines = []
    for c in record.get("candidates", []):
        what = {"approach->grasp": "fingers start closing at frame {f}",
                "grasp->transport": "arm starts moving after the grasp at frame {f}",
                "transport->release": "fingers start opening at frame {f}",
                "release->retract": "arm starts moving after the release at frame {f}"}[c["transition"]]
        exact = "measured by the robot, exact" if c["confidence"] == "high" else "measured by the robot, approximate"
        f = c["frame"] + 1
        lines.append(f"{c['candidate_id']}: {what.format(f=f)} ({f / fps:.2f} s; {exact}); "
                     f"a boundary here ends the earlier segment at frame {c['frame']}")
    return lines


def attempt_lines(record: dict[str, Any]) -> list[str]:
    out = []
    for a in record.get("attempts", []):
        i = a["attempt_idx"]
        if a["outcome"] == "empty":
            out.append(f"attempt {i}: fingers closed at frame {a['closing_onset']} and closed on nothing "
                       f"(missed grasp, evident at frame {a['event_frame']})")
        elif a["outcome"] == "aborted":
            out.append(f"attempt {i}: fingers started closing at frame {a['closing_onset']} and opened again at "
                       f"frame {a['event_frame']} before holding anything")
        elif a["outcome"] == "unknown":
            out.append(f"attempt {i}: fingers started closing at frame {a['closing_onset']}; the signal cannot "
                       "tell whether anything was grasped")
        elif a["outcome"] == "slip":
            out.append(f"attempt {i}: grasp held from frame {a['hold_frame']}, grip lost at frame {a['event_frame']}")
        elif a["outcome"] == "released":
            out.append(f"attempt {i}: grasp held from frame {a['hold_frame']}, fingers opened at frame "
                       f"{a['opening_onset']}")
        else:
            out.append(f"attempt {i}: grasp held from frame {a['hold_frame']} until the end")
    return out


def end_state_lines(record: dict[str, Any]) -> list[str]:
    out = []
    for it in record.get("end_state", []):
        conf = "" if it.get("confidence") == "high" else " (low confidence)"
        if it["predicate"] == "holding":
            out.append(f"holding an object at the last frame: {'yes' if it['value'] else 'no'}{conf}")
        elif it["predicate"] in ("gripper_open", "gripper_closed"):
            out.append(f"gripper at the last frame: {it['predicate'].split('_')[1]}{conf}")
        elif it["predicate"] == "withdrawn":
            out.append(f"arm moved away after the last release: {'yes' if it['value'] else 'no'}{conf}")
    return out
