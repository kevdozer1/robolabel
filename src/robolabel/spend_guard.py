"""Spend guard: reserve before every paid HTTP attempt, reconcile after, never cross a cap.

Adapted from statebench's ``spend_guard_v13`` for robolabel's paid calls. Every attempt (including
retries) reserves its worst case (input estimate at the input price plus ``max_tokens`` at the output
price) and is refused before anything is sent when the reservation could take the committed total
past any of:

* the run cap, and ``available_at_start - balance_floor`` (the balance floor),
* the cap of the bucket the call is charged to (sweep, legacy_arm, debug, stretch),
* the per-model cap (sweep bucket only).

"Committed" is reconciled spend plus unreconciled spend plus open reservations. After a response,
``reconcile`` replaces the reservation with the call's own ``usage.cost``. A timeout, dropped
connection or HTTP error can still be billed, so ``lost`` keeps that attempt's reservation as spent,
marked unreconciled, until ``reconcile_lost`` gets the real figure from the generation endpoint
(``lost_with_generation`` lists the lost attempts that have a generation id, also after a restart). A
lost reservation is never released.

Every event is appended to a JSONL ledger (flushed and fsynced). On restart the ledger is replayed:
reservations that were still open when the process stopped count as spent (unreconciled).

One process at a time may reserve against a ledger. A guard that may reserve creates the lock file
``<ledger>.lock`` (O_CREAT | O_EXCL, holding the pid and the start time) before it replays, and
raises :class:`LedgerLocked` when a live process holds it. A lock whose process is gone is taken over
with a ledger note. The lock is released by ``close()`` and at exit. ``SpendGuard(read_only=True)``
replays without writing anything and refuses every reservation (for status and reports).

A drift check compares the change in the key's own usage (an injected callable, for example GET
/api/v1/key) with the ledger every 10 reconciled calls or 5 minutes; open reservations count in the
margin, since OpenRouter may have billed a call that is not reconciled here yet. While the check
pauses for its re-check, reservations wait for it and then proceed or refuse; a persistent excess
stops paid calls for the night. ``stop`` also serves the other stop rules (HTTP 402 for credits, the
balance floor). Keys and headers never reach the ledger.
"""

from __future__ import annotations

import atexit
import datetime as dt
import itertools
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class SpendRefused(RuntimeError):
    """The guard refused a reservation. Skip the call, log it, never retry it."""


class PaidCallsStopped(RuntimeError):
    """Paid calls are stopped for the night (drift, credits, floor)."""


class LedgerLocked(RuntimeError):
    """Another live process holds the ledger's lock file, so this guard may not reserve against it."""


@dataclass
class GuardConfig:
    run_cap: float
    available_at_start: float
    balance_floor: float
    bucket_caps: dict[str, float] = field(default_factory=dict)
    model_caps: dict[str, float] = field(default_factory=dict)  # sweep bucket only
    model_cap_bucket: str = "sweep"
    drift_every_calls: int = 10
    drift_every_s: float = 300.0
    drift_min_usd: float = 0.10
    drift_share: float = 0.10
    drift_wait_s: float = 60.0

    @property
    def effective_cap(self) -> float:
        return min(self.run_cap, self.available_at_start - self.balance_floor)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------- lock file
_UNREADABLE_LOCK_GRACE_S = 10.0  # an empty or partial lock this young may still be being written


def _pid_alive(pid: int, since: float | None = None) -> bool:
    """True when process ``pid`` is running.

    POSIX asks ``os.kill(pid, 0)``. Windows asks OpenProcess and GetExitCodeProcess; there a process
    created after ``since`` (the lock's start time, Unix seconds) is a reused pid, not the lock's holder.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid, since)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists but belongs to another user
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int, since: float | None) -> bool:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    k32.GetProcessTimes.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    k32.CloseHandle.restype = wintypes.BOOL
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: it exists; anything else: no such process
    try:
        code = wintypes.DWORD()
        if k32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:  # STILL_ACTIVE
            return False
        if since is not None:
            times = [wintypes.FILETIME() for _ in range(4)]
            if k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime  # 100 ns since 1601
                created = ticks / 1e7 - 11644473600.0
                if created > float(since) + 2.0:
                    return False
        return True
    finally:
        k32.CloseHandle(handle)


class SpendGuard:
    """Thread-safe reserve / reconcile ledger. See the module docstring for the rules."""

    def __init__(self, config: GuardConfig, ledger_path: str | Path, *,
                 key_usage: Callable[[], float | None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 read_only: bool = False):
        self.config = config
        self.ledger_path = Path(ledger_path)
        self.lock_path = Path(str(self.ledger_path) + ".lock")
        self.read_only = bool(read_only)
        self.key_usage = key_usage
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._resume = threading.Condition(self._lock)  # reservations wait here while the drift check pauses
        self._ids = itertools.count(1)
        self.open: dict[int, dict[str, Any]] = {}
        self.unreconciled: dict[int, dict[str, Any]] = {}
        self.spent_by: dict[tuple[str, str], float] = {}  # (bucket, model) -> reconciled usd
        self.unrec_by: dict[tuple[str, str], float] = {}
        self.stopped: str | None = None
        self.key_usage_at_start: float | None = None
        self.refusals = 0
        self._since_check = 0
        self._last_check = clock()
        self._paused = False
        self._checking = False
        self._closed = False
        self._lock_fd: int | None = None
        self._lock_token: str | None = None
        # Set by a provider when OpenRouter's in-flight budget is full (HTTP 402); the scheduler runs one
        # job at a time while it is set (budget.yaml http_402). Cleared when a call sent meanwhile succeeds.
        self.in_flight_backoff = threading.Event()
        note = None if self.read_only else self._take_file_lock()
        try:
            self._replay(note)
        except BaseException:  # a guard that failed to start holds no lock
            self._release_file_lock()
            atexit.unregister(self._close_at_exit)
            raise

    # ------------------------------------------------------------------ lock file
    def _read_file_lock(self) -> tuple[dict[str, Any] | None, str | None]:
        """(parsed lock, raw text); (None, None) when there is no lock file."""
        try:
            raw = self.lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, None
        except OSError:
            return None, ""
        try:
            held = json.loads(raw)
        except json.JSONDecodeError:
            return None, raw
        return (held if isinstance(held, dict) else None), raw

    def _lock_is_live(self, held: dict[str, Any] | None) -> bool:
        if held is None:  # empty or partial: a live process may be between creating and writing it
            try:
                age = time.time() - self.lock_path.stat().st_mtime
            except OSError:
                return False
            return age < _UNREADABLE_LOCK_GRACE_S
        pid = held.get("pid")
        if not isinstance(pid, int):
            return False
        since = held.get("started_unix")
        return _pid_alive(pid, float(since) if isinstance(since, (int, float)) else None)

    def _locked_error(self, held: dict[str, Any] | None) -> LedgerLocked:
        if held is not None and held.get("pid") == os.getpid():
            return LedgerLocked(f"{self.lock_path.name} is held by this process: another SpendGuard on this ledger "
                                "is still open. Close it first, or read with SpendGuard(read_only=True).")
        who = "an unreadable lock" if held is None else \
            f"pid {held.get('pid')} (lock taken {held.get('started_utc')})"
        return LedgerLocked(f"{self.lock_path.name} is held by {who}: another process is reserving against this "
                            "ledger. Read it with SpendGuard(read_only=True), or wait for that process to end. Delete "
                            "the lock file only if that process is not a robolabel run.")

    def _take_file_lock(self) -> dict[str, Any] | None:
        """Create the lock file or raise LedgerLocked. Returns a ledger note when a stale lock was taken over."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        info = {"pid": os.getpid(), "started_utc": _utc(), "started_unix": round(time.time(), 3), "token": token}
        stale: list[dict[str, Any]] = []
        for _ in range(5):
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                held, raw = self._read_file_lock()
                if raw is None:  # removed since the open: try again
                    continue
                if self._lock_is_live(held):
                    raise self._locked_error(held) from None
                # Stale. Remove it only if it still holds what was read, so two processes taking over the
                # same stale lock do not remove each other's new lock. On Windows the live holder keeps its
                # lock open, so removing a live lock fails.
                if self._read_file_lock()[1] != raw:
                    continue
                try:
                    os.remove(self.lock_path)
                except FileNotFoundError:
                    pass
                except PermissionError:
                    raise self._locked_error(held) from None
                stale.append(held if held is not None else {"unreadable": raw[:200]})
                continue
            try:
                os.write(fd, json.dumps(info, sort_keys=True).encode("utf-8"))
                os.fsync(fd)
            except OSError:
                os.close(fd)
                raise
            self._lock_fd = fd  # kept open until close(): on Windows no other process can remove it meanwhile
            self._lock_token = token
            atexit.register(self._close_at_exit)
            if not stale:
                return None
            return {"event": "lock_taken_over", "stale_locks": stale, "pid": info["pid"],
                    "reason": "the process that held the lock is not running"}
        raise LedgerLocked(f"could not create {self.lock_path.name}: other processes keep taking it")

    def _owns_file_lock(self) -> bool:
        held, _raw = self._read_file_lock()
        return held is not None and self._lock_token is not None and held.get("token") == self._lock_token

    def _release_file_lock(self) -> None:
        fd, self._lock_fd = self._lock_fd, None
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            pass
        if self._owns_file_lock():
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

    def close(self) -> None:
        """Release the lock file (a read-only guard holds none). A closed guard refuses every reservation."""
        self._close()
        try:
            atexit.unregister(self._close_at_exit)
        except Exception:  # noqa: BLE001 - nothing to undo at exit
            pass

    def _close_at_exit(self) -> None:
        self._close()

    def _close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._lock_fd is not None and self._owns_file_lock():
                self._write({"event": "close", "open_reservations": len(self.open),
                             "committed": round(self.committed(), 6)})
            self._closed = True
            self._release_file_lock()
            self._resume.notify_all()

    def __enter__(self) -> SpendGuard:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ ledger
    def _write(self, rec: dict[str, Any]) -> None:
        if self.read_only or self._closed:  # a closed guard no longer holds the lock
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"utc": _utc(), **rec}, sort_keys=True)
        with open(self.ledger_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _check_writable(self, what: str) -> None:
        if self.read_only:
            raise RuntimeError(f"read-only SpendGuard: {what} would change the ledger")

    def _replay(self, note: dict[str, Any] | None = None) -> None:
        if not self.ledger_path.is_file():
            self._write({"event": "start", "config": asdict(self.config)})
            if note:
                self._write(note)
            return
        open_: dict[int, dict[str, Any]] = {}
        max_id = 0
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = r.get("event")
            rid = int(r.get("rid") or 0)
            max_id = max(max_id, rid)
            if ev == "key_usage_start":
                self.key_usage_at_start = float(r["usage"])
            elif ev == "reserve":
                open_[rid] = {"usd": r["usd"], "bucket": r["bucket"], "model": r["model"], "label": r.get("label")}
            elif ev == "reconcile":
                open_.pop(rid, None)
                self.unreconciled.pop(rid, None)
                self._add(self.spent_by, r["bucket"], r["model"], float(r["recorded_usd"]))
            elif ev == "lost":
                item = open_.pop(rid, None) or {"usd": r["usd"], "bucket": r["bucket"], "model": r["model"],
                                                "label": r.get("label")}
                if r.get("generation_id"):  # kept so reconcile_lost can settle it after a restart
                    item["generation_id"] = r["generation_id"]
                elif rid in self.unreconciled and self.unreconciled[rid].get("generation_id"):
                    item["generation_id"] = self.unreconciled[rid]["generation_id"]
                self.unreconciled[rid] = item
            elif ev == "reconcile_lost":
                item = self.unreconciled.pop(rid, None)
                if item is not None:
                    self._add(self.spent_by, r["bucket"], r["model"], float(r["recorded_usd"]))
            elif ev == "stop":
                self.stopped = r.get("reason")
            elif ev == "unstop":
                self.stopped = None
        if note:
            self._write(note)
        if self.read_only:
            # Another process may still have these in flight: show them as open, write nothing.
            self.open.update(open_)
        else:
            for rid, item in open_.items():  # open at the crash: count as spent, unreconciled
                self.unreconciled[rid] = item
                self._write({"event": "lost", "rid": rid, "usd": item["usd"], "bucket": item["bucket"],
                             "model": item["model"], "label": item.get("label"),
                             "reason": "open reservation found on restart"})
        self._rebuild_unrec()
        self._ids = itertools.count(max_id + 1)
        self._write({"event": "resume", "spent": self.spent(), "unreconciled": self.unreconciled_total()})

    @staticmethod
    def _add(table: dict[tuple[str, str], float], bucket: str, model: str | None, usd: float) -> None:
        k = (bucket, model or "")
        table[k] = table.get(k, 0.0) + usd

    def _rebuild_unrec(self) -> None:
        self.unrec_by = {}
        for item in self.unreconciled.values():
            self._add(self.unrec_by, item["bucket"], item["model"], float(item["usd"]))

    # ------------------------------------------------------------------ totals
    def spent(self) -> float:
        return sum(self.spent_by.values())

    def unreconciled_total(self) -> float:
        return sum(float(i["usd"]) for i in self.unreconciled.values())

    def outstanding(self) -> float:
        return sum(float(i["usd"]) for i in self.open.values())

    def committed(self) -> float:
        return self.spent() + self.unreconciled_total() + self.outstanding()

    def committed_bucket(self, bucket: str) -> float:
        s = sum(v for (b, _), v in self.spent_by.items() if b == bucket)
        u = sum(float(i["usd"]) for i in self.unreconciled.values() if i["bucket"] == bucket)
        o = sum(float(i["usd"]) for i in self.open.values() if i["bucket"] == bucket)
        return s + u + o

    def committed_model(self, model: str, bucket: str | None = None) -> float:
        bucket = bucket or self.config.model_cap_bucket

        def match(b: str, m: str | None) -> bool:
            return b == bucket and m == model

        s = sum(v for (b, m), v in self.spent_by.items() if match(b, m))
        u = sum(float(i["usd"]) for i in self.unreconciled.values() if match(i["bucket"], i["model"]))
        o = sum(float(i["usd"]) for i in self.open.values() if match(i["bucket"], i["model"]))
        return s + u + o

    def model_remaining(self, model: str) -> float:
        cap = self.config.model_caps.get(model)
        return float("inf") if cap is None else cap - self.committed_model(model)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_bucket: dict[str, float] = {}
            by_model: dict[str, float] = {}
            for (b, m), v in list(self.spent_by.items()) + list(self.unrec_by.items()):
                by_bucket[b] = round(by_bucket.get(b, 0.0) + v, 6)
                if m:
                    by_model[m] = round(by_model.get(m, 0.0) + v, 6)
            return {"spent_reconciled": round(self.spent(), 6),
                    "unreconciled": round(self.unreconciled_total(), 6),
                    "outstanding": round(self.outstanding(), 6), "committed": round(self.committed(), 6),
                    "effective_cap": self.config.effective_cap, "by_bucket": by_bucket, "by_model": by_model,
                    "stopped": self.stopped, "refusals": self.refusals,
                    "key_usage_at_start": self.key_usage_at_start, "read_only": self.read_only}

    # ------------------------------------------------------------------ protocol
    def record_key_usage_start(self, usage: float) -> None:
        with self._lock:
            if self.key_usage_at_start is None:
                self.key_usage_at_start = float(usage)
                self._write({"event": "key_usage_start", "usage": float(usage)})

    def reserve(self, worst_usd: float, *, bucket: str, model: str | None = None, label: str = "") -> int:
        worst = float(worst_usd)
        with self._lock:
            if self.read_only:
                raise SpendRefused("read-only guard: it never reserves")
            while self._paused and not self.stopped and not self._closed:
                self._resume.wait()  # the drift check's re-check decides: proceed, or stopped below
            if self._closed:
                raise SpendRefused("the guard is closed")
            if self.stopped:
                raise PaidCallsStopped(f"paid calls stopped: {self.stopped}")
            reasons = []
            total = self.committed() + worst
            if total > self.config.effective_cap + 1e-12:
                reasons.append(f"run cap: committed {self.committed():.6f} + {worst:.6f} > "
                               f"{self.config.effective_cap:.6f}")
            bcap = self.config.bucket_caps.get(bucket)
            if bcap is not None and self.committed_bucket(bucket) + worst > bcap + 1e-12:
                reasons.append(f"bucket {bucket}: {self.committed_bucket(bucket):.6f} + {worst:.6f} > {bcap:.2f}")
            if model is not None and bucket == self.config.model_cap_bucket:
                mcap = self.config.model_caps.get(model)
                if mcap is not None and self.committed_model(model) + worst > mcap + 1e-12:
                    reasons.append(f"model {model}: {self.committed_model(model):.6f} + {worst:.6f} > {mcap:.2f}")
            if reasons:
                self.refusals += 1
                self._write({"event": "refused", "usd": worst, "bucket": bucket, "model": model, "label": label,
                             "reasons": reasons, "committed": round(self.committed(), 6)})
                raise SpendRefused("; ".join(reasons))
            rid = next(self._ids)
            self.open[rid] = {"usd": worst, "bucket": bucket, "model": model, "label": label}
            self._write({"event": "reserve", "rid": rid, "usd": worst, "bucket": bucket, "model": model,
                         "label": label, "committed_after": round(self.committed(), 6)})
            return rid

    def reconcile(self, rid: int, actual_usd: float | None, *, note: str = "") -> float:
        self._check_writable("reconcile")
        with self._lock:
            item = self.open.pop(rid)
            recorded = float(item["usd"]) if actual_usd is None else float(actual_usd)
            self._add(self.spent_by, item["bucket"], item["model"], recorded)
            self._write({"event": "reconcile", "rid": rid, "bucket": item["bucket"], "model": item["model"],
                         "label": item.get("label"), "reserved_usd": item["usd"], "actual_usd": actual_usd,
                         "recorded_usd": recorded, "over_reservation": actual_usd is not None and
                         float(actual_usd) > float(item["usd"]), "note": note})
            self._since_check += 1
        self.maybe_drift_check()
        return recorded

    def lost(self, rid: int, *, reason: str, generation_id: str | None = None) -> None:
        self._check_writable("lost")
        with self._lock:
            item = self.open.pop(rid)
            item["generation_id"] = generation_id
            self.unreconciled[rid] = item
            self._add(self.unrec_by, item["bucket"], item["model"], float(item["usd"]))
            self._write({"event": "lost", "rid": rid, "usd": item["usd"], "bucket": item["bucket"],
                         "model": item["model"], "label": item.get("label"), "reason": reason,
                         "generation_id": generation_id})

    def reconcile_lost(self, rid: int, actual_usd: float | None) -> bool:
        """Settle a lost attempt with its real cost (GET /generation total_cost). False when there is
        nothing to settle: an unknown rid, one already settled, or no usable cost (the reservation stays)."""
        self._check_writable("reconcile_lost")
        if actual_usd is None:
            return False
        try:
            usd = float(actual_usd)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(usd) or usd < 0:
            return False
        with self._lock:
            item = self.unreconciled.pop(rid, None)
            if item is None:
                return False
            self._rebuild_unrec()
            self._add(self.spent_by, item["bucket"], item["model"], usd)
            self._write({"event": "reconcile_lost", "rid": rid, "bucket": item["bucket"], "model": item["model"],
                         "label": item.get("label"), "generation_id": item.get("generation_id"),
                         "reserved_usd": item["usd"], "recorded_usd": usd})
            return True

    def lost_with_generation(self) -> dict[int, str]:
        """rid -> generation id of every unreconciled attempt that has one (replayed from the ledger too)."""
        with self._lock:
            return {rid: i["generation_id"] for rid, i in self.unreconciled.items() if i.get("generation_id")}

    def stop(self, reason: str) -> None:
        self._check_writable("stop")
        with self._lock:
            self._stop_locked(reason)

    def _stop_locked(self, reason: str) -> None:
        if not self.stopped:
            self.stopped = reason
            self._write({"event": "stop", "reason": reason})

    def can_start_episode(self, model: str, expected_usd: float, factor: float = 1.5) -> bool:
        """budget.yaml episode_start_check: the model's remaining cap covers 1.5 times the expected cost."""
        with self._lock:
            need = factor * float(expected_usd)
            ok = self.model_remaining(model) >= need and \
                self.config.effective_cap - self.committed() >= need and not self.stopped
            if not ok:
                self._write({"event": "episode_not_started", "model": model, "expected_usd": expected_usd,
                             "model_remaining": round(self.model_remaining(model), 6),
                             "run_remaining": round(self.config.effective_cap - self.committed(), 6)})
            return ok

    # ------------------------------------------------------------------ drift check
    def maybe_drift_check(self, force: bool = False) -> None:
        if self.read_only or self.key_usage is None or self.key_usage_at_start is None:
            return
        with self._lock:
            due = force or self._since_check >= self.config.drift_every_calls or \
                self._clock() - self._last_check >= self.config.drift_every_s
            if not due or self._checking:  # one check at a time
                return
            self._checking = True
            self._since_check = 0
            self._last_check = self._clock()
        try:
            if not self._drift_exceeded():
                return
            with self._lock:
                self._paused = True
                self._write({"event": "drift_pause", "wait_s": self.config.drift_wait_s})
            still = False
            try:
                self._sleep(self.config.drift_wait_s)
                still = self._drift_exceeded()
            finally:
                with self._lock:  # stop before waking the waiting reservations, so they refuse
                    if still:
                        self._stop_locked("persistent spend drift: the key's usage exceeds the ledger")
                    self._paused = False
                    self._resume.notify_all()
        finally:
            with self._lock:
                self._checking = False

    def _drift_exceeded(self) -> bool:
        try:
            now = self.key_usage()
        except Exception as exc:  # noqa: BLE001 - a failed read is logged, never fatal
            with self._lock:
                self._write({"event": "drift_check_error", "error": type(exc).__name__})
            return False
        if now is None:
            return False
        with self._lock:
            delta = float(now) - float(self.key_usage_at_start or 0.0)
            ledger = self.spent() + self.unreconciled_total()
            # Open reservations widen the margin: OpenRouter may have billed a call not reconciled here yet.
            outstanding = self.outstanding()
            allowed = ledger + outstanding + max(self.config.drift_min_usd, self.config.drift_share * ledger)
            exceeded = delta > allowed
            self._write({"event": "drift_check", "key_usage_delta": round(delta, 6), "ledger": round(ledger, 6),
                         "outstanding": round(outstanding, 6), "allowed": round(allowed, 6), "exceeded": exceeded})
        return exceeded
