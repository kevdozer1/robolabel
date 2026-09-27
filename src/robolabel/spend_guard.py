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
marked unreconciled, until ``reconcile_lost`` gets the real figure from the generation endpoint. A
lost reservation is never released.

Every event is appended to a JSONL ledger (flushed and fsynced). On restart the ledger is replayed:
reservations that were still open when the process stopped count as spent (unreconciled).

A drift check compares the change in the key's own usage (an injected callable, for example GET
/api/v1/key) with the ledger every 10 reconciled calls or 5 minutes; a persistent excess stops paid
calls for the night. ``stop`` also serves the other stop rules (HTTP 402 for credits, the balance
floor). Keys and headers never reach the ledger.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class SpendRefused(RuntimeError):
    """The guard refused a reservation. Skip the call, log it, never retry it."""


class PaidCallsStopped(RuntimeError):
    """Paid calls are stopped for the night (drift, credits, floor)."""


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


class SpendGuard:
    """Thread-safe reserve / reconcile ledger. See the module docstring for the rules."""

    def __init__(self, config: GuardConfig, ledger_path: str | Path, *,
                 key_usage: Callable[[], float | None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.ledger_path = Path(ledger_path)
        self.key_usage = key_usage
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
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
        self._replay()

    # ------------------------------------------------------------------ ledger
    def _write(self, rec: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"utc": _utc(), **rec}, sort_keys=True)
        with open(self.ledger_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _replay(self) -> None:
        if not self.ledger_path.is_file():
            self._write({"event": "start", "config": asdict(self.config)})
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
                item = open_.pop(rid, None) or {"usd": r["usd"], "bucket": r["bucket"], "model": r["model"]}
                self.unreconciled[rid] = item
            elif ev == "reconcile_lost":
                item = self.unreconciled.pop(rid, None)
                if item is not None:
                    self._add(self.spent_by, r["bucket"], r["model"], float(r["recorded_usd"]))
            elif ev == "stop":
                self.stopped = r.get("reason")
            elif ev == "unstop":
                self.stopped = None
        for rid, item in open_.items():  # open at the crash: count as spent, unreconciled
            self.unreconciled[rid] = item
            self._write({"event": "lost", "rid": rid, "usd": item["usd"], "bucket": item["bucket"],
                         "model": item["model"], "reason": "open reservation found on restart"})
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
                    "key_usage_at_start": self.key_usage_at_start}

    # ------------------------------------------------------------------ protocol
    def record_key_usage_start(self, usage: float) -> None:
        with self._lock:
            if self.key_usage_at_start is None:
                self.key_usage_at_start = float(usage)
                self._write({"event": "key_usage_start", "usage": float(usage)})

    def reserve(self, worst_usd: float, *, bucket: str, model: str | None = None, label: str = "") -> int:
        worst = float(worst_usd)
        with self._lock:
            if self.stopped:
                raise PaidCallsStopped(f"paid calls stopped: {self.stopped}")
            if self._paused:
                raise SpendRefused("paid calls paused by the drift check")
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
        with self._lock:
            item = self.open.pop(rid)
            item["generation_id"] = generation_id
            self.unreconciled[rid] = item
            self._add(self.unrec_by, item["bucket"], item["model"], float(item["usd"]))
            self._write({"event": "lost", "rid": rid, "usd": item["usd"], "bucket": item["bucket"],
                         "model": item["model"], "label": item.get("label"), "reason": reason,
                         "generation_id": generation_id})

    def reconcile_lost(self, rid: int, actual_usd: float) -> None:
        with self._lock:
            item = self.unreconciled.pop(rid, None)
            if item is None:
                return
            self._rebuild_unrec()
            self._add(self.spent_by, item["bucket"], item["model"], float(actual_usd))
            self._write({"event": "reconcile_lost", "rid": rid, "bucket": item["bucket"], "model": item["model"],
                         "reserved_usd": item["usd"], "recorded_usd": float(actual_usd)})

    def lost_with_generation(self) -> dict[int, str]:
        with self._lock:
            return {rid: i["generation_id"] for rid, i in self.unreconciled.items() if i.get("generation_id")}

    def stop(self, reason: str) -> None:
        with self._lock:
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
        if self.key_usage is None or self.key_usage_at_start is None:
            return
        due = force or self._since_check >= self.config.drift_every_calls or \
            self._clock() - self._last_check >= self.config.drift_every_s
        if not due:
            return
        self._since_check = 0
        self._last_check = self._clock()
        if not self._drift_exceeded():
            return
        with self._lock:
            self._paused = True
            self._write({"event": "drift_pause", "wait_s": self.config.drift_wait_s})
        self._sleep(self.config.drift_wait_s)
        still = self._drift_exceeded()
        with self._lock:
            self._paused = False
        if still:
            self.stop("persistent spend drift: the key's usage exceeds the ledger")

    def _drift_exceeded(self) -> bool:
        try:
            now = self.key_usage()
        except Exception as exc:  # noqa: BLE001 - a failed read is logged, never fatal
            self._write({"event": "drift_check_error", "error": type(exc).__name__})
            return False
        if now is None:
            return False
        delta = float(now) - float(self.key_usage_at_start or 0.0)
        with self._lock:
            ledger = self.spent() + self.unreconciled_total()
        allowed = ledger + max(self.config.drift_min_usd, self.config.drift_share * ledger)
        exceeded = delta > allowed
        self._write({"event": "drift_check", "key_usage_delta": round(delta, 6), "ledger": round(ledger, 6),
                     "allowed": round(allowed, 6), "exceeded": exceeded})
        return exceeded
