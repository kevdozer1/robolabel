"""Spend guard and OpenRouter provider with a fake transport (no network, no key)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time

import pytest

from robolabel.eval.receipts import ResponseCache
from robolabel.providers.base import CallRequest, ImagePart, TextPart
from robolabel.spend_guard import GuardConfig, LedgerLocked, PaidCallsStopped, SpendGuard, SpendRefused

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
          "properties": {"answer": {"type": "string"}}}

# Synthetic key-shaped strings for the redaction tests. The prefix is joined at runtime so this file
# never contains a literal that a secret scanner would flag; none of these is a real key.
FAKE_KEY_PREFIX = "sk-or" + "-v1-"


def cfg(**kw) -> GuardConfig:
    base = dict(run_cap=1.0, available_at_start=12.0, balance_floor=1.5, bucket_caps={"sweep": 0.8, "debug": 0.3},
                model_caps={"m1": 0.5})
    base.update(kw)
    return GuardConfig(**base)


# ------------------------------------------------------------------------------------------ guard
def test_reserve_refuses_past_run_bucket_and_model_caps(tmp_path):
    g = SpendGuard(cfg(), tmp_path / "ledger.jsonl")
    rid = g.reserve(0.4, bucket="sweep", model="m1")
    g.reconcile(rid, 0.3)
    with pytest.raises(SpendRefused):  # model cap 0.5: 0.3 + 0.25 > 0.5
        g.reserve(0.25, bucket="sweep", model="m1")
    g.reserve(0.19, bucket="sweep", model="m1")  # 0.49 fits
    with pytest.raises(SpendRefused):  # bucket cap 0.8
        g.reserve(0.35, bucket="sweep", model="m2")
    with pytest.raises(SpendRefused):  # debug bucket 0.3
        g.reserve(0.31, bucket="debug")
    assert g.refusals == 3


def test_balance_floor_binds_when_lower_than_run_cap(tmp_path):
    g = SpendGuard(cfg(run_cap=9.0, available_at_start=2.0, balance_floor=1.5, bucket_caps={}), tmp_path / "l.jsonl")
    assert g.config.effective_cap == pytest.approx(0.5)
    g.reserve(0.5, bucket="x")
    with pytest.raises(SpendRefused):
        g.reserve(0.01, bucket="x")


def test_lost_reservation_stays_spent_and_resume_counts_open(tmp_path):
    led = tmp_path / "ledger.jsonl"
    g = SpendGuard(cfg(), led)
    r1 = g.reserve(0.2, bucket="sweep", model="m1")
    g.lost(r1, reason="timeout")
    g.reserve(0.1, bucket="sweep", model="m1")  # left open: a crash
    assert g.committed() == pytest.approx(0.3)
    g.close()  # the process ends with the reservation still open; close only releases the ledger lock
    g2 = SpendGuard(cfg(), led)  # restart
    assert g2.unreconciled_total() == pytest.approx(0.3) and g2.outstanding() == 0
    g2.reconcile_lost(r1, 0.05)
    assert g2.committed() == pytest.approx(0.15)


def test_drift_check_pauses_then_stops(tmp_path):
    usage = {"v": 10.0}
    slept = []
    g = SpendGuard(cfg(drift_every_calls=1), tmp_path / "l.jsonl", key_usage=lambda: usage["v"], sleep=slept.append)
    g.record_key_usage_start(10.0)
    rid = g.reserve(0.1, bucket="sweep")
    usage["v"] = 10.05  # within max(0.10, 10 percent)
    g.reconcile(rid, 0.05)
    assert g.stopped is None
    rid = g.reserve(0.1, bucket="sweep")
    usage["v"] = 11.0  # key usage far ahead of the ledger
    g.reconcile(rid, 0.05)
    assert slept == [60.0] and g.stopped
    with pytest.raises(PaidCallsStopped):
        g.reserve(0.01, bucket="sweep")


def test_episode_start_check(tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    assert g.can_start_episode("m1", 0.3)  # 0.45 <= 0.5
    assert not g.can_start_episode("m1", 0.34)  # 0.51 > 0.5


# ------------------------------------------------------------------------------------------ provider
class FakeTransport:
    """Scripted responses: each item is a dict (status, data), the string "timeout", or "connection:<message>"."""

    def __init__(self, script=None, default=None, get_data=None):
        self.script = list(script or [])
        self.default = default
        self.bodies = []
        self.gets = []
        self.get_data = get_data or {"data": {"usage": 1.0, "total_cost": 0.001}}

    def post(self, url, body, headers, timeout):
        assert "Authorization" in headers
        self.bodies.append(body)
        item = self.script.pop(0) if self.script else self.default
        if callable(item):
            item = item(body)
        if item == "timeout":
            raise TimeoutError("timed out")
        if isinstance(item, str) and item.startswith("connection:"):
            raise ConnectionError(item.split(":", 1)[1])
        return item["status"], item.get("data"), json.dumps(item.get("data")), item.get("headers", {})

    def get(self, url, headers, timeout):
        self.gets.append(url)
        return 200, self.get_data, "", {}


def ok(text='{"answer": "yes"}', cost=0.001, finish="stop", gen="gen-1"):
    return {"status": 200, "data": {"id": gen, "model": "fake/model", "provider": "FakeProv",
                                    "choices": [{"message": {"content": text}, "finish_reason": finish}],
                                    "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": cost,
                                              "completion_tokens_details": {"reasoning_tokens": 20}}}}


def provider(monkeypatch, tmp_path, transport, guard=None, cache=True):
    from robolabel.providers.openrouter import OpenRouterProvider

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    return OpenRouterProvider("fake/model", guard=guard, transport=transport, sleep=lambda s: None,
                              cache=ResponseCache(tmp_path / "cache.jsonl") if cache else None,
                              prices={"fake/model": (1.0, 4.0)})


def request(text="what is it?", max_tokens=1000, bucket="sweep", model_key="m1"):
    return CallRequest(step="scene_inventory", system="sys", parts=[TextPart(text), ImagePart(b"\xff\xd8jpegbytes", "x")],
                       schema=SCHEMA, schema_name="t", max_tokens=max_tokens,
                       reasoning={"effort": "low", "exclude": True},
                       context={"arm": "v@m1", "episode_key": "F1/0", "bucket": bucket, "model_key": model_key},
                       image_tokens_per_image=500)


def test_success_reconciles_and_caches(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport([ok(cost=0.002), ok(cost=0.003)])
    p = provider(monkeypatch, tmp_path, tr, g)
    r = p.call(request())
    assert r.valid and r.data == {"answer": "yes"} and r.usd == pytest.approx(0.002) and r.mode == "json_schema_strict"
    assert g.spent() == pytest.approx(0.002) and g.outstanding() == 0
    body = tr.bodies[0]
    assert body["response_format"]["type"] == "json_schema" and "temperature" not in body
    assert body["provider"]["max_price"] == {"prompt": 1.5, "completion": 6.0}
    assert "test-key" not in json.dumps(r.receipt) and "base64" not in json.dumps(r.receipt)
    r2 = p.call(request())  # identical request: cache hit, no HTTP, cost and latency of the original
    assert r2.cache_hit and len(tr.bodies) == 1 and r2.usd == pytest.approx(0.002)
    r3 = p.call(request(text="a different prompt"))  # prompt change misses the cache
    assert not r3.cache_hit and len(tr.bodies) == 2


def test_schema_rejection_falls_back_to_json_object(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport([{"status": 400, "data": {"error": {"message": "response_format json_schema not supported"}}},
                        ok()])
    r = provider(monkeypatch, tmp_path, tr, g).call(request())
    assert r.valid and r.mode == "json_object_with_schema_in_prompt"
    assert tr.bodies[1]["response_format"] == {"type": "json_object"}
    assert g.committed() == pytest.approx(0.001)  # the rejected attempt is recorded at 0


def test_timeouts_stay_spent_and_retry_once(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport(["timeout", "timeout"])
    r = provider(monkeypatch, tmp_path, tr, g).call(request())
    assert r.status == "failed" and len(tr.bodies) == 2
    assert g.unreconciled_total() > 0 and g.outstanding() == 0


def test_length_retry_doubles_max_tokens_and_repair_retry(monkeypatch, tmp_path):
    tr = FakeTransport([ok(text='{"answer": ', finish="length"), ok(text='{"wrong": 1}'), ok()])
    r = provider(monkeypatch, tmp_path, tr).call(request(max_tokens=3000))
    assert r.valid and r.repaired and r.truncated_retry
    assert tr.bodies[1]["max_tokens"] == 6000
    assert tr.bodies[2]["messages"][-1]["role"] == "user" and "did not validate" in tr.bodies[2]["messages"][-1]["content"]


def test_http_402_credits_stops_paid_calls(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport([{"status": 402, "data": {"error": {"message": "Insufficient credits",
                                                           "metadata": {"limit_source": "key_limit"}}}}])
    r = provider(monkeypatch, tmp_path, tr, g).call(request())
    assert r.status == "stopped" and g.stopped


def test_5xx_retries_then_succeeds(monkeypatch, tmp_path):
    tr = FakeTransport([{"status": 502, "data": {"error": {"message": "bad gateway"}}}, ok()])
    r = provider(monkeypatch, tmp_path, tr).call(request())
    assert r.valid and r.attempts == 2


def test_cap_never_crossed_with_random_timeouts_and_retries(monkeypatch, tmp_path):
    rng = random.Random(7)
    g = SpendGuard(cfg(run_cap=0.5, bucket_caps={"sweep": 0.5}, model_caps={}), tmp_path / "l.jsonl")
    peak = {"v": 0.0}

    def respond(body):
        peak["v"] = max(peak["v"], g.committed())
        x = rng.random()
        if x < 0.3:
            return "timeout"
        if x < 0.4:
            return {"status": 503, "data": {"error": {"message": "overloaded"}}}
        worst = (len(json.dumps(body)) / 3.5 + 500) * 1.0 / 1e6 + body["max_tokens"] * 4.0 / 1e6
        return ok(cost=round(rng.uniform(0.2, 1.0) * worst, 6))

    tr = FakeTransport(default=respond)
    p = provider(monkeypatch, tmp_path, tr, g, cache=False)
    statuses = [p.call(request(text=f"q{i}", max_tokens=2000)).status for i in range(200)]
    assert "refused" in statuses  # the guard ran out of room and refused, never overspent
    assert g.committed() <= 0.5 + 1e-9 and peak["v"] <= 0.5 + 1e-9


def test_legacy_ask_contact_sheet_plain_and_cached(monkeypatch, tmp_path):
    import numpy as np

    g = SpendGuard(cfg(bucket_caps={"legacy_arm": 0.4}), tmp_path / "l.jsonl")
    tr = FakeTransport([ok(text='{"segments": []}', cost=0.004)])
    p = provider(monkeypatch, tmp_path, tr, g)
    p.context = {"arm": "b2b@x", "episode_key": "F1/0", "bucket": "legacy_arm", "model_key": "gemini_flash"}
    frames = [np.zeros((40, 60, 3), dtype=np.uint8)] * 3
    r = p.ask(frames, [0, 5, 9], "segment this episode", tmp_path / "rc" / "subtasks_label.json")
    assert r.answer == '{"segments": []}' and r.estimated_cost_usd == pytest.approx(0.004)
    body = tr.bodies[0]
    assert "response_format" not in body and body["messages"][0]["role"] == "user"
    texts = [c["text"] for c in body["messages"][0]["content"] if c["type"] == "text"]
    assert texts == ["segment this episode"]
    assert sum(1 for c in body["messages"][0]["content"] if c["type"] == "image_url") == 1
    assert (tmp_path / "rc" / "subtasks_label.json").is_file()
    assert g.committed_bucket("legacy_arm") == pytest.approx(0.004)
    r2 = p.ask(frames, [0, 5, 9], "segment this episode", tmp_path / "rc" / "subtasks_label2.json")
    assert r2.answer == r.answer and len(tr.bodies) == 1


# ------------------------------------------------------------------------------------------ review fixes
# review:money 0 (one writer per ledger), 1 (drift pause), 2 and 3 (HTTP 402, legacy rules), 4 (200 with an
# error), 5 (unknown costs, lost attempts), 9 (repair reservation), 14 (redacted error text).
def events(led):
    return [json.loads(line) for line in led.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_py(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=120)


def test_second_writer_is_refused_while_the_lock_is_held(tmp_path):
    led = tmp_path / "ledger.jsonl"
    g = SpendGuard(cfg(), led)
    lock = json.loads((tmp_path / "ledger.jsonl.lock").read_text(encoding="utf-8"))
    assert lock["pid"] == os.getpid() and lock["started_utc"] and lock["token"]
    with pytest.raises(LedgerLocked, match="this process"):
        SpendGuard(cfg(), led)
    g.close()
    assert not (tmp_path / "ledger.jsonl.lock").exists()
    with pytest.raises(SpendRefused):  # a closed guard never reserves
        g.reserve(0.01, bucket="sweep")
    g2 = SpendGuard(cfg(), led)
    assert g2.reserve(0.01, bucket="sweep") == 1
    g2.close()


def test_lock_of_a_live_process_refuses_and_a_dead_one_is_taken_over(tmp_path):
    led = tmp_path / "ledger.jsonl"
    g = SpendGuard(cfg(), led)
    rid = g.reserve(0.2, bucket="sweep", model="m1")  # in flight when its process dies
    g.close()
    lock = tmp_path / "ledger.jsonl.lock"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock.write_text(json.dumps({"pid": proc.pid, "started_utc": "x", "started_unix": time.time(), "token": "a"}),
                        encoding="utf-8")
        with pytest.raises(LedgerLocked, match=f"pid {proc.pid}"):
            SpendGuard(cfg(), led)
    finally:
        proc.kill()
        proc.wait(timeout=60)
    g2 = SpendGuard(cfg(), led)  # the holder is gone: its lock is stale
    ev = events(led)
    note = next(e for e in ev if e["event"] == "lock_taken_over")
    assert note["stale_locks"][0]["pid"] == proc.pid
    assert any(e["event"] == "lost" and e["rid"] == rid for e in ev)
    assert g2.unreconciled_total() == pytest.approx(0.2)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    g2.close()


def test_a_guard_that_fails_to_start_leaves_no_lock(tmp_path):
    led = tmp_path / "ledger.jsonl"
    led.write_text(json.dumps({"event": "reserve", "rid": 1}), encoding="utf-8")  # no usd: the replay fails
    with pytest.raises(KeyError):
        SpendGuard(cfg(), led)
    assert not (tmp_path / "ledger.jsonl.lock").exists()


@pytest.mark.skipif(os.name != "nt", reason="pid reuse is detected from the process creation time on Windows")
def test_lock_with_a_reused_pid_is_stale_on_windows(tmp_path):
    led = tmp_path / "ledger.jsonl"
    lock = tmp_path / "ledger.jsonl.lock"
    # this process is alive, but it was created long after this lock was taken: not the holder
    lock.write_text(json.dumps({"pid": os.getpid(), "started_unix": 1000.0, "token": "old"}), encoding="utf-8")
    g = SpendGuard(cfg(), led)
    assert any(e["event"] == "lock_taken_over" for e in events(led))
    g.close()


def test_lock_released_at_exit_and_left_stale_by_a_crash(tmp_path):
    led = tmp_path / "ledger.jsonl"
    code = ("import sys, os\n"
            "from robolabel.spend_guard import GuardConfig, SpendGuard\n"
            "g = SpendGuard(GuardConfig(run_cap=1.0, available_at_start=12.0, balance_floor=1.5), sys.argv[1])\n"
            "g.reserve(0.1, bucket='sweep')\n"
            "if sys.argv[2] == 'crash':\n"
            "    os._exit(0)\n")
    out = run_py(code, str(led), "exit")
    assert out.returncode == 0, out.stderr
    assert not (tmp_path / "ledger.jsonl.lock").exists()  # released by the atexit hook
    assert events(led)[-1]["event"] == "close"
    out = run_py(code, str(led), "crash")
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "ledger.jsonl.lock").exists()  # a crash leaves the lock behind
    g = SpendGuard(cfg(), led)
    assert any(e["event"] == "lock_taken_over" for e in events(led))
    assert g.unreconciled_total() == pytest.approx(0.2) and g.outstanding() == 0
    g.close()


def test_read_only_guard_replays_without_writing(tmp_path):
    led = tmp_path / "ledger.jsonl"
    g = SpendGuard(cfg(), led)
    g.record_key_usage_start(5.0)
    r1 = g.reserve(0.4, bucket="sweep", model="m1")  # in flight in the paying process
    r2 = g.reserve(0.1, bucket="sweep", model="m1")
    g.reconcile(r2, 0.05)
    before = led.read_bytes()
    ro = SpendGuard(cfg(), led, read_only=True)  # works while the paying process holds the lock
    ro.record_key_usage_start(9.0)
    assert ro.outstanding() == pytest.approx(0.4) and ro.committed() == pytest.approx(g.committed())
    assert ro.summary()["read_only"] and ro.key_usage_at_start == 5.0
    with pytest.raises(SpendRefused):
        ro.reserve(0.01, bucket="sweep")
    with pytest.raises(RuntimeError):
        ro.reconcile_lost(r1, 0.1)
    ro.close()
    assert led.read_bytes() == before  # no lost lines for the other process's reservation, no resume line
    SpendGuard(cfg(), tmp_path / "missing.jsonl", read_only=True)
    assert not (tmp_path / "missing.jsonl").exists() and not (tmp_path / "missing.jsonl.lock").exists()
    g.reconcile(r1, 0.3)
    g.close()
    g2 = SpendGuard(cfg(), led)
    assert g2.unreconciled_total() == 0 and g2.spent() == pytest.approx(0.35)
    g2.close()


def _paused_guard(tmp_path, usage):
    gate = threading.Event()
    slept = []

    def sleep(s):
        slept.append(s)
        assert gate.wait(30)

    g = SpendGuard(cfg(drift_every_calls=1), tmp_path / "l.jsonl", key_usage=lambda: usage["v"], sleep=sleep)
    g.record_key_usage_start(10.0)
    rid = g.reserve(0.1, bucket="sweep")
    usage["v"] = 11.0  # far ahead of the ledger: the check pauses
    checker = threading.Thread(target=g.reconcile, args=(rid, 0.05))
    checker.start()
    for _ in range(300):
        if g._paused:
            break
        time.sleep(0.01)
    assert g._paused
    return g, gate, checker, slept


def _reserve_in_thread(g):
    out = {}

    def run():
        try:
            out["rid"] = g.reserve(0.1, bucket="sweep")
        except Exception as exc:  # noqa: BLE001 - checked by the caller
            out["exc"] = exc

    th = threading.Thread(target=run)
    th.start()
    return th, out


def test_reserve_waits_for_the_drift_recheck_then_proceeds(tmp_path):
    usage = {"v": 10.0}
    g, gate, checker, slept = _paused_guard(tmp_path, usage)
    th, out = _reserve_in_thread(g)
    th.join(0.3)
    assert th.is_alive() and not out  # waiting, not refused
    usage["v"] = 10.1  # the key's usage caught up with the ledger during the wait
    gate.set()
    th.join(10)
    checker.join(10)
    assert out.get("rid") == 2 and g.stopped is None and slept == [60.0]
    assert not any(e["event"] == "refused" for e in events(tmp_path / "l.jsonl"))


def test_reserve_waits_for_the_drift_recheck_then_refuses_when_it_persists(tmp_path):
    usage = {"v": 10.0}
    g, gate, checker, _slept = _paused_guard(tmp_path, usage)
    th, out = _reserve_in_thread(g)
    th.join(0.3)
    assert th.is_alive()
    gate.set()
    th.join(10)
    checker.join(10)
    assert isinstance(out.get("exc"), PaidCallsStopped) and g.stopped


def test_drift_margin_counts_open_reservations(tmp_path):
    usage = {"v": 10.0}
    slept = []
    g = SpendGuard(cfg(drift_every_calls=1), tmp_path / "l.jsonl", key_usage=lambda: usage["v"], sleep=slept.append)
    g.record_key_usage_start(10.0)
    g.reserve(0.5, bucket="sweep")  # in flight; OpenRouter has billed most of it already
    rid = g.reserve(0.1, bucket="sweep")
    usage["v"] = 10.0 + 0.1 + 0.45
    g.reconcile(rid, 0.1)  # ledger 0.1; without the open 0.5 the allowed excess would be 0.1
    assert slept == [] and g.stopped is None
    check = [e for e in events(tmp_path / "l.jsonl") if e["event"] == "drift_check"][-1]
    assert check["outstanding"] == pytest.approx(0.5) and not check["exceeded"]


def in_flight_402(retry_after=None):
    item = {"status": 402, "data": {"error": {"message": "in-flight budget full",
                                              "metadata": {"limit_source": "openrouter_in_flight_budget"}}}}
    if retry_after is not None:
        item["headers"] = {"retry-after": retry_after}
    return item


def test_in_flight_402_waits_retry_after_and_is_bounded(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport(default=in_flight_402("7"))
    p = provider(monkeypatch, tmp_path, tr, g)
    sleeps = []
    p._sleep = sleeps.append
    assert p.in_flight_backoff is g.in_flight_backoff
    r = p.call(request())
    assert r.status == "refused" and "still full" in r.error
    assert sleeps == [7.0] * 5 and len(tr.bodies) == 6
    assert g.in_flight_backoff.is_set() and g.stopped is None and g.committed() == 0


def test_in_flight_402_http_date_retry_after_then_success_clears_the_flag(monkeypatch, tmp_path):
    import datetime as dt
    import email.utils

    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    when = email.utils.format_datetime(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30), usegmt=True)
    tr = FakeTransport([in_flight_402(when), in_flight_402("soon"), ok()])
    p = provider(monkeypatch, tmp_path, tr, g)
    sleeps = []
    p._sleep = sleeps.append
    r = p.call(request())
    assert r.valid and len(tr.bodies) == 3
    assert 20.0 <= sleeps[0] <= 31.0 and sleeps[1] == 10.0  # an unreadable Retry-After waits 10 s
    assert not g.in_flight_backoff.is_set()


def test_retry_after_parsing():
    from robolabel.providers.openrouter import _retry_after_s

    assert _retry_after_s(None) == 10.0 and _retry_after_s("12") == 12.0 and _retry_after_s("0") == 1.0
    assert _retry_after_s("9999") == 120.0 and _retry_after_s("nan") == 10.0 and _retry_after_s("later") == 10.0
    assert _retry_after_s("Wed, 21 Oct 2015 07:28:00 GMT") == 1.0  # a date in the past


def legacy_provider(monkeypatch, tmp_path, tr, g):
    p = provider(monkeypatch, tmp_path, tr, g)
    p.context = {"arm": "b2b@x", "episode_key": "F1/0", "bucket": "legacy_arm", "model_key": "gemini_flash"}
    return p


def legacy_ask(p, tmp_path, name="l.json"):
    import numpy as np

    return p.ask([np.zeros((40, 60, 3), dtype=np.uint8)] * 2, [0, 5], "segment this episode", tmp_path / "rc" / name)


def test_legacy_in_flight_402_waits_and_credits_402_skips_without_stopping(monkeypatch, tmp_path):
    led = tmp_path / "l.jsonl"
    g = SpendGuard(cfg(bucket_caps={"legacy_arm": 0.4}), led)
    tr = FakeTransport([in_flight_402("3"), ok(text='{"segments": []}')])
    p = legacy_provider(monkeypatch, tmp_path, tr, g)
    assert legacy_ask(p, tmp_path).answer == '{"segments": []}' and g.stopped is None
    tr.script = [{"status": 402, "data": {"error": {"message": "too big",
                                                    "metadata": {"limit_source": "openrouter_credits"}}}}]
    p.cache = None  # the same request again, sent this time
    with pytest.raises(RuntimeError, match="refused"):
        legacy_ask(p, tmp_path, "l2.json")
    assert g.stopped is None
    g.close()
    assert SpendGuard(cfg(), led, read_only=True).stopped is None  # nothing persisted a stop
    g = SpendGuard(cfg(bucket_caps={"legacy_arm": 0.4}), led)
    p = legacy_provider(monkeypatch, tmp_path, FakeTransport([{"status": 402, "data": {"error": {
        "message": "Insufficient credits", "metadata": {"limit_source": "key_limit"}}}}]), g)
    p.cache = None
    with pytest.raises(RuntimeError, match="stopped"):
        legacy_ask(p, tmp_path, "l3.json")
    assert g.stopped
    g.close()


def test_legacy_empty_length_doubles_once_then_retries_empty_once(monkeypatch, tmp_path):
    empty_length = ok(text="", finish="length")
    tr = FakeTransport([empty_length, empty_length, empty_length, empty_length])
    p = legacy_provider(monkeypatch, tmp_path, tr, None)
    with pytest.raises(RuntimeError, match="failed"):
        legacy_ask(p, tmp_path)
    assert [b["max_tokens"] for b in tr.bodies] == [8000, 16000, 16000]
    tr2 = FakeTransport([ok(text="", finish="length"), ok(text='{"segments": []}')])
    p2 = legacy_provider(monkeypatch, tmp_path, tr2, None)
    p2.cache = None
    assert legacy_ask(p2, tmp_path, "l2.json").answer == '{"segments": []}'
    assert [b["max_tokens"] for b in tr2.bodies] == [8000, 16000]


def error_200(gen="gen-e"):
    return {"status": 200, "data": {"id": gen, "error": {"code": 502, "message": "upstream provider failed"}}}


def test_200_with_an_error_object_is_retried_once(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport([error_200(), ok()])
    r = provider(monkeypatch, tmp_path, tr, g).call(request())
    assert r.valid and r.attempts == 2
    first = r.receipt["attempts"][0]
    assert first["kind"] == "error_200" and first["usd"] is None and first["unreconciled_reserved_usd"] > 0
    assert list(g.lost_with_generation().values()) == ["gen-e"]  # it may be billed: kept, settled later
    tr2 = FakeTransport([error_200(), error_200()])
    r2 = provider(monkeypatch, tmp_path, tr2, None, cache=False).call(request(text="other"))
    assert r2.status == "failed" and "HTTP 200 with an error" in r2.error and len(tr2.bodies) == 2


def test_unknown_cost_attempts_are_recorded_unreconciled(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    receipts = []

    class Sink:
        def write(self, rec):
            receipts.append(rec)

    tr = FakeTransport(["timeout", ok(cost=0.002)])
    p = provider(monkeypatch, tmp_path, tr, g)
    p.receipts = Sink()
    r = p.call(request())
    reserved = g.unreconciled_total()
    assert r.valid and r.usd == pytest.approx(0.002) and reserved > 0
    assert r.unreconciled and r.unreconciled_reserved_usd == pytest.approx(reserved)
    rc = receipts[-1]
    assert rc["unreconciled"] and rc["unreconciled_reserved_usd"] == pytest.approx(reserved)
    assert rc["attempts"][0]["usd"] is None
    assert rc["attempts"][0]["unreconciled_reserved_usd"] == pytest.approx(reserved)
    assert rc["attempts"][1]["usd"] == pytest.approx(0.002)
    hit = p.call(request())  # the cache keeps the flag with the original cost
    assert hit.cache_hit and hit.unreconciled and receipts[-1]["unreconciled"]
    clean = provider(monkeypatch, tmp_path, FakeTransport([ok()]), g, cache=False).call(request(text="clean"))
    assert clean.unreconciled is False and clean.unreconciled_reserved_usd == 0.0


def test_rejected_before_generation_is_a_known_zero(monkeypatch, tmp_path):
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    tr = FakeTransport([{"status": 400, "data": {"error": {"message": "response_format json_schema not supported"}}},
                        ok()])
    r = provider(monkeypatch, tmp_path, tr, g).call(request())
    assert r.valid and r.unreconciled is False and r.receipt["attempts"][0]["usd"] == 0.0


def test_lost_with_generation_survives_a_restart_and_settles(monkeypatch, tmp_path):
    led = tmp_path / "l.jsonl"
    g = SpendGuard(cfg(), led)
    no_cost = ok(gen="gen-9")
    del no_cost["data"]["usage"]["cost"]
    r = provider(monkeypatch, tmp_path, FakeTransport([no_cost]), g).call(request())
    assert r.valid and r.unreconciled
    rid = next(iter(g.lost_with_generation()))
    g.close()
    g2 = SpendGuard(cfg(), led)
    assert g2.lost_with_generation() == {rid: "gen-9"}  # the generation id is replayed from the ledger
    assert not g2.reconcile_lost(rid, None) and g2.unreconciled_total() > 0  # no cost: the reservation stays
    tr = FakeTransport(get_data={"data": {"total_cost": 0.0042}})
    p = provider(monkeypatch, tmp_path, tr, g2)
    assert p.settle_lost() == {rid: pytest.approx(0.0042)}
    assert tr.gets == ["https://openrouter.ai/api/v1/generation?id=gen-9"]
    assert g2.unreconciled_total() == 0 and g2.committed() == pytest.approx(0.0042)
    assert not g2.reconcile_lost(rid, 0.5)  # already settled
    g2.close()
    g3 = SpendGuard(cfg(), led, read_only=True)
    assert g3.lost_with_generation() == {} and g3.spent() == pytest.approx(0.0042)


def test_repair_retry_reserves_the_repair_messages(monkeypatch, tmp_path):
    led = tmp_path / "l.jsonl"
    g = SpendGuard(cfg(), led)
    long_invalid = '{"wrong": "' + "x" * 12000 + '"}'
    tr = FakeTransport([ok(text=long_invalid), ok()])
    r = provider(monkeypatch, tmp_path, tr, g).call(request())
    assert r.valid and r.repaired
    reserves = [e["usd"] for e in events(led) if e["event"] == "reserve"]
    extra_chars = sum(len(m["content"]) for m in tr.bodies[1]["messages"][-2:])
    assert extra_chars > 12000
    assert reserves[1] - reserves[0] == pytest.approx(extra_chars / 3.5 * 1.0 / 1e6)


def test_error_text_never_carries_media_or_credentials(monkeypatch, tmp_path):
    from robolabel.eval.receipts import JsonlWriter, _forbidden_keys, check_no_media_payload, check_no_secrets

    b64 = "/9j/4AAQSkZJRgABAQ" + "A" * 400
    data_url = "data:image/jpeg;base64," + b64
    msg = f"invalid image_url {data_url} for key {FAKE_KEY_PREFIX}0123456789abcdef"
    raw = f"upstream said: Bearer test-key-not-real rejected {b64} and {'Q' * 90}"
    tr = FakeTransport([{"status": 400, "data": {"error": {"message": msg, "metadata": {"raw": raw}}},
                         "headers": {"set-cookie": "a=b", "retry-after": "1"}},
                        f"connection:Authorization: Bearer {FAKE_KEY_PREFIX}feedfacefeedface " + data_url,
                        "timeout"])
    g = SpendGuard(cfg(), tmp_path / "l.jsonl")
    p = provider(monkeypatch, tmp_path, tr, g)
    p.receipts = JsonlWriter(tmp_path / "receipts.jsonl")
    r1 = p.call(request())
    r2 = p.call(request(text="second"))
    assert r1.status == "failed" and r2.status == "failed"
    text = (tmp_path / "receipts.jsonl").read_text(encoding="utf-8")
    check_no_media_payload(text)
    check_no_secrets(text)
    for bad in ("test-key-not-real", "sk-or-", "/9j/", "base64,", "data:image", "set-cookie"):
        assert bad not in text and bad not in (r1.error or "") + (r2.error or "")
    assert "[data URL removed]" in text and "[redacted]" in text
    for line in text.splitlines():
        assert _forbidden_keys(json.loads(line)) == []
    ledger = (tmp_path / "l.jsonl").read_text(encoding="utf-8")
    assert "sk-or-" not in ledger and "data:image" not in ledger


def test_legacy_raw_receipt_is_scrubbed(monkeypatch, tmp_path):
    from robolabel.eval.receipts import check_no_media_payload, check_no_secrets

    tr = FakeTransport([{"status": 400, "data": {"error": {"message": "bad data:image/png;base64,iVBORw0KGgo"
                                                                      + "B" * 300 + " " + FAKE_KEY_PREFIX
                                                                      + "0123456789ab"}}}])
    p = legacy_provider(monkeypatch, tmp_path, tr, None)
    with pytest.raises(RuntimeError) as exc:
        legacy_ask(p, tmp_path)
    assert "sk-or-" not in str(exc.value) and "base64," not in str(exc.value)
    text = (tmp_path / "rc" / "l.json").read_text(encoding="utf-8")
    check_no_media_payload(text)
    check_no_secrets(text)


# ------------------------------------------------------------------------------------------ second check
# Cases the first round of fixes left untested: escaped media in error text (review:money 14), guard time
# left out of wall_s and one drift check at a time (1), a 200 with an error on the legacy path (3, 4).
def test_escaped_media_in_error_text_is_removed(monkeypatch, tmp_path):
    import base64

    from robolabel.providers.openrouter import _error_message

    bs = chr(92)  # a backslash
    b64 = base64.b64encode(bytes(range(256)) * 3).decode()  # has "/", "+" and "=" in it
    forms = {"json": "data:image" + bs + "/jpeg;base64," + b64.replace("/", bs + "/"),
             "unicode": "data:image/jpeg;base64," + b64.replace("+", bs + "u002B").replace("/", bs + "u002F"),
             "url": "data%3Aimage%2Fjpeg%3Bbase64%2C" + b64.replace("/", "%2F").replace("+", "%2B")}

    def leaked(text):
        for a, b in ((bs + "/", "/"), (bs + "u002B", "+"), (bs + "u002F", "/"), ("%2F", "/"), ("%2B", "+")):
            text = text.replace(a, b)
        return [b64[i:i + 16] for i in range(0, len(b64) - 16, 8) if b64[i:i + 16] in text]

    for name, payload in forms.items():
        err = _error_message({"error": {"message": "bad request", "metadata": {"raw": '{"u": "' + payload + '"}'}}},
                             "")
        assert err.startswith("bad request | ") and leaked(err) == [], name
    plain = "see https://openrouter.ai/docs/api-reference/errors"
    assert _error_message({"error": {"message": plain}}, "") == plain  # ordinary text is kept
    tr = FakeTransport([{"status": 400, "data": {"error": {"message": "invalid image_url",
                                                           "metadata": {"raw": forms["json"]}}}}])
    r = provider(monkeypatch, tmp_path, tr, SpendGuard(cfg(), tmp_path / "l.jsonl")).call(request())
    assert r.status == "failed" and leaked(r.error) == [] and leaked(r.receipt["attempts"][0]["error"]) == []


def test_guard_wait_is_left_out_of_the_call_wall_s(monkeypatch, tmp_path):
    usage = {"v": 10.0}
    g = SpendGuard(cfg(drift_every_calls=1, drift_wait_s=0.3), tmp_path / "l.jsonl", key_usage=lambda: usage["v"],
                   sleep=time.sleep)
    g.record_key_usage_start(10.0)
    usage["v"] = 11.0  # the drift check after this call's reconcile pauses 0.3 s in the calling thread
    p = provider(monkeypatch, tmp_path, FakeTransport([ok()]), g, cache=False)
    t0 = time.perf_counter()
    r = p.call(request())
    total = time.perf_counter() - t0
    waited = r.receipt["guard_wait_s"]
    assert r.valid and g.stopped and waited >= 0.29 and r.receipt["attempts"][0]["guard_wait_s"] == waited
    assert r.wall_s == pytest.approx(total - waited, abs=0.1) and r.latency_s < 0.29


def test_one_drift_check_at_a_time(tmp_path):
    usage = {"v": 10.0}
    gate = threading.Event()
    slept = []

    def sleep(s):
        slept.append(s)
        assert gate.wait(30)

    led = tmp_path / "l.jsonl"
    g = SpendGuard(cfg(drift_every_calls=1), led, key_usage=lambda: usage["v"], sleep=sleep)
    g.record_key_usage_start(10.0)
    r1 = g.reserve(0.1, bucket="sweep")
    r2 = g.reserve(0.1, bucket="sweep")
    usage["v"] = 11.0
    checker = threading.Thread(target=g.reconcile, args=(r1, 0.05))
    checker.start()
    for _ in range(300):
        if g._paused:
            break
        time.sleep(0.01)
    assert g._paused
    other = threading.Thread(target=g.reconcile, args=(r2, 0.05))  # due too, but a check is running
    other.start()
    other.join(5)
    assert not other.is_alive() and slept == [60.0]
    gate.set()
    checker.join(10)
    assert g.stopped and g.spent() == pytest.approx(0.1)
    assert sum(1 for e in events(led) if e["event"] == "drift_pause") == 1


def test_legacy_200_with_an_error_object_is_retried_once(monkeypatch, tmp_path):
    tr = FakeTransport([error_200(), ok(text='{"segments": []}')])
    assert legacy_ask(legacy_provider(monkeypatch, tmp_path, tr, None), tmp_path).answer == '{"segments": []}'
    assert len(tr.bodies) == 2
    tr2 = FakeTransport([error_200(), error_200()])
    p2 = legacy_provider(monkeypatch, tmp_path, tr2, None)
    p2.cache = None
    with pytest.raises(RuntimeError, match="HTTP 200 with an error"):
        legacy_ask(p2, tmp_path, "l2.json")
    assert len(tr2.bodies) == 2
