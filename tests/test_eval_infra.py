"""Tests for the measurement-harness infrastructure: prices, cache key, response cache, receipts, the seal."""

from __future__ import annotations

import base64
import datetime as dt
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from robolabel.eval.heldout import (
    HeldoutGuard,
    HeldoutRefused,
    filter_dev,
    load_heldout_keys,
    normalized_text_sha256,
    require_explicit_episodes,
    write_heldout_ids,
)
from robolabel.eval.prices import PriceTable, UnknownModelPrice
from robolabel.eval.receipts import (
    JsonlWriter,
    ResponseCache,
    build_receipt,
    cache_key,
    canonical_json,
    sha256_bytes,
    sha256_text,
)

SWEEP_IDS = [
    "anthropic/claude-opus-5.5",
    "openai/gpt-6-sol",
    "openai/gpt-6-astra",
    "google/gemini-3.8-flash",
    "qwen/qwen3.8-max-0902",
    "meta/muse-spark-1.3",
    "xiaomi/mimo-v2.6-pro",
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-6-luna",
]
H = "a" * 64  # a well-formed SHA-256 hex digest


@pytest.fixture(scope="module")
def prices() -> PriceTable:
    return PriceTable.load()


# ---------------------------------------------------------------- prices


def test_unknown_model_raises(prices):
    with pytest.raises(UnknownModelPrice):
        prices.lookup("openrouter", "openai/gpt-7-nonexistent")
    with pytest.raises(UnknownModelPrice):
        prices.cost("openrouter", "qwen/qwen3.8-max-prime", 100, 100)
    with pytest.raises(UnknownModelPrice):
        prices.cost("nobody", "openai/gpt-6-sol", 100, 100)
    with pytest.raises(KeyError):  # UnknownModelPrice is a KeyError
        prices.worst_case("openrouter", "not/a-model", 10, 10)


def test_every_sweep_model_is_priced(prices):
    assert sorted(prices.known_models("openrouter")) == sorted(SWEEP_IDS)
    for model in SWEEP_IDS:
        rate = prices.lookup("openrouter", model)
        assert rate.price_in > 0 and rate.price_out > 0
        assert rate.source_url == "https://openrouter.ai/api/v1/models"
        assert rate.accessed == "2026-09-27"


def test_cost_exact_hand_computed(prices):
    # sol: 12,345 prompt tokens of which 2,000 cached, 678 output (reasoning included)
    # (10,345 * 2.00 + 2,000 * 0.20 + 678 * 10.00) / 1e6 = (20,690 + 400 + 6,780) / 1e6 = 0.02787
    assert prices.cost("openrouter", "openai/gpt-6-sol", 12_345, 678, cached_tokens=2_000) == 0.02787
    # mimo: (1,000 * 0.435 + 1,000 * 0.87) / 1e6 = 0.001305
    assert prices.cost("openrouter", "xiaomi/mimo-v2.6-pro", 1_000, 1_000) == 0.001305
    # opus: (1,000 * 4 + 500 * 20) / 1e6 = 0.014
    assert prices.cost("openrouter", "anthropic/claude-opus-5.5", 1_000, 500) == 0.014
    assert prices.cost("openrouter", "anthropic/claude-opus-5.5", 0, 0) == 0.0


def test_cost_rejects_bad_counts(prices):
    with pytest.raises(ValueError):
        prices.cost("openrouter", "openai/gpt-6-sol", 100, None)
    with pytest.raises(ValueError):
        prices.cost("openrouter", "openai/gpt-6-sol", -1, 10)
    with pytest.raises(ValueError):
        prices.cost("openrouter", "openai/gpt-6-sol", 100, 10, cached_tokens=101)


def test_batch_lookup(prices):
    rate = prices.lookup("openrouter", "openai/gpt-6-sol:batch")
    assert rate.batch and rate.entry == "openai/gpt-6-sol"
    assert (rate.price_in, rate.price_out) == (1.0, 5.0)
    assert (rate.sync_in, rate.sync_out) == (2.0, 10.0)
    assert prices.cost("openrouter", "openai/gpt-6-sol:batch", 1_000_000, 1_000_000) == 6.0
    assert prices.cost("openrouter", "openai/gpt-6-sol", 1_000_000, 1_000_000, batch=True) == 6.0
    # no cheaper batch variant: a real batch call is not priced
    with pytest.raises(UnknownModelPrice):
        prices.lookup("openrouter", "qwen/qwen3.8-max-0902:batch")


def test_deepseek_batch_equivalent_equals_sync(prices):
    m = "deepseek/deepseek-v4.1-flash"
    sync = prices.cost("openrouter", m, 40_000, 3_000, cached_tokens=1_000)
    assert prices.batch_equivalent("openrouter", m, 40_000, 3_000, cached_tokens=1_000) == sync
    # (39,000 * 0.035 + 1,000 * 0.001 + 3,000 * 0.29) / 1e6 = (1,365 + 1 + 870) / 1e6
    assert sync == 0.002236
    for m in ("qwen/qwen3.8-max-0902", "meta/muse-spark-1.3", "xiaomi/mimo-v2.6-pro"):
        assert prices.batch_equivalent("openrouter", m, 5_000, 700) == prices.cost("openrouter", m, 5_000, 700)


def test_batch_equivalent_uses_cheaper_batch(prices):
    m = "anthropic/claude-opus-5.5"
    assert prices.cost("openrouter", m, 1_000_000, 1_000_000) == 24.0
    assert prices.batch_equivalent("openrouter", m, 1_000_000, 1_000_000) == 12.0


def test_worst_case(prices):
    # gemini flash: (10,000 * 0.75 + 8,000 * 3.75) / 1e6 = 0.0375; a float estimate rounds up
    assert prices.worst_case("openrouter", "google/gemini-3.8-flash", 10_000, 8_000) == 0.0375
    assert prices.worst_case("openrouter", "google/gemini-3.8-flash", 9_999.2, 8_000) == 0.0375


def test_mock_is_free_and_jev_is_priced(prices):
    assert prices.cost("mock", "anything-at-all", 10_000, 5_000) == 0.0
    assert prices.cost("jev", "jev-1.13.0", 1_000_000, 1_000_000) == 0.042


def test_version_and_promo_window(prices):
    v = prices.version
    assert v.startswith("2026-09-27+") and len(v.split("+")[1]) == 12
    assert PriceTable.load().version == v
    flash = prices.lookup("openrouter", "google/gemini-3.8-flash")
    assert flash.effective_to == "2026-12-31"
    assert flash.is_effective("2026-09-27") and not flash.is_effective("2027-01-01")


def test_price_file_validation(tmp_path):
    bad = tmp_path / "prices.yaml"
    bad.write_text("version: '1'\nproviders:\n  p:\n    m:\n      price_in: 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        PriceTable.load(bad)  # price_out missing
    bad.write_text("version: '1'\nproviders:\n  p:\n    m:\n      price_in: 1.0\n      price_out: 2.0\n"
                   "      price_inn: 3.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        PriceTable.load(bad)  # typo in a field name
    good = tmp_path / "ok.yaml"
    good.write_text("version: '1'\nproviders:\n  p:\n    m:\n      price_in: 1.0\n      price_out: 2.0\n",
                    encoding="utf-8")
    table = PriceTable.load(good)
    assert table.cost("p", "m", 1_000_000, 1_000_000) == 3.0
    assert table.version == "1+" + sha256_bytes(good.read_bytes())[:12]


# ---------------------------------------------------------------- cache key


def test_cache_key_changes_and_is_stable():
    cfg = {"max_tokens": 8000, "reasoning": {"effort": "low", "exclude": True},
           "response_schema_sha256": H, "structured_mode": "json_schema_strict"}
    media = [sha256_bytes(b"frame-0"), sha256_bytes(b"frame-1"), sha256_bytes(b"frame-2")]
    base = cache_key("openrouter", "openai/gpt-6-luna", "Segment this episode.", cfg, media)
    assert base == cache_key("openrouter", "openai/gpt-6-luna", "Segment this episode.", dict(cfg), list(media))
    assert len(base) == 64
    assert cache_key("openrouter", "openai/gpt-6-luna", "Segment this episode!", cfg, media) != base
    changed = list(media)
    changed[1] = sha256_bytes(b"frame-1-other")
    assert cache_key("openrouter", "openai/gpt-6-luna", "Segment this episode.", cfg, changed) != base
    reordered = [media[1], media[0], media[2]]
    assert cache_key("openrouter", "openai/gpt-6-luna", "Segment this episode.", cfg, reordered) != base
    assert cache_key("openrouter", "openai/gpt-6-luna", "Segment this episode.", {**cfg, "max_tokens": 16000},
                     media) != base
    assert cache_key("openrouter", "openai/gpt-6-sol", "Segment this episode.", cfg, media) != base
    # the key is the SHA-256 of the canonical JSON of the five parts
    assert base == sha256_text(canonical_json(["openrouter", "openai/gpt-6-luna", "Segment this episode.",
                                               cfg, media]))
    with pytest.raises(TypeError):
        cache_key("openrouter", "openai/gpt-6-luna", "p", cfg, media[0])  # a single string is not a list


def test_canonical_json_is_canonical():
    assert canonical_json({"b": 1, "a": ["é", 2]}) == '{"a":["é",2],"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


# ---------------------------------------------------------------- response cache


def test_response_cache_round_trip(tmp_path):
    path = tmp_path / "llm_cache" / "responses.jsonl"
    cache = ResponseCache(path)
    assert cache.get(H) is None and cache.stats() == {"entries": 0, "corrupt_lines": 0}
    k2 = "b" * 64
    cache.put(H, {"response_text": "first", "usd": 0.00123456789})
    cache.put(k2, {"response_json": {"segments": []}})
    cache.put(H, {"response_text": "second"})  # a later put for the same key wins
    assert cache.get(H)["response_text"] == "second"
    assert cache.get(H)["cache_key"] == H and "stored_utc" in cache.get(H)
    reopened = ResponseCache(path)
    assert reopened.get(H)["response_text"] == "second"
    assert reopened.get(k2)["response_json"] == {"segments": []}
    assert reopened.stats() == {"entries": 2, "corrupt_lines": 0}
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # one file, one line per put
    assert json.loads(lines[0])["usd"] == 0.001235  # floats rounded to 6 decimals


def test_response_cache_survives_truncated_last_line(tmp_path):
    path = tmp_path / "responses.jsonl"
    cache = ResponseCache(path)
    cache.put(H, {"response_text": "one"})
    cache.put("c" * 64, {"response_text": "two"})
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"cache_key": "' + "d" * 64 + '", "response_te')  # crash mid-write
    reopened = ResponseCache(path)
    assert reopened.stats() == {"entries": 2, "corrupt_lines": 1}
    assert reopened.get("d" * 64) is None
    reopened.put("e" * 64, {"response_text": "three"})  # must not be glued onto the torn line
    again = ResponseCache(path)
    assert again.stats() == {"entries": 3, "corrupt_lines": 1}
    assert again.get("e" * 64)["response_text"] == "three"


def test_response_cache_refuses_image_payloads(tmp_path):
    path = tmp_path / "responses.jsonl"
    cache = ResponseCache(path)
    with pytest.raises(ValueError):
        cache.put(H, {"request": {"image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg"}})
    with pytest.raises(ValueError):
        cache.put(H, {"blob": "video/mp4;base64,AAAAIGZ0eXBpc29t"})
    with pytest.raises(ValueError):
        cache.put(H, {"response_text": "ok", "headers": {"x": "y"}})
    with pytest.raises(ValueError):
        cache.put("not-a-hash", {"response_text": "ok"})
    assert not path.exists() and cache.stats()["entries"] == 0


# ---------------------------------------------------------------- receipts


def _receipt_fields(**over):
    fields = dict(
        provider="openrouter", model="openai/gpt-6-luna", model_version="openai/gpt-6-luna-20260901",
        request_id="gen-123", cache_key=H, prompt_sha256=sha256_text("prompt"),
        inputs={"episode_key": "F1/0", "camera": "observation.images.up",
                "frame_indices": [0, 15, np.int64(30)], "media_resolution": [448, 336],
                "media_sha256": [sha256_bytes(b"jpeg-bytes")]},
        generation_config={"max_tokens": 8000, "reasoning": {"effort": "low"}},
        usage={"input_text_tokens": 900, "input_image_tokens": 1500, "cached_tokens": 0,
               "output_tokens": 400, "reasoning_tokens": 250},
        latency_s=3.14159265, retries=0, price_table_version="2026-09-27+abcdefabcdef",
        usd=0.000123456789, cache_hit=False, status="ok", response_text='{"segments": []}',
        arm="v@luna", step="segments",
    )
    fields.update(over)
    return fields


def test_build_receipt_has_spec_fields():
    r = build_receipt(**_receipt_fields())
    for name in ("provider", "model", "model_version", "request_id", "utc_time", "cache_key",
                 "prompt_sha256", "inputs", "generation_config", "usage", "latency_s", "retries",
                 "batch_job_id", "price_table_version", "usd", "cache_hit", "status", "response_text"):
        assert name in r
    assert r["batch_job_id"] is None and r["arm"] == "v@luna" and r["step"] == "segments"
    assert r["usage"]["input_video_tokens"] is None and r["usage"]["input_audio_tokens"] is None
    assert r["inputs"]["frame_indices"] == [0, 15, 30]
    assert r["usd"] == 0.000123 and r["latency_s"] == 3.141593
    json.dumps(r)  # plain JSON


def test_build_receipt_rejects_headers_and_payloads():
    with pytest.raises(ValueError):
        build_receipt(**_receipt_fields(headers={"Authorization": "x"}))
    with pytest.raises(ValueError):
        build_receipt(**_receipt_fields(generation_config={"max_tokens": 10, "Authorization": "x"}))
    with pytest.raises(ValueError):
        build_receipt(**_receipt_fields(response_text="data:image/png;base64,iVBORw0KGgo"))
    with pytest.raises(ValueError):
        build_receipt(**_receipt_fields(response_text="key " + "sk-or-" + "v1-0123456789abcdef"))
    with pytest.raises(ValueError):
        build_receipt(**_receipt_fields(usd=None))  # a cost is never null
    with pytest.raises(ValueError):
        build_receipt(**_receipt_fields(cache_key="abc"))
    fields = _receipt_fields()
    del fields["price_table_version"]
    with pytest.raises(ValueError):
        build_receipt(**fields)


# ---------------------------------------------------------------- JsonlWriter


def test_jsonl_writer_threads_and_determinism(tmp_path):
    path = tmp_path / "log" / "calls.jsonl"
    writer = JsonlWriter(path)

    def work(t):
        for i in range(25):
            writer.write({"thread": t, "i": i, "x": 1.23456789})

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    assert all(json.loads(line)["x"] == 1.234568 for line in lines)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for p in (a, b):
        JsonlWriter(p).write({"z": 1, "a": [np.float32(0.5), 2.0000001]})
    assert a.read_bytes() == b.read_bytes() == b'{"a":[0.5,2.0],"z":1}\n'


# ---------------------------------------------------------------- held-out guard


@pytest.fixture()
def seal(tmp_path):
    ids = tmp_path / "eval" / "heldout_ids.json"
    write_heldout_ids(ids, {
        "F1": {"repo_id": "lerobot/svla_so101_pickplace", "revision": "abc123", "rule": "legacy test list",
               "episode_keys": ["F1/7", "F1/2", "F1/6"]},
        "F3": {"repo_id": "armnet/armnetbench_v01_lerobot_so101", "revision": "def456", "rule": "second halves",
               "episode_keys": ["F3/1234"]},
        "F4": {"repo_id": "armnet/busybox_multitask", "revision": "0ab", "rule": "all dev", "episode_keys": []},
    }, created_utc="2026-09-27T09:00:00Z")
    prereg = tmp_path / "prereg" / "final_prereg.md"
    prereg.parent.mkdir(parents=True)
    prereg.write_bytes(b"# Final preregistration\r\nPM1 = T1 F1 at tau 5.   \r\n")
    log = tmp_path / "eval" / "heldout_access_log.jsonl"
    return ids, prereg, log


def _log_lines(log):
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


def test_heldout_ids_file_format(seal):
    ids, _, _ = seal
    assert load_heldout_keys(ids) == {"F1/2", "F1/6", "F1/7", "F3/1234"}
    doc = json.loads(ids.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "robolabel/heldout_ids/v1"
    assert doc["families"]["F1"]["episode_keys"] == ["F1/2", "F1/6", "F1/7"]
    bad = ids.parent / "bad.json"
    bad.write_text(json.dumps({"schema_version": "v0", "families": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_heldout_keys(bad)
    bad.write_text(json.dumps({"schema_version": "robolabel/heldout_ids/v1",
                               "families": {"F1": {"episode_keys": ["F3/2"]}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_heldout_keys(bad)
    with pytest.raises(FileNotFoundError):
        HeldoutGuard(ids.parent / "missing.json", ids.parent / "log.jsonl")


def test_guard_dev_only_logs_nothing(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    guard.check(["F1/0", "F1/1", "F1/3", "F4/5"], command="robolabel eval --split dev")
    assert not log.exists()


def test_guard_refuses_without_final(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    with pytest.raises(HeldoutRefused, match="2 held-out"):
        guard.check(["F1/0", "F1/2", "F1/06"], command="robolabel eval --split heldout", commit="c0ffee")
    lines = _log_lines(log)
    assert len(lines) == 1
    entry = lines[0]
    assert entry["outcome"] == "refused" and entry["reason"] == "no_final"
    assert entry["heldout_keys"] == ["F1/2", "F1/6"] and entry["commit"] == "c0ffee"
    assert entry["user"] and entry["command"] == "robolabel eval --split heldout" and entry["utc_time"]


def test_guard_refuses_wrong_hash(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    with pytest.raises(HeldoutRefused):
        guard.check(["F3/1234"], command="eval", final="0" * 64)
    with pytest.raises(HeldoutRefused):
        guard.check(["F3/1234"], command="eval", final="not-a-hash")
    no_prereg = HeldoutGuard(ids, log)  # no preregistration file: nothing can be allowed
    with pytest.raises(HeldoutRefused):
        no_prereg.check(["F3/1234"], command="eval", final=sha256_bytes(prereg.read_bytes()))
    lines = _log_lines(log)
    assert [x["outcome"] for x in lines] == ["refused"] * 3
    assert [x["reason"] for x in lines] == ["hash_mismatch", "hash_mismatch", "no_prereg_file"]
    assert lines[1]["final"] == "<not a sha256>"


def test_guard_allows_right_hash(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    raw_hash = sha256_bytes(prereg.read_bytes())
    assert guard.check(["F1/2", "F1/0"], command="eval --final", final=raw_hash) is None
    # the spec 7.3 normalized hash (LF, trailing whitespace stripped) identifies the same file
    norm = normalized_text_sha256(prereg)
    assert norm != raw_hash
    assert norm == sha256_text("# Final preregistration\nPM1 = T1 F1 at tau 5.\n")
    bom = prereg.parent / "bom.md"  # a BOM and trailing blank lines do not change the normalized hash
    bom.write_bytes(bytes([0xEF, 0xBB, 0xBF]) + b"# Final preregistration\nPM1 = T1 F1 at tau 5.\n\n\n")
    assert normalized_text_sha256(bom) == norm
    guard.check(["F1/7"], command="eval --final", final=norm.upper())
    guard.check(["F1/0"], command="eval --split dev")  # dev only: no line
    lines = _log_lines(log)
    assert [x["outcome"] for x in lines] == ["allowed_final", "allowed_final"]
    assert lines[0]["heldout_keys"] == ["F1/2"] and lines[0]["final"] == raw_hash


def test_guard_redacts_machine_paths_and_rejects_bad_keys(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    drive_path = "Z" + ":\\data\\gold\\v2\\F1.json"
    with pytest.raises(HeldoutRefused):
        guard.check(["F1/2"], command=f"robolabel eval --gold {drive_path} --out /home/user/x/out", commit="c")
    command = _log_lines(log)[0]["command"]
    assert command == "robolabel eval --gold <abs>/F1.json --out <abs>/out"
    with pytest.raises(ValueError):
        guard.check([2], command="eval")  # an int is not an episode key
    with pytest.raises(ValueError):
        guard.check("F1/2", command="eval")  # a single string is not a list
    n_lines = len(_log_lines(log))
    # F2 is not listed: by default its keys are dev (the task: no held-out key means return) with a warning
    with pytest.warns(RuntimeWarning, match="F2"):
        assert guard.check(["F2/5"], command="eval") is None
    strict = HeldoutGuard(ids, log, prereg, allow_unlisted_families=False)
    with pytest.raises(ValueError):
        strict.check(["F2/5"], command="eval")  # strict mode: dev cannot be told from held-out
    strict.check(["F4/5"], command="eval")  # F4 is listed with no held-out keys
    assert len(_log_lines(log)) == n_lines  # none of these calls involved a held-out key


def test_require_explicit_episodes():
    for bad in (range(8), None, 8, [], "F1/0", {"F1/0"}, (k for k in ["F1/0"]), ["F1/0", "F1/00"], [0, 1]):
        with pytest.raises(ValueError):
            require_explicit_episodes(bad)
    assert require_explicit_episodes(["F1/0", "F3/0012"]) == ["F1/0", "F3/12"]
    assert require_explicit_episodes(("F2/5",)) == ["F2/5"]


def test_filter_dev(seal):
    ids, prereg, log = seal
    heldout = load_heldout_keys(ids)
    keys = ["F1/0", "F1/2", "F1/3", "F3/1234", "F3/1"]
    assert filter_dev(keys, heldout) == ["F1/0", "F1/3", "F3/1"]
    assert filter_dev(keys, HeldoutGuard(ids, log, prereg)) == ["F1/0", "F1/3", "F3/1"]


# ================================================================ added by the verification pass

# Prices from the task text (live catalogue read 2026-09-27), independent of prices.yaml:
# model id -> (price_in, price_out, cached_in, batch_in, batch_out), USD per million tokens.
TASK_PRICES = {
    "anthropic/claude-opus-5.5": (4.0, 20.0, 0.2, 2.0, 10.0),
    "openai/gpt-6-sol": (2.0, 10.0, 0.2, 1.0, 5.0),
    "openai/gpt-6-astra": (10.0, 50.0, 1.0, 5.0, 25.0),
    "openai/gpt-6-luna": (0.10, 0.50, 0.01, 0.05, 0.25),
    "google/gemini-3.8-flash": (0.75, 3.75, 0.075, 0.375, 1.875),
    "qwen/qwen3.8-max-0902": (2.0, 6.0, 0.25, None, None),
    "meta/muse-spark-1.3": (1.25, 4.25, 0.15, None, None),
    "xiaomi/mimo-v2.6-pro": (0.435, 0.87, 0.0036, None, None),
    "deepseek/deepseek-v4.1-flash": (0.035, 0.29, 0.001, None, None),
}
SRC = Path(__file__).resolve().parents[1] / "src" / "robolabel"
INFRA_FILES = [SRC / "prices.yaml", SRC / "eval" / "prices.py", SRC / "eval" / "receipts.py",
               SRC / "eval" / "heldout.py"]


def test_price_table_matches_task_prices(prices):
    for model, (p_in, p_out, cached, b_in, b_out) in TASK_PRICES.items():
        rate = prices.lookup("openrouter", model)
        assert (rate.sync_in, rate.sync_out, rate.cached_in, rate.batch_in, rate.batch_out) == (
            p_in, p_out, cached, b_in, b_out), model
        assert rate.effective_to == ("2026-12-31" if model == "google/gemini-3.8-flash" else None)
    jev = prices.lookup("jev", "jev-1.13.0")
    assert (jev.sync_in, jev.sync_out, jev.accessed) == (0.042, 0.0, "2026-09-26")
    assert "typesafe.ai" in jev.source_url
    header = (SRC / "prices.yaml").read_text(encoding="utf-8")
    assert "USD per million tokens" in header and "272,000" in header and "1.50 / 7.50" in header


def test_costs_hand_computed_per_model(prices):
    c = prices.cost
    # luna: 200,000 * 0.10 + 50,000 * 0.01 + 20,000 * 0.50 = 20,000 + 500 + 10,000 = 30,500 -> 0.0305
    assert c("openrouter", "openai/gpt-6-luna", 250_000, 20_000, cached_tokens=50_000) == 0.0305
    # luna batch-equivalent: 200,000 * 0.05 + 500 + 20,000 * 0.25 = 10,000 + 500 + 5,000 -> 0.0155
    assert prices.batch_equivalent("openrouter", "openai/gpt-6-luna", 250_000, 20_000,
                                   cached_tokens=50_000) == 0.0155
    # sol batch-equivalent: 10,345 * 1.0 + 2,000 * 0.2 + 678 * 5.0 = 10,345 + 400 + 3,390 = 14,135
    assert prices.batch_equivalent("openrouter", "openai/gpt-6-sol", 12_345, 678, cached_tokens=2_000) == 0.014135
    # gemini flash, 57 images at 1,120 tokens plus 2,000 text: 65,840 * 0.75 + 3,000 * 3.75 = 49,380 + 11,250
    assert c("openrouter", "google/gemini-3.8-flash", 65_840, 3_000) == 0.06063
    # astra: 1,000 * 10 + 1,000 * 50 = 60,000
    assert c("openrouter", "openai/gpt-6-astra", 1_000, 1_000) == 0.06
    # muse: 2,500 * 1.25 + 500 * 0.15 + 1,000 * 4.25 = 3,125 + 75 + 4,250 = 7,450
    assert c("openrouter", "meta/muse-spark-1.3", 3_000, 1_000, cached_tokens=500) == 0.00745
    # qwen: 10,000 * 2 + 2,000 * 6 = 32,000
    assert c("openrouter", "qwen/qwen3.8-max-0902", 10_000, 2_000) == 0.032
    # every input token cached, no output: mimo 10,000 * 0.0036 = 36; deepseek 1e6 * 0.001 = 1,000
    assert c("openrouter", "xiaomi/mimo-v2.6-pro", 10_000, 0, cached_tokens=10_000) == 0.000036
    assert c("openrouter", "deepseek/deepseek-v4.1-flash", 1_000_000, 0, cached_tokens=1_000_000) == 0.001
    # jev lists no cached price, so cached tokens cost the input price
    assert c("jev", "jev-1.13.0", 1_000_000, 7, cached_tokens=500_000) == 0.042


def test_batch_edge_cases(prices):
    ds = "deepseek/deepseek-v4.1-flash"
    with pytest.raises(UnknownModelPrice):
        prices.lookup("openrouter", ds + ":batch")  # listed only at a dearer price, so not priced
    with pytest.raises(UnknownModelPrice):
        prices.cost("openrouter", ds, 10, 10, batch=True)
    assert prices.batch_equivalent("openrouter", ds + ":batch", 5_000, 700) == prices.cost("openrouter", ds,
                                                                                         5_000, 700)
    assert prices.lookup("mock", "anything:batch").price_in == 0.0
    # worst case of a batch id uses batch prices: 10,000 * 1.0 + 1,000 * 5.0 = 15,000
    assert prices.worst_case("openrouter", "openai/gpt-6-sol:batch", 10_000, 1_000) == 0.015
    # opus worst case: 100,000 * 4 + 16,000 * 20 = 720,000
    assert prices.worst_case("openrouter", "anthropic/claude-opus-5.5", 100_000, 16_000) == 0.72
    with pytest.raises(ValueError):
        prices.worst_case("openrouter", "anthropic/claude-opus-5.5", 100, None)  # never None


def test_cost_accepts_numpy_counts_and_rejects_bools(prices):
    sol = "openai/gpt-6-sol"
    assert prices.cost("openrouter", sol, np.int64(12_345), np.int64(678), cached_tokens=np.int32(2_000)) == 0.02787
    # 9,999.5 rounds up to 10,000: 10,000 * 2.00 = 20,000
    assert prices.worst_case("openrouter", sol, np.float64(9_999.5), np.int64(0)) == 0.02
    for bad in (True, np.bool_(True), "10", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            prices.cost("openrouter", sol, bad, 10)


def test_is_effective_accepts_dates_datetimes_and_timestamps(prices):
    flash = prices.lookup("openrouter", "google/gemini-3.8-flash")
    assert flash.is_effective(dt.datetime(2026, 12, 31, 23, 59))  # the last promotional day
    assert flash.is_effective("2026-12-31T23:59:59Z") and flash.is_effective(dt.date(2026, 9, 26))
    assert not flash.is_effective(dt.date(2027, 1, 1)) and not flash.is_effective("2026-09-25")
    assert prices.lookup("openrouter", "openai/gpt-6-sol").is_effective("2031-01-01")  # no end date


def test_infra_sources_are_clean():
    forbidden = ["C:" + "\\Users", "C:" + "/Users", "D:" + "\\", "D:" + "/"]
    for path in INFRA_FILES:
        raw = path.read_bytes()
        assert b"\r" not in raw, path.name  # LF only, so the price table version is the same everywhere
        text = raw.decode("ascii")  # ASCII only: no em dashes, no stray BOM characters
        assert not any(f in text for f in forbidden), path.name


def test_modules_import_without_optional_deps():
    code = ("import sys; import robolabel.eval.prices, robolabel.eval.receipts, robolabel.eval.heldout; "
            "print(sorted(m for m in ('scipy', 'jsonschema', 'requests') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    assert out.stdout.strip() == "[]"


def test_cache_key_edge_cases():
    cfg = {"max_tokens": 10, "reasoning": {"effort": "low"}}
    reordered_cfg = {"reasoning": {"effort": "low"}, "max_tokens": 10}
    assert cache_key("p", "m", "q", cfg, []) == cache_key("p", "m", "q", reordered_cfg, ())
    assert cache_key("p", "m", "q", cfg, ["h1"]) != cache_key("p", "m", "q", cfg, [])
    assert cache_key("p", "m", "", cfg, []) != cache_key("p", "m", " ", cfg, [])
    with pytest.raises(TypeError):
        cache_key("p", "m", "q", cfg, ["h", ""])  # an empty hash is not a hash
    with pytest.raises(TypeError):
        cache_key("p", "m", "q", None, [])


def test_response_cache_empty_and_middle_corruption(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert ResponseCache(empty).stats() == {"entries": 0, "corrupt_lines": 0}
    blank = tmp_path / "blank.jsonl"
    blank.write_bytes(b"\n\n  \n")
    assert ResponseCache(blank).stats() == {"entries": 0, "corrupt_lines": 0}
    path = tmp_path / "mixed.jsonl"
    cache = ResponseCache(path)
    cache.put(H, {"response_text": "one"})
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"response_text": "no key"}) + "\n")
    cache.put("b" * 64, {"response_text": "two"})
    reopened = ResponseCache(path)
    assert reopened.stats() == {"entries": 2, "corrupt_lines": 2}
    assert reopened.get(H)["response_text"] == "one" and reopened.get("b" * 64)["response_text"] == "two"
    got = reopened.get(H)
    got["response_text"] = "mutated"
    assert reopened.get(H)["response_text"] == "one"  # get returns a copy


def test_bare_base64_media_is_refused(tmp_path):
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + bytes(range(256)) * 2).decode("ascii")
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 2).decode("ascii")
    mp4 = base64.b64encode(b"\x00\x00\x00\x1cftypisom" + bytes(range(256)) * 2).decode("ascii")
    cache = ResponseCache(tmp_path / "c.jsonl")
    for blob in (jpeg, png, mp4):
        with pytest.raises(ValueError):  # Gemini-style inline_data: no data URL prefix
            cache.put(H, {"request": {"inline_data": {"mime_type": "image/jpeg", "data": blob}}})
        with pytest.raises(ValueError):
            build_receipt(**_receipt_fields(request_parts=[{"data": blob}]))
    assert not (tmp_path / "c.jsonl").exists()
    # hashes, ids and ordinary text are not mistaken for media
    ok = build_receipt(**_receipt_fields(response_text="The gripper closes. " * 200,
                                         media_ids=[sha256_bytes(bytes([i])) for i in range(40)]))
    assert ok["response_text"].startswith("The gripper")


def test_build_receipt_numpy_empty_usage_and_extras():
    r = build_receipt(**_receipt_fields(
        usage={"input_text_tokens": np.int64(900), "output_tokens": np.int32(12)},
        usd=np.float32(0.25), latency_s=np.float64(1.5), cache_hit=np.bool_(True)))
    assert type(r["usage"]["input_text_tokens"]) is int and r["usage"]["cached_tokens"] is None
    assert r["usd"] == 0.25 and r["cache_hit"] is True
    json.dumps(r)
    empty = build_receipt(**_receipt_fields(usage={}))
    assert set(empty["usage"]) == {"input_text_tokens", "input_image_tokens", "input_video_tokens",
                                   "input_audio_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"}
    assert all(v is None for v in empty["usage"].values())
    jev = build_receipt(**_receipt_fields(provider="jev", model="jev-1.13.0",
                                          usage={"input_tokens": 1_000, "output_tokens": 0},
                                          question_count=3, usd_batch_eq=0.000042))
    assert jev["usage"]["input_tokens"] == 1_000 and jev["question_count"] == 3
    assert jev["usd_batch_eq"] == 0.000042
    with_json = build_receipt(**_receipt_fields(response_json={"segments": []}))
    assert with_json["response_json"] == {"segments": []}
    fields = _receipt_fields()
    del fields["response_text"]
    assert build_receipt(**fields)["response_text"] is None
    with pytest.raises(ValueError):
        build_receipt()
    for bad in ({"usage": {"output_tokens": -1}}, {"usage": {"output_tokens": 1.5}}, {"usd": -0.01},
                {"usd": float("nan")}, {"retries": -1}, {"cache_hit": "no"}, {"inputs": None},
                {"usage": {"x-api-key": 1}}, {"cookie": "a=b"}):
        with pytest.raises(ValueError):
            build_receipt(**_receipt_fields(**bad))


def test_build_receipt_is_deterministic(tmp_path):
    fields = _receipt_fields(utc_time="2026-09-27T10:00:00Z")
    a, b = build_receipt(**fields), build_receipt(**fields)
    assert canonical_json(a) == canonical_json(b)
    for name in ("a.jsonl", "b.jsonl"):
        JsonlWriter(tmp_path / name).write(build_receipt(**fields))
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert fields["inputs"]["frame_indices"][2] == np.int64(30)  # the caller's dicts are not changed


def test_jsonl_writer_refuses_bad_objects(tmp_path):
    path = tmp_path / "x.jsonl"
    with pytest.raises(TypeError):
        JsonlWriter(path).write(["not", "a", "dict"])
    with pytest.raises(ValueError):
        JsonlWriter(path).write({"jpeg": b"\xff\xd8\xff"})  # raw bytes never reach JSON
    assert not path.exists()


def test_guard_empty_single_and_generator_inputs(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    assert guard.check([], command="eval") is None
    assert guard.check((k for k in ["F1/0", "F3/1"]), command="eval") is None
    assert not log.exists()  # dev-only and empty calls log nothing
    with pytest.raises(ValueError):
        guard.check(None, command="eval")
    with pytest.raises(ValueError):
        guard.check(range(8), command="eval")  # ints are not episode keys
    with pytest.raises(HeldoutRefused, match="1 held-out"):
        guard.check(["F3/1234"], command="eval", commit="abc")
    assert issubclass(HeldoutRefused, PermissionError)
    assert guard.heldout_in(["F1/7", "F1/0", "F1/2"]) == ["F1/2", "F1/7"]


def test_guard_log_entry_fields(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    with pytest.raises(HeldoutRefused):
        guard.check(["F1/2", "F1/0", "F1/2"], command="eval", commit="abc")
    guard.check(["F1/6"], command="eval --final", final=sha256_bytes(prereg.read_bytes()), commit="abc")
    refused, allowed = _log_lines(log)
    expected = {"utc_time", "command", "commit", "user", "heldout_keys", "n_heldout", "n_keys", "outcome",
                "reason", "final", "heldout_ids_sha256"}
    assert set(refused) == expected and set(allowed) == expected
    assert (refused["n_heldout"], refused["n_keys"], refused["final"]) == (1, 2, None)
    assert (allowed["outcome"], allowed["reason"]) == ("allowed_final", None)
    assert refused["heldout_ids_sha256"] == sha256_bytes(ids.read_bytes())
    assert len(refused["utc_time"]) == 20 and refused["utc_time"].endswith("Z")
    assert log.read_bytes().count(b"\n") == 2 and b"\r" not in log.read_bytes()


def test_guard_fails_closed_when_log_or_prereg_is_unreadable(seal, monkeypatch):
    ids, prereg, log = seal
    blocked = log.parent / "log_is_a_dir"
    blocked.mkdir()
    guard = HeldoutGuard(ids, blocked, prereg)
    with pytest.raises(HeldoutRefused, match="could not be written"):
        guard.check(["F1/2"], command="eval")
    with pytest.raises(OSError) as info:  # allowed, but no access without its log line
        guard.check(["F1/2"], command="eval", final=sha256_bytes(prereg.read_bytes()))
    assert not isinstance(info.value, HeldoutRefused)

    import robolabel.eval.heldout as heldout_mod

    def unreadable(path):
        raise PermissionError("locked")

    monkeypatch.setattr(heldout_mod, "file_sha256", unreadable)
    with pytest.raises(HeldoutRefused, match="prereg_unreadable"):
        HeldoutGuard(ids, log, prereg).check(["F1/2"], command="eval", final="0" * 64)
    assert _log_lines(log)[-1]["reason"] == "prereg_unreadable"


def test_redaction_of_shell_style_paths(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    # Git Bash (/c/...) and WSL (/mnt/<drive>/...) spellings of drive paths, with neutral folders.
    command = ("robolabel eval --gold /c/data/gold/F1.json --out /mnt/e/data/runs/r1 "
               "--cfg /root/x.yaml --prices src/robolabel/prices.yaml --systems v/b2b "
               "--src https://openrouter.ai/api/v1/models")
    with pytest.raises(HeldoutRefused):
        guard.check(["F1/2"], command=command, commit="c")
    assert _log_lines(log)[0]["command"] == (
        "robolabel eval --gold <abs>/F1.json --out <abs>/r1 --cfg <abs>/x.yaml --prices src/robolabel/prices.yaml "
        "--systems v/b2b --src https://openrouter.ai/api/v1/models")


def test_write_heldout_ids_is_byte_identical(tmp_path):
    fams = {"F3": {"repo_id": "r", "revision": "s", "rule": "x", "episode_keys": ["F3/10", "F3/2", "F3/02"]},
            "F1": {"repo_id": "r", "revision": "s", "rule": "y", "episode_keys": []}}
    h1 = write_heldout_ids(tmp_path / "a.json", fams, created_utc="2026-09-27T09:00:00Z")
    h2 = write_heldout_ids(tmp_path / "b.json", dict(reversed(list(fams.items()))),
                           created_utc="2026-09-27T09:00:00Z")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes() and h1 == h2
    assert h1 == sha256_bytes((tmp_path / "a.json").read_bytes())
    assert load_heldout_keys(tmp_path / "a.json") == {"F3/2", "F3/10"}


def test_require_explicit_episodes_more_cases():
    for bad in (True, np.array(["F1/0"]), ["F1"], ["F1/-1"], ["/3"], [None]):
        with pytest.raises(ValueError):
            require_explicit_episodes(bad)
    assert require_explicit_episodes(["F1/5"]) == ["F1/5"]  # a one-key list is fine
    assert filter_dev([], {"F1/2"}) == [] and filter_dev(["F1/2", "F1/3"], set()) == ["F1/2", "F1/3"]


def test_real_heldout_ids_file_seals_the_f1_prefix():
    path = Path(__file__).resolve().parents[1] / "eval" / "heldout_ids.json"
    if not path.exists():
        pytest.skip("eval/heldout_ids.json not written yet")
    keys = load_heldout_keys(path)
    prefix = {f"F1/{i}" for i in range(8)}
    assert prefix & keys == {"F1/2", "F1/6", "F1/7"}  # spec 2.3: why range(limit) is not allowed
    assert sum(1 for k in keys if k.startswith("F1/")) == 20
