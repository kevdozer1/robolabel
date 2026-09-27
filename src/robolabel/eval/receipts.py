"""Receipts, the response cache key and the response cache (MEASUREMENT_SPEC 9.1 and 9.2).

- :func:`cache_key` is the SHA-256 of the provider, model, prompt text, generation config and the
  ordered input media hashes. A cached response is reused only on an exact key match (the legacy
  Gemini cache matched on output path and model only, so a changed prompt silently reused old
  answers).
- :class:`ResponseCache` keeps every cached response in ONE append-only JSONL file, never a file per
  call: the data drive is exFAT with 512 KB clusters (STORAGE_AND_ENV section 1). A later line for
  the same key wins.
- :func:`build_receipt` assembles one receipt with the spec 9.1 field names and refuses request
  headers, API keys and image or video payloads.
- :class:`JsonlWriter` appends one JSON object per line, thread-safe, flushed and fsynced.

JSON written by this module has sorted keys, compact separators, UTF-8 text and floats rounded to 6
decimals, so identical input gives identical bytes. Nothing here makes a network call.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FLOAT_DIGITS = 6

# Spec 9.1 field names, in the order a receipt lists them.
RECEIPT_FIELDS: tuple[str, ...] = (
    "provider", "model", "model_version", "request_id", "utc_time", "cache_key", "prompt_sha256",
    "inputs", "generation_config", "usage", "latency_s", "retries", "batch_job_id",
    "price_table_version", "usd", "cache_hit", "status",
)
RESPONSE_FIELDS: tuple[str, ...] = ("response_text", "response_json")
INPUT_FIELDS: tuple[str, ...] = ("episode_key", "camera", "frame_indices", "media_resolution", "media_sha256")
USAGE_FIELDS: tuple[str, ...] = (
    "input_text_tokens", "input_image_tokens", "input_video_tokens", "input_audio_tokens",
    "cached_tokens", "output_tokens", "reasoning_tokens",
)
_REQUIRED = (
    "provider", "model", "cache_key", "prompt_sha256", "inputs", "generation_config", "usage",
    "price_table_version", "usd", "status",
)
_DEFAULTS: dict[str, Any] = {
    "model_version": None, "request_id": None, "latency_s": None, "retries": 0,
    "batch_job_id": None, "cache_hit": False,
}

# Keys that would carry request headers or credentials. Compared lowercased, "_" read as "-".
FORBIDDEN_KEYS = frozenset({
    "authorization", "proxy-authorization", "headers", "request-headers", "response-headers",
    "api-key", "apikey", "x-api-key", "cookie", "set-cookie",
})
# Markers of an inline image or video payload (a data URL or any base64 blob with a media prefix).
_MEDIA_MARKERS = ("data:image", "base64,")
# A bare base64 payload with no data URL prefix (Gemini-style inline_data): the base64 form of a JPEG,
# PNG, GIF, RIFF (WebP, AVI, WAV), MP4 or WebM file header, followed by a long base64 run.
_BARE_MEDIA_RE = re.compile(
    r"(?<![A-Za-z0-9+/])(?:/9j/|iVBORw0KGgo|R0lGOD|UklGR|AAAA[A-Za-z0-9+/]{2}Z0eX|GkXfo)[A-Za-z0-9+/]{64,}"
)
# Credential shapes that must never reach a receipt or the cache (the OpenRouter key, a bearer token).
_SECRET_PATTERNS = (
    re.compile(r"sk-or-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# One lock per file, shared by every writer in this process that appends to that file.
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


# ---------------------------------------------------------------------------
# hashing and canonical JSON
# ---------------------------------------------------------------------------


def sha256_bytes(b: bytes) -> str:
    """SHA-256 hex digest of ``b``."""
    return hashlib.sha256(bytes(b)).hexdigest()


def sha256_text(s: str) -> str:
    """SHA-256 hex digest of ``s`` encoded as UTF-8."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    """Serialize numpy scalars and arrays as plain Python; refuse raw bytes."""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raise ValueError("raw bytes cannot be written to JSON here (no image or video bytes in records)")
    if type(obj).__module__ == "numpy" and hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, separators ``(",", ":")``, non-ASCII kept, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                      default=_json_default)


def round_floats(obj: Any, ndigits: int = FLOAT_DIGITS) -> Any:
    """Copy of ``obj`` with every float rounded to ``ndigits`` decimals (tuples become lists)."""
    if type(obj).__module__ == "numpy" and hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits) + 0.0 if math.isfinite(obj) else obj  # + 0.0 turns -0.0 into 0.0
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_sha256_hex(value: Any) -> bool:
    """True for a 64-character lowercase hex string."""
    return isinstance(value, str) and _HEX64.match(value) is not None


# ---------------------------------------------------------------------------
# cache key
# ---------------------------------------------------------------------------


def cache_key(provider: str, model: str, prompt_text: str, generation_config: dict[str, Any],
              media_hashes: list[str]) -> str:
    """Spec 9.2 cache key: SHA-256 of ``canonical_json([provider, model, prompt_text,
    generation_config, media_hashes])``.

    ``generation_config`` holds everything that shapes the answer besides the prompt and media: the
    response schema hash, max tokens, reasoning settings, structured-output mode (and temperature
    when one is sent). ``media_hashes`` are the SHA-256 digests of the image or video byte payloads
    in the order they are sent; the order is part of the key.
    """
    if not isinstance(provider, str) or not isinstance(model, str):
        raise TypeError("provider and model must be strings")
    if not isinstance(prompt_text, str):
        raise TypeError("prompt_text must be a string")
    if not isinstance(generation_config, dict):
        raise TypeError("generation_config must be a dict")
    if isinstance(media_hashes, (str, bytes)) or not isinstance(media_hashes, (list, tuple)):
        raise TypeError("media_hashes must be a list of hash strings, in send order")
    if not all(isinstance(h, str) and h for h in media_hashes):
        raise TypeError("every media hash must be a non-empty string")
    return sha256_text(canonical_json([provider, model, prompt_text, generation_config, list(media_hashes)]))


# ---------------------------------------------------------------------------
# payload checks
# ---------------------------------------------------------------------------


def check_no_media_payload(text: str) -> None:
    """Raise ValueError when serialized JSON ``text`` carries an inline image or base64 payload.

    Refused: any ``data:image`` or ``base64,`` marker (any case), and a bare base64 string that
    starts with an image or video file header (``/9j/`` for JPEG, ``iVBORw0KGgo`` for PNG, ...).
    """
    lowered = text.lower()
    for marker in _MEDIA_MARKERS:
        if marker in lowered:
            raise ValueError(f"record contains {marker!r}: image or video bytes never go into receipts or the cache")
    if _BARE_MEDIA_RE.search(text):
        raise ValueError("record contains a base64 image or video payload: image or video bytes never go into "
                         "receipts or the cache")


def check_no_secrets(text: str) -> None:
    """Raise ValueError when ``text`` looks like it carries an API key or a bearer token."""
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError("record looks like it contains an API key or bearer token; refusing to write it")


def _forbidden_keys(obj: Any, path: str = "") -> list[str]:
    """Dotted paths of every header-like or credential key anywhere in ``obj``."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k).lower().replace("_", "-") in FORBIDDEN_KEYS:
                found.append(here)
            found.extend(_forbidden_keys(v, here))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(_forbidden_keys(v, f"{path}[{i}]"))
    return found


# ---------------------------------------------------------------------------
# append-only JSONL
# ---------------------------------------------------------------------------


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.resolve()))
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.Lock())


def _ends_without_newline(path: Path) -> bool:
    """True when ``path`` is a non-empty file whose last byte is not a newline (a torn last line)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return False
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) != b"\n"
    except FileNotFoundError:
        return False


def _append_line(path: Path, text: str) -> None:
    """Append ``text`` as one line, flush and fsync. Call with the file's lock held.

    A torn last line (from a crash mid-write) is closed with a newline first, so it stays one
    corrupt line instead of swallowing this record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\n" if _ends_without_newline(path) else ""
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(prefix + text + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def dumps_line(obj: Any, float_digits: int | None = FLOAT_DIGITS) -> str:
    """One JSONL line (no newline): floats rounded, then canonical JSON."""
    if float_digits is not None:
        obj = round_floats(obj, float_digits)
    return canonical_json(obj)


class JsonlWriter:
    """Thread-safe append of one JSON object per line (sorted keys), flushed and fsynced per line.

    Floats are rounded to ``float_digits`` decimals (None keeps them as they are). The file and its
    parent folder are created on the first write. Appends from several processes to one file are
    not coordinated; give each process its own file.
    """

    def __init__(self, path: str | Path, *, float_digits: int | None = FLOAT_DIGITS):
        self.path = Path(path)
        self.float_digits = float_digits
        self._lock = _lock_for(self.path)

    def write(self, obj: dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            raise TypeError("JsonlWriter.write takes a dict")
        text = dumps_line(obj, self.float_digits)
        with self._lock:
            _append_line(self.path, text)


# ---------------------------------------------------------------------------
# response cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Cached responses keyed by :func:`cache_key`, stored in one append-only JSONL file.

    Opening reads every line into memory; a truncated or corrupt line (a crash mid-write) is skipped
    and counted in :meth:`stats`. :meth:`put` appends ``{"cache_key", "stored_utc", ...record}`` and
    fsyncs it; a later put for the same key wins. Records never hold image bytes or base64 data URLs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = _lock_for(self.path)
        self._records: dict[str, dict[str, Any]] = {}
        self._corrupt_lines = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw in self.path.read_bytes().split(b"\n"):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except ValueError:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
                self._corrupt_lines += 1
                continue
            key = rec.get("cache_key") if isinstance(rec, dict) else None
            if not is_sha256_hex(key):
                self._corrupt_lines += 1
                continue
            self._records[key] = rec

    def get(self, key: str) -> dict[str, Any] | None:
        """The stored record for ``key`` (with ``cache_key`` and ``stored_utc``), or None."""
        with self._lock:
            rec = self._records.get(key)
        return copy.deepcopy(rec) if rec is not None else None

    def put(self, key: str, record: dict[str, Any]) -> dict[str, Any]:
        """Append ``record`` under ``key`` and return the stored line as a dict.

        Raises ValueError for a malformed key, a record whose ``cache_key`` disagrees with ``key``,
        or a record that carries image or video payloads, headers or credentials.
        """
        if not is_sha256_hex(key):
            raise ValueError("cache key must be a 64-character lowercase SHA-256 hex digest")
        if not isinstance(record, dict):
            raise TypeError("record must be a dict")
        if "cache_key" in record and record["cache_key"] != key:
            raise ValueError("record['cache_key'] disagrees with the key it is stored under")
        bad = _forbidden_keys(record)
        if bad:
            raise ValueError(f"record contains header-like or credential keys: {', '.join(bad)}")
        line = round_floats({**record, "cache_key": key, "stored_utc": utc_now()})
        text = canonical_json(line)
        check_no_media_payload(text)
        check_no_secrets(text)
        with self._lock:
            _append_line(self.path, text)
            self._records[key] = line
        return copy.deepcopy(line)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def stats(self) -> dict[str, int]:
        """``{"entries": distinct keys held, "corrupt_lines": lines skipped on open}``."""
        with self._lock:
            return {"entries": len(self._records), "corrupt_lines": self._corrupt_lines}


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------


def _is_bool(value: Any) -> bool:
    """A Python bool or a numpy bool (named ``bool_`` before numpy 2, ``bool`` from numpy 2)."""
    kind = type(value)
    return isinstance(value, bool) or (kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_"))


def _is_count(value: Any) -> bool:
    """A non-negative Python or numpy integer (not a bool)."""
    return isinstance(value, numbers.Integral) and not _is_bool(value) and value >= 0


def _is_amount(value: Any) -> bool:
    """A finite non-negative Python or numpy number (not a bool)."""
    return (isinstance(value, numbers.Real) and not _is_bool(value)
            and math.isfinite(value) and value >= 0)


def _normalize_inputs(inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a dict (episode_key, camera, frame_indices, media_resolution, media_sha256)")
    out = {name: inputs.get(name) for name in INPUT_FIELDS}
    out.update({k: v for k, v in inputs.items() if k not in out})  # clip start, end, fps and the like
    hashes = out["media_sha256"]
    if hashes is None:
        hashes = []
    if isinstance(hashes, (str, bytes)) or not isinstance(hashes, (list, tuple)):
        raise ValueError("inputs.media_sha256 must be a list of SHA-256 hex digests")
    if not all(isinstance(h, str) and h for h in hashes):
        raise ValueError("inputs.media_sha256 must hold non-empty strings")
    out["media_sha256"] = list(hashes)
    return out


def _normalize_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        raise ValueError("usage must be a dict of token counts")
    out = {name: usage.get(name) for name in USAGE_FIELDS}
    out.update({k: v for k, v in usage.items() if k not in out})  # e.g. Jev's input_tokens, question_count
    for name in USAGE_FIELDS:
        if out[name] is not None and not _is_count(out[name]):
            raise ValueError(f"usage.{name} must be a non-negative int or None, got {out[name]!r}")
    return out


def build_receipt(**fields: Any) -> dict[str, Any]:
    """One spec 9.1 receipt from keyword fields.

    Required: provider, model, cache_key, prompt_sha256, inputs, generation_config, usage,
    price_table_version, usd, status. Defaults: model_version, request_id, latency_s and
    batch_job_id None, retries 0, cache_hit False, utc_time now. At least one of response_text and
    response_json is always present (response_text None when neither is given). ``inputs`` and
    ``usage`` always carry every spec key (None when unknown); unknown usage counts stay None, not 0.

    Any other keyword (arm, step, bucket, structured_mode, finish_reason, provider_name,
    or_latency_ms, generation_time_ms, wall_s, attempts, ...) is kept as an extra key.

    Raises ValueError for a missing required field, a null or negative ``usd`` (a cost is never
    null), malformed hashes, header-like or credential keys anywhere, or an image or base64 payload.
    Floats are rounded to 6 decimals.
    """
    missing = [name for name in _REQUIRED if name not in fields]
    if missing:
        raise ValueError(f"receipt is missing required fields: {', '.join(missing)}")
    receipt: dict[str, Any] = {}
    for name in RECEIPT_FIELDS:
        if name in fields:
            receipt[name] = fields[name]
        elif name == "utc_time":
            receipt[name] = utc_now()
        else:
            receipt[name] = _DEFAULTS[name]
    responses = [name for name in RESPONSE_FIELDS if name in fields]
    for name in responses or ["response_text"]:
        receipt[name] = fields.get(name)
    for name in sorted(fields):
        if name not in receipt:
            receipt[name] = fields[name]

    for name in ("provider", "model", "status", "price_table_version", "utc_time"):
        if not isinstance(receipt[name], str) or not receipt[name]:
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("cache_key", "prompt_sha256"):
        if not is_sha256_hex(receipt[name]):
            raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    if not isinstance(receipt["generation_config"], dict):
        raise ValueError("generation_config must be a dict")
    if not _is_amount(receipt["usd"]):
        raise ValueError(f"usd must be a finite non-negative number (never null), got {receipt['usd']!r}")
    if receipt["latency_s"] is not None and not _is_amount(receipt["latency_s"]):
        raise ValueError("latency_s must be a non-negative number or None")
    if not _is_count(receipt["retries"]):
        raise ValueError("retries must be a non-negative int")
    if not _is_bool(receipt["cache_hit"]):
        raise ValueError("cache_hit must be a bool")
    receipt["inputs"] = _normalize_inputs(receipt["inputs"])
    receipt["usage"] = _normalize_usage(receipt["usage"])

    bad = _forbidden_keys(receipt)
    if bad:
        raise ValueError(f"receipt contains header-like or credential keys: {', '.join(bad)}")
    receipt = round_floats(receipt)
    text = canonical_json(receipt)
    check_no_media_payload(text)
    check_no_secrets(text)
    return receipt
