"""OpenRouter provider (OpenAI-compatible chat completions), one file.

Credential: ``OPENROUTER_API_KEY``. Two entry points:

* ``ask``: the legacy contract (one contact sheet plus a question, the answer as text), used by the
  existing labelers (the B2b arm).
* ``call``: a multi-part request with separate images, a JSON schema and reasoning settings, used by
  the schema v7 layers. Structured output tries ``json_schema`` strict first, then ``json_object``
  with the schema in the prompt, then plain text with the schema in the prompt; the mode is recorded.

Every HTTP attempt goes through an optional :class:`~robolabel.spend_guard.SpendGuard` (reserve the
worst case first; reconcile with ``usage.cost``; a lost response stays reserved as spent). Responses
are cached by the MEASUREMENT_SPEC 9.2 key in one JSONL file, so a rerun costs nothing, and every call
appends one receipt line (spec 9.1) to a JSONL file. Receipts, logs and exceptions never contain the
key, request headers or image bytes.

Error handling follows MODEL_SWEEP section 3: a 400 naming the schema or ``response_format`` and a
404/503 "no endpoints" move to the next structured-output mode; a 400 about images is a model limit;
402 is handled by ``limit_source``; 429 and 5xx retry up to 3 times (1.5 s doubling, at most 20 s);
a timeout retries once; a 200 with an error, empty content or ``finish_reason`` "error" retries once;
``finish_reason`` "length" retries once with ``max_tokens`` doubled (at most 16,000); invalid JSON
gets one repair retry that sends the validation error back.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    CallRequest,
    CallResult,
    ImagePart,
    ProviderResponse,
    TextPart,
    VideoPart,
    VLMProvider,
    load_secret,
    make_contact_sheet,
    register_provider,
    schema_sha256,
    write_receipt,
)

API_BASE = "https://openrouter.ai/api/v1"
MODES = ("json_schema_strict", "json_object_with_schema_in_prompt", "plain_with_schema_in_prompt")
RETRY_STATUSES = {429, 500, 502, 504, 520, 522, 524}
MAX_TOKENS_CAP = 16000


class _Unavailable(Exception):
    """The model has no provider for this parameter combination (after the last mode)."""


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _b64_jpeg(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class _RequestsTransport:
    """Default HTTP transport. Returns (status, json or None, text, headers); raises TimeoutError.

    ``timeout`` is the read timeout (no byte for that long). OpenRouter keeps long non-streamed
    generations alive with filler bytes, so a separate total deadline (``total_deadline_s``) bounds
    every POST; a POST past it is abandoned and reported as a timeout.
    """

    def __init__(self, total_deadline_s: float = 1200.0) -> None:
        import requests

        self._requests = requests
        self._session = requests.Session()
        self.total_deadline_s = total_deadline_s

    def post(self, url: str, body: dict[str, Any], headers: dict[str, str], timeout: float):
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["r"] = self._requests.post(url, json=body, headers=headers, timeout=timeout)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                box["e"] = exc

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(self.total_deadline_s)
        if th.is_alive():
            raise TimeoutError(f"no complete response within {self.total_deadline_s:.0f} s")
        exc = box.get("e")
        if exc is not None:
            if isinstance(exc, self._requests.Timeout):
                raise TimeoutError("request timed out") from exc
            if isinstance(exc, self._requests.RequestException):
                raise ConnectionError(type(exc).__name__) from exc
            raise exc
        return _parse(box["r"])

    def get(self, url: str, headers: dict[str, str], timeout: float):
        try:
            r = self._session.get(url, headers=headers, timeout=timeout)
        except self._requests.Timeout as exc:
            raise TimeoutError("request timed out") from exc
        except self._requests.RequestException as exc:
            raise ConnectionError(type(exc).__name__) from exc
        return _parse(r)


def _parse(r) -> tuple[int, Any, str, dict[str, str]]:
    text = r.text
    try:
        data = r.json()
    except ValueError:
        data = None
    return int(r.status_code), data, text, {k.lower(): v for k, v in r.headers.items()}


def _error_message(data: Any, text: str) -> str:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        meta = err.get("metadata")
        raw = ""
        if isinstance(meta, dict) and meta.get("raw"):
            raw = f" | {str(meta.get('raw'))[:300]}"
        return f"{err.get('message', '')}{raw}"[:600]
    return (text or "")[:400]


def _limit_source(data: Any) -> str | None:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        meta = data["error"].get("metadata")
        if isinstance(meta, dict):
            return meta.get("limit_source")
    return None


class OpenRouterProvider(VLMProvider):
    name = "openrouter"

    def __init__(self, model: str | None = None, *, timeout_seconds: float = 240.0,
                 guard: Any = None, cache: Any = None, receipts: Any = None,
                 prices: dict[str, tuple[float, float]] | None = None,
                 max_price_factor: float = 1.5, transport: Any = None,
                 sleep: Callable[[float], None] = time.sleep, price_table_version: str = "",
                 legacy_max_tokens: int = 8000, legacy_reasoning: dict[str, Any] | None = None):
        super().__init__(model=model or os.environ.get("ROBOVID_MODEL") or "google/gemini-3.8-flash")
        self.timeout_seconds = timeout_seconds
        self.api_key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
        self.guard = guard
        self.cache = cache
        self.receipts = receipts
        self.prices = prices or {}
        self.max_price_factor = max_price_factor
        self.transport = transport or _RequestsTransport()
        self._sleep = sleep
        self.price_table_version = price_table_version
        self.context: dict[str, Any] = {}  # arm, episode_key, bucket, model_key for legacy ask()
        self.legacy_max_tokens = legacy_max_tokens
        self.legacy_reasoning = legacy_reasoning if legacy_reasoning is not None else {"effort": "low",
                                                                                        "exclude": True}
        self.in_flight_backoff = threading.Event()  # set when OpenRouter's in-flight budget is full

    # ------------------------------------------------------------------ helpers
    def _headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
                "X-Title": "robolabel"}

    def price(self, model: str | None = None) -> tuple[float, float]:
        """(input, output) USD per million tokens. Raises KeyError for an unknown model."""
        m = model or self.model
        if m not in self.prices:
            raise KeyError(f"no price for {m}")
        return self.prices[m]

    def worst_case(self, text_chars: int, n_images: int, image_tokens: float, max_tokens: int) -> float:
        pin, pout = self.price()
        tokens_in = text_chars / 3.5 + n_images * image_tokens
        return tokens_in * pin / 1e6 + max_tokens * pout / 1e6

    def _payload(self, req: CallRequest, mode: str, max_tokens: int,
                 extra_messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for p in req.parts:
            if isinstance(p, ImagePart):
                content.append({"type": "image_url", "image_url": {"url": _b64_jpeg(p.jpeg)}})
            elif isinstance(p, VideoPart):
                url = f"data:{p.mime};base64," + base64.b64encode(p.data).decode("ascii")
                content.append({"type": "video_url", "video_url": {"url": url}})
            else:
                content.append({"type": "text", "text": p.text if isinstance(p, TextPart) else str(p)})
        if mode != "json_schema_strict":
            content.append({"type": "text", "text": "Return only JSON that matches this JSON Schema:\n"
                            + json.dumps(req.schema, separators=(",", ":"))})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": ([{"role": "system", "content": req.system}] if req.system else [])
            + [{"role": "user", "content": content}, *(extra_messages or [])],
            "max_tokens": int(max_tokens),
            "usage": {"include": True},
        }
        if req.reasoning is not None:
            body["reasoning"] = req.reasoning
        if mode == "json_schema_strict":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": req.schema_name, "strict": True, "schema": req.schema}}
        elif mode == "json_object_with_schema_in_prompt":
            body["response_format"] = {"type": "json_object"}
        prov: dict[str, Any] = {"require_parameters": True, "allow_fallbacks": True}
        if self.model in self.prices:
            pin, pout = self.prices[self.model]
            prov["max_price"] = {"prompt": round(pin * self.max_price_factor, 6),
                                 "completion": round(pout * self.max_price_factor, 6)}
        body["provider"] = prov
        return body

    @staticmethod
    def _prompt_text(req: CallRequest) -> str:
        texts = [req.system] + [p.text for p in req.parts if isinstance(p, TextPart)]
        return "\n\n".join(texts)

    # ------------------------------------------------------------------ one HTTP attempt
    def _attempt(self, body: dict[str, Any], req: CallRequest, n_images: int, text_chars: int,
                 label: str) -> dict[str, Any]:
        """Reserve, send, reconcile. Returns an attempt record; raises SpendRefused / PaidCallsStopped."""
        ctx = req.context
        rid = None
        if self.guard is not None:
            worst = self.worst_case(text_chars, n_images, req.image_tokens_per_image, body["max_tokens"])
            rid = self.guard.reserve(worst, bucket=ctx.get("bucket", "sweep"), model=ctx.get("model_key"),
                                     label=label)
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"started_utc": _utc(), "max_tokens": body["max_tokens"]}
        try:
            status, data, text, headers = self.transport.post(API_BASE + "/chat/completions", body,
                                                              self._headers(), self.timeout_seconds)
        except TimeoutError:
            rec.update(kind="timeout", status=None, wall_s=round(time.perf_counter() - t0, 3))
            if rid is not None:
                self.guard.lost(rid, reason="timeout")
            return rec
        except ConnectionError as exc:
            rec.update(kind="connection", status=None, error=str(exc), wall_s=round(time.perf_counter() - t0, 3))
            if rid is not None:
                self.guard.lost(rid, reason=f"connection error {exc}")
            return rec
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        rec["status"] = status
        gen_id = data.get("id") if isinstance(data, dict) else None
        usage = data.get("usage") if isinstance(data, dict) and isinstance(data.get("usage"), dict) else {}
        cost = usage.get("cost") if isinstance(usage, dict) else None
        rec["generation_id"] = gen_id
        if status == 200 and isinstance(data, dict) and not data.get("error"):
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):  # some providers return parts
                content = "".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
            rec.update(kind="ok", content=content or "", finish_reason=choice.get("finish_reason"),
                       native_finish_reason=choice.get("native_finish_reason"), usage=usage,
                       model_version=data.get("model"), provider_name=data.get("provider"))
            if not (content or "").strip() or choice.get("finish_reason") == "error":
                rec["kind"] = "empty"
        else:
            rec.update(kind="http_error", error=_error_message(data, text), usage=usage,
                       limit_source=_limit_source(data), retry_after=headers.get("retry-after"))
        if rid is not None:
            if cost is not None:
                self.guard.reconcile(rid, float(cost))
            elif rec["kind"] == "http_error" and status in (400, 401, 402, 403, 404, 429) and not gen_id:
                # Rejected by OpenRouter before any provider generated (validation, routing, credits, rate
                # limit): recorded as 0. The drift check against the key's usage catches it if that is wrong.
                self.guard.reconcile(rid, 0.0, note=f"HTTP {status} before generation, no usage; recorded as 0")
            else:
                self.guard.lost(rid, reason=f"no usage.cost ({rec['kind']}, HTTP {status})", generation_id=gen_id)
        rec["usd"] = float(cost) if cost is not None else None
        return rec

    # ------------------------------------------------------------------ the structured call
    def call(self, req: CallRequest) -> CallResult:
        from ..spend_guard import PaidCallsStopped, SpendRefused

        prompt = self._prompt_text(req)
        media = [_sha(p.jpeg) if isinstance(p, ImagePart) else _sha(p.data) for p in req.parts
                 if isinstance(p, (ImagePart, VideoPart))]
        n_images = sum(1 for p in req.parts if isinstance(p, ImagePart))
        # a video counts as one image-equivalent per second of clip (worst-case input estimate only)
        n_images += sum(max(1, int(round(p.seconds))) for p in req.parts if isinstance(p, VideoPart))
        text_chars = len(prompt) + len(json.dumps(req.schema))
        t_start = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        start = MODES.index(req.start_mode) if req.start_mode in MODES else 0
        last_error = None
        for mode in MODES[start:]:
            gen_config = {"max_tokens": req.max_tokens, "reasoning": req.reasoning, "mode": mode,
                          "schema_sha256": schema_sha256(req.schema), "temperature": None}
            key = self._cache_key(prompt, gen_config, media)
            if self.cache is not None:
                hit = self.cache.get(key)
                if hit is not None:
                    return self._from_cache(hit, req, key)
            try:
                result = self._run_mode(req, mode, key, prompt, media, n_images, text_chars, attempts)
            except _Unavailable as exc:
                last_error = str(exc)
                continue
            except SpendRefused as exc:
                return self._finish(req, CallResult(False, None, "", "refused", mode=mode, error=str(exc)),
                                    key, prompt, media, attempts, t_start)
            except PaidCallsStopped as exc:
                return self._finish(req, CallResult(False, None, "", "stopped", mode=mode, error=str(exc)),
                                    key, prompt, media, attempts, t_start)
            if result is None:  # the mode was rejected: try the next one
                continue
            return self._finish(req, result, key, prompt, media, attempts, t_start)
        return self._finish(req, CallResult(False, None, "", "unavailable", error=last_error or "no mode worked"),
                            None, prompt, media, attempts, t_start)

    def _cache_key(self, prompt: str, gen_config: dict[str, Any], media: list[str]) -> str:
        from ..eval.receipts import cache_key

        return cache_key(self.name, self.model, prompt, gen_config, media)

    def _run_mode(self, req: CallRequest, mode: str, key: str, prompt: str, media: list[str], n_images: int,
                  text_chars: int, attempts: list[dict[str, Any]]) -> CallResult | None:
        max_tokens = min(int(req.max_tokens), MAX_TOKENS_CAP)
        extra: list[dict[str, Any]] = []
        retried_length = retried_empty = repaired = timeouts = 0
        retries_5xx = 0
        while True:
            body = self._payload(req, mode, max_tokens, extra)
            label = f"{req.context.get('arm', '')} {req.context.get('episode_key', '')} {req.step} {mode}"
            rec = self._attempt(body, req, n_images, text_chars, label)
            rec["mode"] = mode
            attempts.append(rec)
            kind = rec.get("kind")
            status = rec.get("status")
            if kind == "timeout" or kind == "connection":
                timeouts += 1
                if timeouts <= 1:
                    continue
                return CallResult(False, None, "", "failed", mode=mode, error=f"{kind} twice")
            if kind == "http_error":
                msg = (rec.get("error") or "").lower()
                if status == 402:
                    src = rec.get("limit_source")
                    if src == "openrouter_in_flight_budget":
                        self.in_flight_backoff.set()
                        wait = float(rec.get("retry_after") or 10.0)
                        self._sleep(min(max(wait, 1.0), 120.0))
                        continue
                    if src == "openrouter_credits":
                        from ..spend_guard import SpendRefused

                        raise SpendRefused("HTTP 402 openrouter_credits: this request's estimate exceeds the "
                                           "in-flight budget")
                    if self.guard is not None:
                        self.guard.stop(f"HTTP 402 ({src or 'insufficient credits or key limit'})")
                    from ..spend_guard import PaidCallsStopped

                    raise PaidCallsStopped(f"HTTP 402: {rec.get('error')}")
                if status == 400 and ("image" in msg and ("size" in msg or "many" in msg or "count" in msg
                                                          or "limit" in msg or "exceed" in msg)):
                    return CallResult(False, None, "", "failed", mode=mode, error=f"model image limit: {msg[:200]}")
                if status == 400 and ("schema" in msg or "response_format" in msg or "json" in msg
                                      or "structured" in msg):
                    return None
                if status in (404, 503) and ("no endpoints" in msg or "routing requirements" in msg
                                             or "no available" in msg):
                    if mode == MODES[-1]:
                        raise _Unavailable(f"HTTP {status}: {msg[:200]}")
                    return None
                if status in RETRY_STATUSES or (status is not None and 500 <= status < 600):
                    retries_5xx += 1
                    if retries_5xx <= 3:
                        self._sleep(min(20.0, 1.5 * (2 ** (retries_5xx - 1))))
                        continue
                return CallResult(False, None, "", "failed", mode=mode, error=f"HTTP {status}: {msg[:300]}")
            if kind == "empty" and rec.get("finish_reason") == "length" and retried_length == 0                     and max_tokens < MAX_TOKENS_CAP:
                # reasoning used the whole budget before any answer: the length rule, not the empty rule
                retried_length = 1
                max_tokens = min(max_tokens * 2, MAX_TOKENS_CAP)
                continue
            if kind == "empty":
                retried_empty += 1
                if retried_empty <= 1:
                    continue
                return CallResult(False, None, "", "failed", mode=mode, error="empty content or finish_reason error")
            # kind ok
            if rec.get("finish_reason") == "length" and retried_length == 0 and max_tokens < MAX_TOKENS_CAP:
                retried_length = 1
                max_tokens = min(max_tokens * 2, MAX_TOKENS_CAP)
                continue
            text = rec.get("content") or ""
            data, errors = self._parse_and_validate(text, req)
            if not errors:
                return CallResult(True, data, text, "ok", mode=mode, finish_reason=rec.get("finish_reason"),
                                  repaired=repaired > 0, truncated_retry=retried_length > 0)
            if repaired == 0:
                repaired = 1
                extra = [{"role": "assistant", "content": text[:20000]},
                         {"role": "user", "content": "That answer did not validate: " + "; ".join(errors)[:1500]
                          + ". Return the corrected JSON only, matching the schema exactly."}]
                continue
            return CallResult(False, data, text, "invalid", mode=mode, finish_reason=rec.get("finish_reason"),
                              repaired=True, error="; ".join(errors)[:600])

    @staticmethod
    def _parse_and_validate(text: str, req: CallRequest) -> tuple[Any, list[str]]:
        from .base import extract_json

        try:
            data = extract_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            return None, [f"not JSON: {type(exc).__name__}"]
        errors: list[str] = []
        try:
            import jsonschema

            v = jsonschema.Draft202012Validator(req.schema)
            for err in sorted(v.iter_errors(data), key=lambda e: list(e.absolute_path)):
                path = "/".join(str(p) for p in err.absolute_path)
                errors.append(f"{path or '(root)'}: {err.message[:160]}")
                if len(errors) >= 8:
                    break
        except ImportError:
            pass
        if not errors and req.validate is not None:
            errors = list(req.validate(data))
        return data, errors

    # ------------------------------------------------------------------ results, cache, receipts
    def _finish(self, req: CallRequest, result: CallResult, key: str | None, prompt: str, media: list[str],
                attempts: list[dict[str, Any]], t_start: float) -> CallResult:
        usage_tot: dict[str, float] = {}
        usd = 0.0
        lat = 0.0
        gen_ids = []
        for a in attempts:
            u = a.get("usage") or {}
            for k_src, k_dst in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                if isinstance(u.get(k_src), (int, float)):
                    usage_tot[k_dst] = usage_tot.get(k_dst, 0) + u[k_src]
            det = u.get("completion_tokens_details") or {}
            if isinstance(det.get("reasoning_tokens"), (int, float)):
                usage_tot["reasoning_tokens"] = usage_tot.get("reasoning_tokens", 0) + det["reasoning_tokens"]
            pdet = u.get("prompt_tokens_details") or {}
            if isinstance(pdet.get("cached_tokens"), (int, float)):
                usage_tot["cached_tokens"] = usage_tot.get("cached_tokens", 0) + pdet["cached_tokens"]
            if a.get("usd") is not None:
                usd += float(a["usd"])
            lat += float(a.get("wall_s") or 0.0)
            if a.get("generation_id"):
                gen_ids.append(a["generation_id"])
        result.usage = usage_tot
        result.usd = round(usd, 8)
        result.latency_s = round(lat, 3)
        result.wall_s = round(time.perf_counter() - t_start, 3)
        result.attempts = len(attempts)
        result.retries = max(0, len(attempts) - 1)
        result.generation_ids = gen_ids
        last_ok = next((a for a in reversed(attempts) if a.get("kind") == "ok"), {})
        ctx = req.context
        receipt = {
            "provider": self.name, "model": self.model, "model_version": last_ok.get("model_version"),
            "request_id": gen_ids[-1] if gen_ids else None, "utc_time": _utc(), "cache_key": key,
            "prompt_sha256": _sha(prompt.encode("utf-8")),
            "inputs": {"episode_key": ctx.get("episode_key"), "camera": ctx.get("cameras"),
                       "frame_indices": ctx.get("frame_indices"), "media_resolution": ctx.get("media_resolution"),
                       "media_sha256": media},
            "generation_config": {"max_tokens": req.max_tokens, "reasoning": req.reasoning, "mode": result.mode,
                                  "response_schema_sha256": schema_sha256(req.schema), "temperature": None},
            "usage": {"input_text_tokens": None, "input_image_tokens": None, "input_video_tokens": 0,
                      "input_audio_tokens": 0, "input_tokens": usage_tot.get("input_tokens"),
                      "cached_tokens": usage_tot.get("cached_tokens"), "output_tokens": usage_tot.get("output_tokens"),
                      "reasoning_tokens": usage_tot.get("reasoning_tokens")},
            "latency_s": result.latency_s, "wall_s": result.wall_s, "retries": result.retries,
            "batch_job_id": None, "price_table_version": self.price_table_version, "usd": result.usd,
            "cache_hit": False, "status": result.status, "structured_mode": result.mode,
            "finish_reason": result.finish_reason, "provider_name": last_ok.get("provider_name"),
            "repaired": result.repaired, "truncated_retry": result.truncated_retry, "error": result.error,
            "step": req.step, "arm": ctx.get("arm"), "bucket": ctx.get("bucket"), "model_key": ctx.get("model_key"),
            "attempts": [{k: v for k, v in a.items() if k not in ("content",)} for a in attempts],
            "generation_ids": gen_ids, "n_images": len(media),
            "response_text": result.text[:60000] if result.text else None,
        }
        result.receipt = receipt
        if self.receipts is not None:
            self.receipts.write(receipt)
        if self.cache is not None and key is not None and result.status == "ok":
            self.cache.put(key, {"text": result.text, "mode": result.mode, "finish_reason": result.finish_reason,
                                 "usage": usage_tot, "usd": result.usd, "latency_s": result.latency_s,
                                 "wall_s": result.wall_s, "attempts": result.attempts, "repaired": result.repaired,
                                 "truncated_retry": result.truncated_retry, "generation_ids": gen_ids,
                                 "model": self.model, "provider_name": receipt["provider_name"],
                                 "model_version": receipt["model_version"], "utc_time": receipt["utc_time"]})
        return result

    def _from_cache(self, hit: dict[str, Any], req: CallRequest, key: str) -> CallResult:
        data, errors = self._parse_and_validate(hit.get("text") or "", req)
        result = CallResult(not errors, data, hit.get("text") or "", "ok" if not errors else "invalid",
                            mode=hit.get("mode"), finish_reason=hit.get("finish_reason"),
                            usage=hit.get("usage") or {}, usd=float(hit.get("usd") or 0.0),
                            latency_s=float(hit.get("latency_s") or 0.0), wall_s=float(hit.get("wall_s") or 0.0),
                            attempts=int(hit.get("attempts") or 1), repaired=bool(hit.get("repaired")),
                            truncated_retry=bool(hit.get("truncated_retry")), cache_hit=True,
                            generation_ids=list(hit.get("generation_ids") or []),
                            error="; ".join(errors) if errors else None)
        ctx = req.context
        receipt = {"provider": self.name, "model": self.model, "model_version": hit.get("model_version"),
                   "request_id": (result.generation_ids or [None])[-1], "utc_time": _utc(), "cache_key": key,
                   "prompt_sha256": _sha(self._prompt_text(req).encode("utf-8")),
                   "inputs": {"episode_key": ctx.get("episode_key"), "camera": ctx.get("cameras"),
                              "frame_indices": ctx.get("frame_indices"),
                              "media_sha256": [_sha(p.jpeg) for p in req.parts if isinstance(p, ImagePart)]},
                   "generation_config": {"max_tokens": req.max_tokens, "reasoning": req.reasoning,
                                         "mode": result.mode, "response_schema_sha256": schema_sha256(req.schema)},
                   "usage": result.usage, "latency_s": result.latency_s, "wall_s": result.wall_s,
                   "retries": max(0, result.attempts - 1), "batch_job_id": None,
                   "price_table_version": self.price_table_version, "usd": result.usd, "cache_hit": True,
                   "status": result.status, "structured_mode": result.mode, "finish_reason": result.finish_reason,
                   "provider_name": hit.get("provider_name"), "step": req.step, "arm": ctx.get("arm"),
                   "bucket": ctx.get("bucket"), "model_key": ctx.get("model_key"),
                   "generation_ids": result.generation_ids, "original_utc": hit.get("utc_time"),
                   "response_text": result.text[:60000]}
        result.receipt = receipt
        if self.receipts is not None:
            self.receipts.write(receipt)
        return result

    # ------------------------------------------------------------------ free endpoints
    def generation_stats(self, gen_id: str, tries: int = 5, wait_s: float = 2.0) -> dict[str, Any] | None:
        """GET /generation?id= (free; it can lag). Returns the data dict or None, never guesses."""
        for i in range(tries):
            try:
                status, data, _text, _h = self.transport.get(f"{API_BASE}/generation?id={gen_id}", self._headers(), 30)
            except (TimeoutError, ConnectionError):
                status, data = None, None
            if status == 200 and isinstance(data, dict):
                return data.get("data", data)
            if i < tries - 1:
                self._sleep(wait_s)
        return None

    def key_usage(self) -> float | None:
        """This key's lifetime usage from GET /key (works with a normal key)."""
        try:
            status, data, _text, _h = self.transport.get(API_BASE + "/key", self._headers(), 30)
        except (TimeoutError, ConnectionError):
            return None
        if status == 200 and isinstance(data, dict):
            d = data.get("data", data)
            if isinstance(d, dict) and isinstance(d.get("usage"), (int, float)):
                return float(d["usage"])
        return None

    def credits(self) -> dict[str, float] | None:
        try:
            status, data, _text, _h = self.transport.get(API_BASE + "/credits", self._headers(), 30)
        except (TimeoutError, ConnectionError):
            return None
        if status == 200 and isinstance(data, dict):
            d = data.get("data", data)
            try:
                return {"total_credits": float(d["total_credits"]), "total_usage": float(d["total_usage"])}
            except (KeyError, TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------ legacy contract (contact sheet)
    def ask(self, frames: list[np.ndarray], frame_labels: list[int], question: str, receipt_path: Path, *,
            frame_captions: list[str] | None = None, temperature: float | None = None) -> ProviderResponse:
        import io

        sheet = make_contact_sheet(frames, frame_labels, captions=frame_captions)
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=88)
        req = CallRequest(step=Path(receipt_path).stem, system="", parts=[TextPart(question), ImagePart(buf.getvalue(),
                          "contact sheet")], schema={}, schema_name="legacy", max_tokens=self.legacy_max_tokens,
                          reasoning=self.legacy_reasoning, context=dict(self.context),
                          start_mode="plain_with_schema_in_prompt")
        req.context["frame_indices"] = [int(x) for x in frame_labels]
        result = self._legacy_call(req)
        raw = {"provider": self.name, "model": self.model, "question": question, "frame_labels": list(frame_labels),
               "request_image_note": "image bytes omitted", "status": result.status, "usd": result.usd,
               "usage": result.usage, "cache_hit": result.cache_hit, "structured_mode": result.mode,
               "error": result.error, "response_text": result.text, "temperature_ignored": temperature}
        write_receipt(receipt_path, raw)
        if result.status not in ("ok", "invalid"):
            raise RuntimeError(f"OpenRouter call {result.status}: {result.error}")
        return ProviderResponse(result.text, raw, self.name, self.model, result.wall_s, result.usd)

    def _legacy_call(self, req: CallRequest) -> CallResult:
        """Plain-text call for the legacy prompts: no schema and no system message; JSON parsed by the labeler."""
        from ..spend_guard import PaidCallsStopped, SpendRefused

        prompt = req.parts[0].text
        media = [_sha(p.jpeg) for p in req.parts if isinstance(p, ImagePart)]
        gen_config = {"max_tokens": req.max_tokens, "reasoning": req.reasoning, "mode": "legacy_plain",
                      "temperature": None}
        key = self._cache_key(prompt, gen_config, media)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return self._from_cache(hit, CallRequest(req.step, "", req.parts, {}, "legacy", req.max_tokens,
                                                         req.reasoning, req.context,
                                                         validate=lambda d: []), key)
        attempts: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        req.validate = lambda d: []
        try:
            result = self._run_legacy(req, attempts)
        except SpendRefused as exc:
            result = CallResult(False, None, "", "refused", error=str(exc))
        except PaidCallsStopped as exc:
            result = CallResult(False, None, "", "stopped", error=str(exc))
        return self._finish(req, result, key, prompt, media, attempts, t0)

    def _run_legacy(self, req: CallRequest, attempts: list[dict[str, Any]]) -> CallResult:
        max_tokens = req.max_tokens
        body_req = CallRequest(req.step, req.system, req.parts, {"type": "object"}, "legacy", max_tokens,
                               req.reasoning, req.context)
        tries = 0
        while True:
            body = self._payload(body_req, "legacy_plain", max_tokens)
            body["messages"][-1]["content"] = [c for c in body["messages"][-1]["content"]
                                              if not (c.get("type") == "text"
                                                      and c.get("text", "").startswith("Return only JSON that"))]
            rec = self._attempt(body, body_req, 1, len(req.parts[0].text), f"{req.context.get('arm', '')} "
                                f"{req.context.get('episode_key', '')} {req.step} legacy")
            rec["mode"] = "legacy_plain"
            attempts.append(rec)
            tries += 1
            if rec.get("kind") == "ok":
                if rec.get("finish_reason") == "length" and max_tokens < MAX_TOKENS_CAP and tries <= 2:
                    max_tokens = min(max_tokens * 2, MAX_TOKENS_CAP)
                    continue
                return CallResult(True, None, rec.get("content") or "", "ok", mode="legacy_plain",
                                  finish_reason=rec.get("finish_reason"))
            status = rec.get("status")
            if status == 402:
                if self.guard is not None:
                    self.guard.stop(f"HTTP 402 ({rec.get('limit_source') or 'credits or key limit'})")
                from ..spend_guard import PaidCallsStopped

                raise PaidCallsStopped("HTTP 402")
            if tries <= 3 and (rec.get("kind") in ("timeout", "connection", "empty")
                               or (status is not None and (status in RETRY_STATUSES or 500 <= status < 600))):
                if rec.get("kind") == "timeout" and tries > 1:
                    return CallResult(False, None, "", "failed", mode="legacy_plain", error="timeout twice")
                self._sleep(min(20.0, 1.5 * (2 ** (tries - 1))))
                continue
            return CallResult(False, None, "", "failed", mode="legacy_plain",
                              error=f"{rec.get('kind')} HTTP {status}: {str(rec.get('error'))[:300]}")


register_provider("openrouter", OpenRouterProvider)
