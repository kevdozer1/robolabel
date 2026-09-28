"""Price table for paid and local model calls (MEASUREMENT_SPEC 9.3).

``prices.yaml`` (packaged with robolabel) lists USD per million tokens per provider and model id.
:class:`PriceTable` turns token counts into dollars. A model without an entry raises
:class:`UnknownModelPrice`; a cost is never None. Arithmetic is exact decimal, converted to float
once at the end.

Token conventions, for every method:

- ``input_tokens`` is the whole prompt, cached tokens included (OpenRouter's ``prompt_tokens``).
  Image, video and audio tokens are part of it and are priced at the input price.
- ``cached_tokens`` is the part of ``input_tokens`` read from the provider's prompt cache; it is
  billed at ``cached_in`` (or at the input price when the table lists no cached price).
- ``output_tokens`` is every billed completion token. Reasoning (thinking) tokens are billed as
  output, and the caller must include them in ``output_tokens``: OpenRouter's
  ``completion_tokens`` already does, Gemini's native ``candidatesTokenCount`` does not.
- A trailing ``:batch`` on the model id, or ``batch=True``, selects the batch prices. A model with
  no cheaper batch variant has no batch prices, so a batch call to it raises; use
  :meth:`PriceTable.batch_equivalent` for projections (it falls back to sync prices).
"""

from __future__ import annotations

import hashlib
import math
import numbers
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

BATCH_SUFFIX = ":batch"
WILDCARD = "*"
UNIT = "usd_per_million_tokens"
_MILLION = Decimal(1_000_000)
_PRICE_FIELDS = ("price_in", "price_out", "cached_in", "batch_in", "batch_out")
_TEXT_FIELDS = ("effective_from", "effective_to", "source_url", "accessed", "notes")
_ALLOWED_FIELDS = frozenset(_PRICE_FIELDS + _TEXT_FIELDS)


class UnknownModelPrice(KeyError):
    """No price entry for a (provider, model), or no batch price for a batch call."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown model price"


@dataclass(frozen=True)
class ModelPrice:
    """The rates (USD per million tokens) that apply to calls of one model, plus the entry's facts.

    ``price_in``, ``price_out`` and ``cached_in`` are the applied rates: the batch rates when
    ``batch`` is True, and ``cached_in`` falls back to the applied input rate when the table lists
    no cached price. ``sync_in`` and ``sync_out`` are always the sync prices.
    """

    provider: str
    model: str  # the id asked for, including any ":batch" suffix
    entry: str  # the table key that matched: the base model id, or "*"
    batch: bool
    price_in: float
    price_out: float
    cached_in: float
    sync_in: float
    sync_out: float
    batch_in: float | None
    batch_out: float | None
    effective_from: str | None
    effective_to: str | None
    source_url: str | None
    accessed: str | None
    notes: str

    def is_effective(self, day: str | date) -> bool:
        """True when ``day`` lies inside ``[effective_from, effective_to]``, both ends inclusive.

        ``day`` is a date, a datetime (only its calendar date counts) or an ISO string whose first 10
        characters are ``YYYY-MM-DD`` (a UTC timestamp such as a receipt's ``utc_time`` works).
        """
        if isinstance(day, datetime):
            day = day.date()
        iso = day.isoformat() if isinstance(day, date) else str(day)[:10]
        if self.effective_from is not None and iso < self.effective_from:
            return False
        return self.effective_to is None or iso <= self.effective_to


@dataclass(frozen=True)
class _Entry:
    price_in: float
    price_out: float
    cached_in: float | None
    batch_in: float | None
    batch_out: float | None
    effective_from: str | None
    effective_to: str | None
    source_url: str | None
    accessed: str | None
    notes: str


def split_batch(model: str) -> tuple[str, bool]:
    """``("openai/gpt-6-sol", True)`` for ``"openai/gpt-6-sol:batch"``; the id unchanged otherwise."""
    if model.endswith(BATCH_SUFFIX):
        return model[: -len(BATCH_SUFFIX)], True
    return model, False


def _dec(x: float) -> Decimal:
    """Decimal of a price or count as written (``repr`` keeps 0.435 as 0.435, not its binary value)."""
    if isinstance(x, numbers.Integral):
        return Decimal(int(x))
    return Decimal(repr(float(x)))


def _count(value: Any, name: str) -> Decimal:
    """A token count as Decimal; raises ValueError for None, bools, negatives and non-finite values.

    Python and numpy integers and floats are accepted (counts summed with pandas are numpy ints).
    """
    kind = type(value)
    if value is None or isinstance(value, bool) or (kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_")):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    if not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    return _dec(value)


def _price(value: Any, where: str) -> float:
    if value is None:
        raise ValueError(f"{where} must be a number (USD per million tokens), not null")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{where} must be a non-negative number, got {value!r}")
    return float(value)


def _optional_price(value: Any, where: str) -> float | None:
    return None if value is None else _price(value, where)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, date) else str(value)


def _parse_entry(fields: Any, where: str) -> _Entry:
    if not isinstance(fields, dict):
        raise ValueError(f"{where} must be a mapping of price fields")
    unknown = sorted(set(fields) - _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"{where} has unknown fields {unknown}; allowed: {sorted(_ALLOWED_FIELDS)}")
    batch_in = _optional_price(fields.get("batch_in"), f"{where}.batch_in")
    batch_out = _optional_price(fields.get("batch_out"), f"{where}.batch_out")
    if (batch_in is None) != (batch_out is None):
        raise ValueError(f"{where}: batch_in and batch_out must both be set or both be null")
    return _Entry(
        price_in=_price(fields.get("price_in"), f"{where}.price_in"),
        price_out=_price(fields.get("price_out"), f"{where}.price_out"),
        cached_in=_optional_price(fields.get("cached_in"), f"{where}.cached_in"),
        batch_in=batch_in,
        batch_out=batch_out,
        effective_from=_text(fields.get("effective_from")),
        effective_to=_text(fields.get("effective_to")),
        source_url=_text(fields.get("source_url")),
        accessed=_text(fields.get("accessed")),
        notes=_text(fields.get("notes")) or "",
    )


def _parse_table(data: Any) -> tuple[str, dict[str, dict[str, _Entry]]]:
    if not isinstance(data, dict):
        raise ValueError("prices.yaml must be a mapping")
    version = _text(data.get("version"))
    if not version:
        raise ValueError("prices.yaml needs a 'version'")
    if data.get("currency", "USD") != "USD":
        raise ValueError(f"prices.yaml currency must be USD, got {data.get('currency')!r}")
    if data.get("unit", UNIT) != UNIT:
        raise ValueError(f"prices.yaml unit must be {UNIT}, got {data.get('unit')!r}")
    providers = data.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("prices.yaml needs a non-empty 'providers' mapping")
    table: dict[str, dict[str, _Entry]] = {}
    for provider, models in providers.items():
        if not isinstance(models, dict) or not models:
            raise ValueError(f"providers.{provider} must be a non-empty mapping of model ids")
        table[str(provider)] = {}
        for model, fields in models.items():
            model = str(model)
            if model.endswith(BATCH_SUFFIX):
                raise ValueError(f"providers.{provider}.{model}: put batch prices in batch_in and batch_out "
                                 "of the base model id, not in a ':batch' entry")
            table[str(provider)][model] = _parse_entry(fields, f"providers.{provider}.{model}")
    return version, table


class PriceTable:
    """Prices from one ``prices.yaml``. Build it with :meth:`load`."""

    def __init__(self, raw: bytes, source: str = "<bytes>"):
        self.source = source
        self._sha256 = hashlib.sha256(raw).hexdigest()
        self._file_version, self._table = _parse_table(yaml.safe_load(raw.decode("utf-8")))

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceTable:
        """Load ``path``, or the ``prices.yaml`` packaged with robolabel when ``path`` is None."""
        if path is None:
            raw = resources.files("robolabel").joinpath("prices.yaml").read_bytes()
            return cls(raw, source="bundled:prices.yaml")
        p = Path(path)
        return cls(p.read_bytes(), source=p.name)

    @property
    def version(self) -> str:
        """The file's ``version`` plus the first 12 hex chars of its SHA-256, e.g. ``2026-09-27+1a2b3c4d5e6f``."""
        return f"{self._file_version}+{self._sha256[:12]}"

    @property
    def sha256(self) -> str:
        """SHA-256 of the file bytes."""
        return self._sha256

    def providers(self) -> list[str]:
        return sorted(self._table)

    def known_models(self, provider: str) -> list[str]:
        """Model ids with an entry under ``provider`` (``"*"`` for a wildcard entry)."""
        return sorted(self._table.get(provider, {}))

    def lookup(self, provider: str, model: str, *, batch: bool = False) -> ModelPrice:
        """The rates for ``(provider, model)``; a ``:batch`` suffix or ``batch=True`` selects batch prices.

        Raises UnknownModelPrice for an unknown provider, a model with no entry (a provider's ``"*"``
        entry matches any id), or a batch call to a model without batch prices.
        """
        base, suffixed = split_batch(str(model))
        want_batch = suffixed or bool(batch)
        models = self._table.get(provider)
        if models is None:
            raise UnknownModelPrice(
                f"no prices for provider {provider!r} in prices.yaml ({self.version}); "
                f"known providers: {', '.join(self.providers())}")
        key = base if base in models else WILDCARD if WILDCARD in models else None
        if key is None:
            raise UnknownModelPrice(
                f"no price entry for {provider} model {base!r} in prices.yaml ({self.version}); "
                "add one before calling it")
        e = models[key]
        rate_in, rate_out = e.price_in, e.price_out
        if want_batch:
            if e.batch_in is None or e.batch_out is None:
                raise UnknownModelPrice(
                    f"{provider} model {base!r} has no cheaper batch variant in prices.yaml, so a batch call "
                    "is not priced; batch_equivalent() gives the batch-equivalent cost (= sync)")
            rate_in, rate_out = e.batch_in, e.batch_out
        return ModelPrice(
            provider=provider, model=str(model), entry=key, batch=want_batch,
            price_in=rate_in, price_out=rate_out,
            cached_in=e.cached_in if e.cached_in is not None else rate_in,
            sync_in=e.price_in, sync_out=e.price_out, batch_in=e.batch_in, batch_out=e.batch_out,
            effective_from=e.effective_from, effective_to=e.effective_to,
            source_url=e.source_url, accessed=e.accessed, notes=e.notes,
        )

    def cost(self, provider: str, model: str, input_tokens: int, output_tokens: int,
             cached_tokens: int = 0, batch: bool = False) -> float:
        """USD for one call. ``output_tokens`` must include reasoning tokens (billed as output).

        ``input_tokens`` includes ``cached_tokens``. Raises UnknownModelPrice (see :meth:`lookup`)
        and ValueError for a negative or missing count or more cached than input tokens.
        """
        return _usd(self.lookup(provider, model, batch=batch), input_tokens, output_tokens, cached_tokens)

    def batch_equivalent(self, provider: str, model: str, input_tokens: int, output_tokens: int,
                         cached_tokens: int = 0) -> float:
        """USD for the same tokens at batch prices, or at sync prices when there is no cheaper batch variant."""
        base, _ = split_batch(str(model))
        sync = self.lookup(provider, base)
        rate = sync if sync.batch_in is None else self.lookup(provider, base, batch=True)
        return _usd(rate, input_tokens, output_tokens, cached_tokens)

    def worst_case(self, provider: str, model: str, input_tokens: float, max_tokens: int) -> float:
        """Most a call can cost: every input token uncached plus ``max_tokens`` of output.

        ``input_tokens`` may be an estimate (a float is rounded up). ``max_tokens`` covers reasoning
        tokens too, since they count against it.
        """
        rate = self.lookup(provider, model)
        n_in = Decimal(math.ceil(_count(input_tokens, "input_tokens")))
        n_out = _count(max_tokens, "max_tokens")
        return float((n_in * _dec(rate.price_in) + n_out * _dec(rate.price_out)) / _MILLION)


def _usd(rate: ModelPrice, input_tokens: Any, output_tokens: Any, cached_tokens: Any) -> float:
    n_in = _count(input_tokens, "input_tokens")
    n_out = _count(output_tokens, "output_tokens")
    n_cached = _count(cached_tokens, "cached_tokens")
    if n_cached > n_in:
        raise ValueError(f"cached_tokens ({cached_tokens}) exceeds input_tokens ({input_tokens}); "
                         "input_tokens must include the cached ones")
    total = ((n_in - n_cached) * _dec(rate.price_in) + n_cached * _dec(rate.cached_in)
             + n_out * _dec(rate.price_out))
    return float(total / _MILLION)
