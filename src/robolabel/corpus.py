"""Corpus-level rescore of the population-relative fields (novelty, curation, speed tier).

The per-run modules normalize within a single small, often same-y dataset, which produces
meaningless ultra-narrow tier bands. The honest fix is to compute the population-relative fields
**pooled across all episodes/categories provided**, with global thresholds, and to assign tiers
only when the pooled population is large/heterogeneous enough (else leave the tier null and keep
the raw continuous score). This module does that pooled pass; ``scripts/make_gallery.py`` calls
it across the task datasets before building the gallery.

Raw continuous fields (``novelty``, ``curation_value``, ``speed_norm``, ``active_*``) are always
emitted by the run; this pass only *re-references* them to the pooled distribution and decides
tiers. No VLM, deterministic.
"""
from __future__ import annotations

from typing import Any

from .curation import assign_tiers, curation_values, tierable
from .novelty import novelty_scores
from .speed import bin_speeds


def rescore_corpus(pool: dict[Any, dict], *, k: int = 8, w_quality: float = 0.5,
                   w_novelty: float = 0.5, min_population: int = 15) -> dict[Any, dict]:
    """Pool {key: {emb, quality, speed_norm}} -> {key: {novelty, curation_value, curation_tier,
    speed, tiered}}. ``novelty`` is pooled kNN distance; tiers use global percentiles + the guard.
    """
    keys = list(pool)
    novelty = novelty_scores({kk: pool[kk]["emb"] for kk in keys}, k)
    quality = {kk: pool[kk].get("quality") for kk in keys}
    value = curation_values(quality, novelty, w_quality, w_novelty)
    ctier = assign_tiers(value, compress=True, min_population=min_population)
    stier = bin_speeds({kk: float(pool[kk].get("speed_norm") or 0.0) for kk in keys},
                       min_population=min_population)
    tiered = tierable(value, min_population)
    return {kk: {"novelty": novelty[kk], "curation_value": value[kk],
                 "curation_tier": ctier[kk], "speed": stier[kk], "tiered": tiered} for kk in keys}
