"""Measurement harness for the redesign (MEASUREMENT_SPEC): metrics, gold v2, prices, receipts, the seal.

Everything here is deterministic and makes no network call. Optional dependencies (scipy, jsonschema)
come from the ``eval`` extra and are imported inside the functions that need them.

:func:`derive_attempt_outcome` fills the v1.1 ``attempt_outcome`` of segments that lack it (SPEC_V1_1 4,
:mod:`robolabel.eval.failure`).
"""

from .failure import derive_attempt_outcome

__all__ = ["derive_attempt_outcome"]
