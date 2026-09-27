"""Measurement harness for the redesign (MEASUREMENT_SPEC): metrics, gold v2, prices, receipts, the seal.

Everything here is deterministic and makes no network call. Optional dependencies (scipy, jsonschema)
come from the ``eval`` extra and are imported inside the functions that need them.
"""
