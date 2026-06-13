"""robolabel: pi0.7-style conditioning annotations for LeRobot datasets.

Labels + measurement + a fixing loop. The honest claim is not "good labels";
it is that VLM labels are wrong often enough that you need to measure how wrong,
and this package gives you the measurement and the human calibration loop.
"""

__version__ = "0.1.0"

# v2 added `phase`, `boundary_evidence`, `strategy`. v3 adds a `target` subtask column
# (the grounded object/destination slot — "phase -> target"). Additive: v1/v2 files still
# read — absent columns are treated as null.
SCHEMA_VERSION = "robolabel/annotations/v3"
