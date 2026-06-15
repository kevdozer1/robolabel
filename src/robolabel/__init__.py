"""robolabel: pi0.7-style conditioning annotations for LeRobot datasets.

Labels + measurement + a fixing loop. The honest claim is not "good labels";
it is that VLM labels are wrong often enough that you need to measure how wrong,
and this package gives you the measurement and the human calibration loop.
"""

__version__ = "0.1.0"

# v2 added `phase`, `boundary_evidence`, `strategy`. v3 added a `target` subtask column
# (the grounded object/destination slot — "phase -> target"). v4 adds deterministic,
# data-derived conditioning fields: `control_modality` (episode), `active_dof` (subtask),
# and a `retrieved_subgoal_*` pair (subgoal — a same-phase keyframe from a DIFFERENT episode,
# stored alongside the real same-episode keyframe, never replacing it). Additive: v1/v2/v3
# files still read — absent columns are treated as null.
SCHEMA_VERSION = "robolabel/annotations/v4"
