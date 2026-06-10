"""robovid_conditioner: pi0.7-style conditioning annotations for LeRobot datasets.

Labels + measurement + a fixing loop. The honest claim is not "good labels";
it is that VLM labels are wrong often enough that you need to measure how wrong,
and this package gives you the measurement and the human calibration loop.
"""

__version__ = "0.1.0"

SCHEMA_VERSION = "robovid_conditioner/annotations/v1"
