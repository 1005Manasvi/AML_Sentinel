"""AML Sentinel: anti-money-laundering alert triage with per-transaction scoring.

The package is organised as a two-stage triage pipeline:

* System 1 (fast, every transaction): rules, graph features + LightGBM, Laya triage,
  fusion and calibration.
* System 2 (slow, escalated alerts only): graph pattern detectors, GNN experiment and
  SAR drafting.

Security note:
    The package is designed for synthetic data (IBM AMLworld). It must never be fed
    real customer data without a separate privacy and legal review.
"""

__version__ = "1.1.0"
