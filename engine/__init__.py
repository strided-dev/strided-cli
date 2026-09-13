"""strided diagnosis engine.

The fan-in half of the pipeline: it takes one ``DiagnosisInput``, runs the
registered rules, resolves their relationships, and returns a ranked, annotated
``DiagnosisReport``. Pure and in-process — no I/O, no telemetry, no persistence.
Formatting and printing belong to the CLI, not here.
"""

from engine.confidence import ConfidenceAdjustment
from engine.runner import (
    DiagnosisReport,
    InsufficientDataNote,
    RankedDiagnosis,
    RuleError,
    SuppressedDiagnosis,
    run_diagnosis,
)

__all__ = [
    "run_diagnosis",
    "DiagnosisReport",
    "RankedDiagnosis",
    "InsufficientDataNote",
    "SuppressedDiagnosis",
    "RuleError",
    "ConfidenceAdjustment",
]
