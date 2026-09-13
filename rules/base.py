"""
Abstract Rule contract for strided.

Every diagnostic rule lives in `rules/r{NN}_{name}.py` and inherits from `Rule`.
The contract here is load-bearing: changing it requires maintainer review.

Thesis: strided translates kernel-level inefficiencies into LLM-model-level
diagnoses with transparent, inspectable confidence. Three implications shape
this file:

1. Diagnoses carry both kernel-level evidence and model-level cause/fix.
2. Confidence is decomposed into the signals that produced it, not opaque.
3. Rules can abstain in two distinct ways:
   - INSUFFICIENT_DATA: required schema fields are missing
   - BELOW_THRESHOLD: data is present, signal does not fire

The engine treats these differently. INSUFFICIENT_DATA is a warning; the user
should know strided couldn't evaluate. BELOW_THRESHOLD is silent; the rule
simply did not apply.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Abstention vocabulary
# --------------------------------------------------------------------------- #

class Abstention(Enum):
    """Why a rule chose not to produce a diagnosis."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Required schema fields were missing or None. Surface to user."""

    BELOW_THRESHOLD = "below_threshold"
    """Data present, signal did not exceed the rule's threshold. Stay silent."""


# --------------------------------------------------------------------------- #
# Confidence model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfidenceBreakdown:
    """
    The inputs that produced a confidence score.

    A scalar 0-1 alone tells the user nothing about *why* a rule is confident.
    We require every rule to expose:

    - `signal_strength`: how far past the firing threshold the data sits
    - `data_completeness`: fraction of optional schema fields the rule had
    - `notes`: free-form per-rule context (e.g., "based on 200 samples")

    The final scalar is the rule's responsibility to compute and justify in
    its docstring. There is no enforced formula; the breakdown is documentation
    for the user, not arithmetic for the engine.
    """

    signal_strength: float  # 0.0 - 1.0
    data_completeness: float  # 0.0 - 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in [
            ("signal_strength", self.signal_strength),
            ("data_completeness", self.data_completeness),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"ConfidenceBreakdown.{name} must be in [0, 1]; got {value}"
                )


# --------------------------------------------------------------------------- #
# Diagnosis output
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Diagnosis:
    """
    A single ranked diagnosis. The unit of strided's output.

    Fields:
        rule_id: stable identifier matching the rule filename, e.g. "r03".
        cause: plain-English root cause translating kernel signals to model
            terms. One to three sentences. Past tense ("KV cache utilization
            was 91% ..."), not imperative.
        fix: plain-English suggested action with concrete commands or config
            changes where applicable. Present tense, actionable.
        confidence: scalar in [0, 1]. The rule's overall confidence in the
            diagnosis. Computed however the rule decides; justified by
            confidence_breakdown.
        confidence_breakdown: the signals that produced the scalar. Must be
            populated; rules may not return Diagnosis with a placeholder
            breakdown.
        evidence: schema field paths and their values that fired the rule.
            Keys should be dotted paths into DiagnosisInput
            (e.g., "decode.hbm_bandwidth_util"). Values are the actual numbers
            from the input.

    Rules MUST NOT return a Diagnosis with confidence < 0.5. If the rule is
    less confident than that, it should abstain via BELOW_THRESHOLD instead.
    Trust collapses on the first wrong answer; we'd rather say nothing.
    """

    rule_id: str
    cause: str
    fix: str
    confidence: float
    confidence_breakdown: ConfidenceBreakdown
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.5 <= self.confidence <= 1.0:
            raise ValueError(
                f"Diagnosis.confidence must be in [0.5, 1.0]; got {self.confidence}. "
                f"Rules below 0.5 must abstain via Abstention.BELOW_THRESHOLD."
            )
        if not self.rule_id:
            raise ValueError("Diagnosis.rule_id may not be empty")
        if not self.cause.strip() or not self.fix.strip():
            raise ValueError("Diagnosis.cause and Diagnosis.fix may not be empty")


# --------------------------------------------------------------------------- #
# Insufficient-data payload
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InsufficientData:
    """An abstention that names *why* a rule could not evaluate.

    Closes the gap where ``Abstention.INSUFFICIENT_DATA`` could say a rule did not
    run but not *which* field was missing. ``missing`` lists dotted schema paths
    the rule needed but found absent (e.g. ``"decode.sm_occupancy"``); ``reason``
    is a free-form note for non-missing causes — a sample-size floor or a
    malformed dump. At least one of the two must be set.

    The engine treats this exactly like ``Abstention.INSUFFICIENT_DATA`` — it is
    surfaced to the user, never silent — but carries the payload through to the
    report so the CLI (and live ``watch``) can say *what to provide* rather than a
    bare "couldn't run". ``Abstention.INSUFFICIENT_DATA`` remains valid (empty
    payload) for rules not yet migrated; new rules should prefer this type.
    """

    missing: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.missing and not self.reason:
            raise ValueError(
                "InsufficientData must name at least one missing field or a reason."
            )


# Type alias for what a rule may return. ``InsufficientData`` and
# ``Abstention.INSUFFICIENT_DATA`` are equivalent to the engine (both surface);
# the former additionally carries the missing-field payload.
RuleResult = Diagnosis | InsufficientData | Abstention


def is_insufficient_data(result: RuleResult) -> bool:
    """True if ``result`` is an insufficient-data abstention, in either form.

    Treats the payload-carrying ``InsufficientData`` and the legacy
    ``Abstention.INSUFFICIENT_DATA`` as equivalent, so callers that only need the
    boolean do not have to special-case both.
    """
    return isinstance(result, InsufficientData) or result is Abstention.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# The Rule contract
# --------------------------------------------------------------------------- #

class Rule(ABC):
    """
    Abstract base class for every strided diagnostic rule.

    Each concrete rule lives in its own file under rules/ and is named
    r{NN}_{snake_case_name}.py. The rule class inside should be named
    {CamelCaseName}Rule. Example:

        rules/r01_decode_memory_bound.py
        class DecodeMemoryBoundRule(Rule): ...

    The rule sees the full DiagnosisInput. It is the rule's responsibility
    to pull the fields it needs and to check that they are populated.
    """

    # Class-level metadata. Subclasses MUST override.

    rule_id: str = ""
    """Stable identifier, e.g. "r01". Matches the filename prefix."""

    title: str = ""
    """One-line human-readable name, e.g. "Decode memory-bound at low batch"."""

    references: tuple[str, ...] = ()
    """
    Literature backing this rule. URLs or short citations. At least one
    reference is required; rules cannot be folk wisdom.
    """

    # The evaluation entry point. Subclasses MUST override.

    @abstractmethod
    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        """
        Evaluate the rule against a DiagnosisInput.

        Returns:
            Diagnosis: the rule fires with confidence >= 0.5.
            InsufficientData(missing=..., reason=...): required schema fields
                missing (or data untrustworthy); names the gap. Preferred over the
                bare enum so the user learns *what* to provide.
            Abstention.INSUFFICIENT_DATA: legacy payload-free equivalent.
            Abstention.BELOW_THRESHOLD: data present, rule does not fire.

        Implementations should:
            1. Check that required fields are populated; return
               InsufficientData naming them early if not.
            2. Compute the relevant signal(s) from the input.
            3. Compare against the rule's threshold(s).
            4. If below threshold, return BELOW_THRESHOLD.
            5. Otherwise compute confidence, build the breakdown, and
               return a populated Diagnosis.

        Rules MUST NOT mutate the DiagnosisInput. Treat it as read-only.
        Rules MUST NOT perform I/O (no network, no disk, no logging beyond
        what the engine provides).
        """

    # Convenience for self-validation. Subclasses may override but rarely should.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses (intermediate base classes) skip validation.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("rule_id", "title"):
            if not getattr(cls, attr, None):
                raise TypeError(
                    f"{cls.__name__} must set class attribute `{attr}`."
                )
        if not re.fullmatch(r"r\d{2}", cls.rule_id):
            raise TypeError(
                f"{cls.__name__}.rule_id must match 'rNN' (e.g. 'r01'); got {cls.rule_id!r}."
            )
        if not cls.references:
            raise TypeError(
                f"{cls.__name__} must set class attribute `references` with "
                f"at least one citation. Rules require literature backing."
            )


__all__ = [
    "Abstention",
    "ConfidenceBreakdown",
    "Diagnosis",
    "InsufficientData",
    "Rule",
    "RuleResult",
    "is_insufficient_data",
]