"""
Tests for the Rule contract.

These tests enforce the contract itself, not any concrete rule. They fail if
someone changes the base API in a way that breaks the guarantees rules can
rely on.
"""

from __future__ import annotations

import pytest

from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def valid_breakdown() -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        signal_strength=0.8,
        data_completeness=1.0,
        notes="all required fields present",
    )


@pytest.fixture
def valid_diagnosis(valid_breakdown: ConfidenceBreakdown) -> Diagnosis:
    return Diagnosis(
        rule_id="r99",
        cause="Test cause statement.",
        fix="Test fix recommendation.",
        confidence=0.84,
        confidence_breakdown=valid_breakdown,
        evidence={"decode.hbm_bandwidth_util": 0.89},
    )


@pytest.fixture
def empty_input() -> DiagnosisInput:
    return DiagnosisInput(
        model_name="test-model",
        gpu_type="H100-SXM",
        source_files=["test"],
    )


# --------------------------------------------------------------------------- #
# ConfidenceBreakdown
# --------------------------------------------------------------------------- #

class TestConfidenceBreakdown:
    def test_valid_construction(self) -> None:
        b = ConfidenceBreakdown(signal_strength=0.7, data_completeness=0.9)
        assert b.signal_strength == 0.7
        assert b.data_completeness == 0.9
        assert b.notes == ""

    def test_notes_optional(self) -> None:
        b = ConfidenceBreakdown(signal_strength=0.5, data_completeness=0.5)
        assert b.notes == ""

    def test_signal_strength_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="signal_strength"):
            ConfidenceBreakdown(signal_strength=-0.1, data_completeness=1.0)

    def test_signal_strength_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="signal_strength"):
            ConfidenceBreakdown(signal_strength=1.1, data_completeness=1.0)

    def test_data_completeness_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="data_completeness"):
            ConfidenceBreakdown(signal_strength=0.5, data_completeness=1.5)

    def test_is_frozen(self, valid_breakdown: ConfidenceBreakdown) -> None:
        with pytest.raises(Exception):  # FrozenInstanceError
            valid_breakdown.signal_strength = 0.5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #

class TestDiagnosis:
    def test_valid_construction(self, valid_diagnosis: Diagnosis) -> None:
        assert valid_diagnosis.rule_id == "r99"
        assert valid_diagnosis.confidence == 0.84
        assert valid_diagnosis.evidence == {"decode.hbm_bandwidth_util": 0.89}

    def test_evidence_defaults_to_empty_dict(
        self, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        d = Diagnosis(
            rule_id="r99",
            cause="x",
            fix="y",
            confidence=0.7,
            confidence_breakdown=valid_breakdown,
        )
        assert d.evidence == {}

    def test_confidence_below_floor_rejected(
        self, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        # Rules below 0.5 confidence must abstain; they may not return a
        # Diagnosis. This is the trust-collapse insurance.
        with pytest.raises(ValueError, match="confidence"):
            Diagnosis(
                rule_id="r99",
                cause="x",
                fix="y",
                confidence=0.49,
                confidence_breakdown=valid_breakdown,
            )

    def test_confidence_above_one_rejected(
        self, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Diagnosis(
                rule_id="r99",
                cause="x",
                fix="y",
                confidence=1.1,
                confidence_breakdown=valid_breakdown,
            )

    def test_empty_rule_id_rejected(
        self, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        with pytest.raises(ValueError, match="rule_id"):
            Diagnosis(
                rule_id="",
                cause="x",
                fix="y",
                confidence=0.8,
                confidence_breakdown=valid_breakdown,
            )

    def test_empty_cause_rejected(
        self, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        with pytest.raises(ValueError, match="cause"):
            Diagnosis(
                rule_id="r99",
                cause="   ",
                fix="y",
                confidence=0.8,
                confidence_breakdown=valid_breakdown,
            )

    def test_empty_fix_rejected(
        self, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        with pytest.raises(ValueError, match="fix"):
            Diagnosis(
                rule_id="r99",
                cause="x",
                fix="",
                confidence=0.8,
                confidence_breakdown=valid_breakdown,
            )

    def test_is_frozen(self, valid_diagnosis: Diagnosis) -> None:
        with pytest.raises(Exception):  # FrozenInstanceError
            valid_diagnosis.confidence = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #

class TestAbstention:
    def test_has_insufficient_data(self) -> None:
        assert Abstention.INSUFFICIENT_DATA.value == "insufficient_data"

    def test_has_below_threshold(self) -> None:
        assert Abstention.BELOW_THRESHOLD.value == "below_threshold"

    def test_distinct_members(self) -> None:
        assert Abstention.INSUFFICIENT_DATA is not Abstention.BELOW_THRESHOLD


# --------------------------------------------------------------------------- #
# Rule subclass contract
# --------------------------------------------------------------------------- #

class TestRuleSubclassValidation:
    """A Rule subclass missing required class attributes must fail to define."""

    def test_missing_rule_id_rejected(self) -> None:
        with pytest.raises(TypeError, match="rule_id"):
            class BadRule(Rule):
                title = "Has title but no rule_id"
                references = ("paper://x",)

                def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                    return Abstention.BELOW_THRESHOLD

    def test_missing_title_rejected(self) -> None:
        with pytest.raises(TypeError, match="title"):
            class BadRule(Rule):
                rule_id = "r99"
                references = ("paper://x",)

                def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                    return Abstention.BELOW_THRESHOLD

    def test_missing_references_rejected(self) -> None:
        with pytest.raises(TypeError, match="references"):
            class BadRule(Rule):
                rule_id = "r99"
                title = "Test"

                def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                    return Abstention.BELOW_THRESHOLD

    def test_well_formed_subclass_accepted(self) -> None:
        # This should define cleanly.
        class GoodRule(Rule):
            rule_id = "r99"
            title = "Test"
            references = ("https://example.com/paper",)

            def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                return Abstention.BELOW_THRESHOLD

        instance = GoodRule()
        assert instance.rule_id == "r99"


# --------------------------------------------------------------------------- #
# Rule evaluation contract
# --------------------------------------------------------------------------- #

class TestRuleEvaluation:
    """A rule's evaluate() method must return one of the three valid results."""

    def test_can_return_diagnosis(
        self, empty_input: DiagnosisInput, valid_breakdown: ConfidenceBreakdown
    ) -> None:
        class FiringRule(Rule):
            rule_id = "r99"
            title = "Always fires"
            references = ("https://example.com",)

            def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                return Diagnosis(
                    rule_id=self.rule_id,
                    cause="testing",
                    fix="testing",
                    confidence=0.9,
                    confidence_breakdown=valid_breakdown,
                )

        result = FiringRule().evaluate(empty_input)
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r99"

    def test_can_abstain_insufficient_data(
        self, empty_input: DiagnosisInput
    ) -> None:
        class AbstainingRule(Rule):
            rule_id = "r99"
            title = "Cannot evaluate"
            references = ("https://example.com",)

            def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                return Abstention.INSUFFICIENT_DATA

        result = AbstainingRule().evaluate(empty_input)
        assert result is Abstention.INSUFFICIENT_DATA

    def test_can_abstain_below_threshold(
        self, empty_input: DiagnosisInput
    ) -> None:
        class QuietRule(Rule):
            rule_id = "r99"
            title = "Does not fire"
            references = ("https://example.com",)

            def evaluate(self, dx: DiagnosisInput) -> RuleResult:
                return Abstention.BELOW_THRESHOLD

        result = QuietRule().evaluate(empty_input)
        assert result is Abstention.BELOW_THRESHOLD