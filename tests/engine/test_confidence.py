"""Tests for the relational confidence policy.

The policy is the one place the engine produces a number of its own, so the
bounds and the default-off behaviour are pinned tightly here.
"""

from __future__ import annotations

import pytest

from engine.confidence import (
    CONFIDENCE_CEILING,
    CORROBORATION_STEP,
    ConfidenceAdjustment,
    adjust,
)
from rules.base import ConfidenceBreakdown, Diagnosis


def _diag(rule_id: str = "r01", confidence: float = 0.70) -> Diagnosis:
    return Diagnosis(
        rule_id=rule_id,
        cause="cause",
        fix="fix",
        confidence=confidence,
        confidence_breakdown=ConfidenceBreakdown(signal_strength=0.5, data_completeness=1.0),
    )


class TestNoBoostDefault:
    def test_isolated_diagnosis_is_no_op(self) -> None:
        adj = adjust(_diag(confidence=0.70), (), ())
        assert adj.adjusted_confidence == 0.70
        assert adj.delta == 0.0
        assert adj.reason == "no adjustment"

    def test_corroboration_annotates_but_does_not_change_scalar(self) -> None:
        adj = adjust(_diag(confidence=0.70), ("r03",), (), enable_boost=False)
        assert adj.adjusted_confidence == 0.70  # default: annotation only
        assert adj.corroborated_by == ("r03",)
        assert "corroborated by r03" in adj.reason
        assert "+" not in adj.reason  # no numeric change advertised

    def test_conflict_annotation(self) -> None:
        adj = adjust(_diag(), (), ("r02",))
        assert adj.conflicts_with == ("r02",)
        assert "conflicts with r02" in adj.reason
        assert adj.delta == 0.0


class TestBoostEnabled:
    def test_single_corroborator_adds_one_step(self) -> None:
        adj = adjust(_diag(confidence=0.70), ("r03",), (), enable_boost=True)
        assert adj.adjusted_confidence == pytest.approx(0.70 + CORROBORATION_STEP)
        assert adj.delta == pytest.approx(CORROBORATION_STEP)
        assert f"+{CORROBORATION_STEP:.2f}" in adj.reason

    def test_multiple_corroborators_accumulate(self) -> None:
        adj = adjust(_diag(confidence=0.70), ("r03", "r05"), (), enable_boost=True)
        assert adj.adjusted_confidence == pytest.approx(0.70 + 2 * CORROBORATION_STEP)

    def test_capped_at_ceiling(self) -> None:
        adj = adjust(_diag(confidence=0.90), ("r02", "r03", "r05", "r07"), (), enable_boost=True)
        assert adj.adjusted_confidence == CONFIDENCE_CEILING

    def test_ceiling_strictly_below_one(self) -> None:
        assert CONFIDENCE_CEILING < 1.0

    def test_boost_with_no_corroborators_is_no_op(self) -> None:
        adj = adjust(_diag(confidence=0.70), (), (), enable_boost=True)
        assert adj.adjusted_confidence == 0.70


class TestAdjustmentValueObject:
    def test_is_frozen(self) -> None:
        adj = adjust(_diag(), (), ())
        with pytest.raises(Exception):  # FrozenInstanceError
            adj.adjusted_confidence = 0.99  # type: ignore[misc]

    def test_delta_is_derived(self) -> None:
        adj = ConfidenceAdjustment(
            base_confidence=0.70,
            adjusted_confidence=0.75,
            corroborated_by=("r03",),
            conflicts_with=(),
            reason="x",
        )
        assert adj.delta == pytest.approx(0.05)
