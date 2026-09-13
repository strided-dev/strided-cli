"""Tests for the engine runner.

These use synthetic stub rules rather than the real r01 so that engine
correctness (firing, partitioning, ranking, relations, error handling,
determinism) is decoupled from any single rule's threshold tuning.
"""

from __future__ import annotations

import pytest

import engine.relations as relations
import engine.runner as runner
from engine.runner import DiagnosisReport, run_diagnosis
from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput, Distribution, PhaseMetrics


# --------------------------------------------------------------------------- #
# Stub-rule helpers
# --------------------------------------------------------------------------- #

def _diag(rule_id: str, confidence: float) -> Diagnosis:
    return Diagnosis(
        rule_id=rule_id,
        cause=f"cause {rule_id}",
        fix=f"fix {rule_id}",
        confidence=confidence,
        confidence_breakdown=ConfidenceBreakdown(signal_strength=0.5, data_completeness=1.0),
    )


def _make_rule(rule_id: str, result: RuleResult | None = None, *, raises: Exception | None = None,
               returns_garbage: bool = False) -> type[Rule]:
    """Build a concrete Rule subclass with a fixed evaluate() behaviour.

    Built via ``type()`` so ``evaluate`` is present at class creation: an
    abstract-then-patched class would stay un-instantiable under ABCMeta.
    """

    def evaluate(self, dx):  # type: ignore[no-untyped-def]
        if raises is not None:
            raise raises
        if returns_garbage:
            return 42  # not a RuleResult
        return result

    return type(
        f"Stub_{rule_id}",
        (Rule,),
        {
            "rule_id": rule_id,
            "title": f"Stub {rule_id}",
            "references": ("test://ref",),
            "evaluate": evaluate,
        },
    )


@pytest.fixture
def dx() -> DiagnosisInput:
    return DiagnosisInput(model_name="m", gpu_type="H100-SXM", decode=PhaseMetrics())


@pytest.fixture
def use_rules(monkeypatch):
    """Replace the live registry with a given tuple of stub rule classes."""

    def _apply(*rule_classes: type[Rule]) -> None:
        monkeypatch.setattr(runner, "ALL_RULES", tuple(rule_classes))

    return _apply


@pytest.fixture
def use_relations(monkeypatch):
    """Patch the relation tables the runner consults."""

    def _apply(*, conflict=(), corroboration=()) -> None:
        monkeypatch.setattr(relations, "CONFLICT_SETS", tuple(frozenset(g) for g in conflict))
        monkeypatch.setattr(relations, "CORROBORATION_SETS", tuple(frozenset(g) for g in corroboration))

    return _apply


# --------------------------------------------------------------------------- #
# Firing & partitioning
# --------------------------------------------------------------------------- #

class TestFiringAndPartitioning:
    def test_single_fire_ranks_first(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", _diag("r01", 0.80)))
        report = run_diagnosis(dx)

        assert isinstance(report, DiagnosisReport)
        assert len(report.diagnoses) == 1
        ranked = report.diagnoses[0]
        assert ranked.rank == 1
        assert ranked.diagnosis.rule_id == "r01"
        assert ranked.adjusted_confidence == 0.80
        assert ranked.adjustment_reason == "no adjustment"
        assert report.rules_run == 1

    def test_below_threshold_is_silent(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", Abstention.BELOW_THRESHOLD))
        report = run_diagnosis(dx)
        assert report.diagnoses == ()
        assert report.insufficient_data == ()
        assert report.errors == ()

    def test_insufficient_data_is_surfaced(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", Abstention.INSUFFICIENT_DATA))
        report = run_diagnosis(dx)
        assert report.diagnoses == ()
        assert len(report.insufficient_data) == 1
        note = report.insufficient_data[0]
        assert note.rule_id == "r01"
        assert note.title == "Stub r01"

    def test_insufficient_data_payload_is_carried(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", InsufficientData(missing=("decode.sm_occupancy",))))
        report = run_diagnosis(dx)
        note = report.insufficient_data[0]
        assert note.rule_id == "r01"
        assert note.missing == ("decode.sm_occupancy",)
        assert note.reason == ""

    def test_legacy_insufficient_data_has_empty_payload(self, dx, use_rules) -> None:
        # The payload-free enum still works; its note simply carries no detail.
        use_rules(_make_rule("r01", Abstention.INSUFFICIENT_DATA))
        report = run_diagnosis(dx)
        note = report.insufficient_data[0]
        assert note.missing == ()
        assert note.reason == ""


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

class TestErrorHandling:
    def test_raising_rule_is_isolated(self, dx, use_rules) -> None:
        use_rules(
            _make_rule("r01", raises=ValueError("secret metric 0.123")),
            _make_rule("r02", _diag("r02", 0.75)),
        )
        report = run_diagnosis(dx)

        assert len(report.errors) == 1
        assert report.errors[0].rule_id == "r01"
        # Only the exception class name leaks, never the message.
        assert report.errors[0].error == "ValueError"
        assert "secret" not in report.errors[0].error
        # The healthy rule still produced its diagnosis.
        assert [d.diagnosis.rule_id for d in report.diagnoses] == ["r02"]

    def test_strict_reraises(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", raises=ValueError("boom")))
        with pytest.raises(ValueError, match="boom"):
            run_diagnosis(dx, strict=True)

    def test_contract_violation_recorded(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", returns_garbage=True))
        report = run_diagnosis(dx)
        assert report.diagnoses == ()
        assert report.errors[0].error == "ContractViolation"

    def test_contract_violation_strict_raises(self, dx, use_rules) -> None:
        use_rules(_make_rule("r01", returns_garbage=True))
        with pytest.raises(TypeError, match="not a RuleResult"):
            run_diagnosis(dx, strict=True)


# --------------------------------------------------------------------------- #
# Ranking & determinism
# --------------------------------------------------------------------------- #

class TestRanking:
    def test_sorted_by_confidence_desc(self, dx, use_rules) -> None:
        use_rules(
            _make_rule("r01", _diag("r01", 0.60)),
            _make_rule("r02", _diag("r02", 0.90)),
            _make_rule("r03", _diag("r03", 0.75)),
        )
        report = run_diagnosis(dx)
        order = [d.diagnosis.rule_id for d in report.diagnoses]
        assert order == ["r02", "r03", "r01"]
        assert [d.rank for d in report.diagnoses] == [1, 2, 3]

    def test_confidence_tie_breaks_on_rule_id(self, dx, use_rules) -> None:
        use_rules(
            _make_rule("r05", _diag("r05", 0.80)),
            _make_rule("r02", _diag("r02", 0.80)),
        )
        report = run_diagnosis(dx)
        assert [d.diagnosis.rule_id for d in report.diagnoses] == ["r02", "r05"]

    def test_deterministic_under_registry_reorder(self, dx, use_rules) -> None:
        rules_forward = (
            _make_rule("r01", _diag("r01", 0.60)),
            _make_rule("r02", _diag("r02", 0.90)),
            _make_rule("r03", _diag("r03", 0.75)),
        )
        use_rules(*rules_forward)
        forward = [d.diagnosis.rule_id for d in run_diagnosis(dx).diagnoses]

        use_rules(*reversed(rules_forward))
        reverse = [d.diagnosis.rule_id for d in run_diagnosis(dx).diagnoses]

        assert forward == reverse == ["r02", "r03", "r01"]


# --------------------------------------------------------------------------- #
# Relations: corroboration & conflict
# --------------------------------------------------------------------------- #

class TestCorroboration:
    def test_annotated_without_boost(self, dx, use_rules, use_relations) -> None:
        use_rules(
            _make_rule("r01", _diag("r01", 0.70)),
            _make_rule("r03", _diag("r03", 0.65)),
        )
        use_relations(corroboration=[{"r01", "r03"}])
        report = run_diagnosis(dx)  # boost off by default

        first = next(d for d in report.diagnoses if d.diagnosis.rule_id == "r01")
        assert first.corroborated_by == ("r03",)
        assert first.adjusted_confidence == 0.70  # unchanged
        assert "corroborated by r03" in first.adjustment_reason

    def test_boost_changes_scalar_when_enabled(self, dx, use_rules, use_relations) -> None:
        use_rules(
            _make_rule("r01", _diag("r01", 0.70)),
            _make_rule("r03", _diag("r03", 0.65)),
        )
        use_relations(corroboration=[{"r01", "r03"}])
        report = run_diagnosis(dx, enable_corroboration_boost=True)

        first = next(d for d in report.diagnoses if d.diagnosis.rule_id == "r01")
        assert first.adjusted_confidence == pytest.approx(0.75)


class TestConflict:
    def test_annotated_but_not_removed_by_default(self, dx, use_rules, use_relations) -> None:
        use_rules(
            _make_rule("r01", _diag("r01", 0.90)),
            _make_rule("r02", _diag("r02", 0.60)),
        )
        use_relations(conflict=[{"r01", "r02"}])
        report = run_diagnosis(dx)  # suppression off by default

        assert {d.diagnosis.rule_id for d in report.diagnoses} == {"r01", "r02"}
        assert report.suppressed == ()
        loser = next(d for d in report.diagnoses if d.diagnosis.rule_id == "r02")
        assert loser.conflicts_with == ("r01",)

    def test_suppression_removes_loser_when_enabled(self, dx, use_rules, use_relations) -> None:
        use_rules(
            _make_rule("r01", _diag("r01", 0.90)),
            _make_rule("r02", _diag("r02", 0.60)),
        )
        use_relations(conflict=[{"r01", "r02"}])
        report = run_diagnosis(dx, enable_conflict_suppression=True)

        assert [d.diagnosis.rule_id for d in report.diagnoses] == ["r01"]
        assert len(report.suppressed) == 1
        assert report.suppressed[0].diagnosis.rule_id == "r02"
        assert report.suppressed[0].suppressed_by == "r01"


# --------------------------------------------------------------------------- #
# Input hardening
# --------------------------------------------------------------------------- #

class TestNonFiniteGuard:
    # Schema 1.5.0 rejects non-finite floats at construction, so the validated
    # path can no longer produce these inputs (tests/test_schema_nonfinite.py
    # pins that). The scan remains defense-in-depth for inputs that BYPASS
    # validation — model_construct, or state deserialized from pre-1.5.0 — which
    # is what these tests now simulate.
    def test_non_finite_surfaced_as_warning(self, use_rules) -> None:
        use_rules()  # no rules; isolate the guard
        bad = DiagnosisInput(
            model_name="m",
            gpu_type="H100-SXM",
            prefill=PhaseMetrics.model_construct(achieved_flops=float("inf")),
        )
        report = run_diagnosis(bad)
        assert any("prefill.achieved_flops" in w for w in report.warnings)

    def test_non_finite_raises_in_strict(self, use_rules) -> None:
        use_rules()
        bad = DiagnosisInput(
            model_name="m",
            gpu_type="H100-SXM",
            ttft_ms=Distribution.model_construct(mean=float("nan")),
        )
        with pytest.raises(ValueError, match="Non-finite"):
            run_diagnosis(bad, strict=True)

    def test_clean_input_has_no_warnings(self, dx, use_rules) -> None:
        use_rules()
        assert run_diagnosis(dx).warnings == ()
