"""The engine runner: fire rules, partition results, rank, report.

Single public entry point: ``run_diagnosis(dx) -> DiagnosisReport``.

Design constraints (see ``docs/engine/ARCHITECTURE.md`` in the repo):

- Pure and in-process: no network, no disk, no telemetry, no logging of input.
- Deterministic: explicit rule order plus a total-order sort key.
- Defensive: one misbehaving rule must not void the diagnoses of the others.
- Conservative: ranks and annotates; it neither invents confidence nor deletes
  a diagnosis unless explicitly asked to via a flag.

Formatting and printing are the CLI's job. The engine returns data only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import engine.relations as relations
from engine.confidence import adjust
from engine.registry import ALL_RULES
from rules.base import Abstention, Diagnosis, InsufficientData
from schema import DiagnosisInput

ENGINE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Report value objects (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankedDiagnosis:
    """A surviving diagnosis with its rank and engine annotations.

    ``diagnosis`` is the rule's original, untouched output. ``adjusted_confidence``
    equals ``diagnosis.confidence`` unless the corroboration boost was enabled.
    """

    rank: int
    diagnosis: Diagnosis
    adjusted_confidence: float
    corroborated_by: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    adjustment_reason: str


@dataclass(frozen=True)
class InsufficientDataNote:
    """A rule that could not evaluate because required fields were absent.

    ``missing`` holds the dotted schema paths the rule needed but lacked;
    ``reason`` is a free-form note for non-missing causes (sample-size floor,
    malformed dump). Both are empty for rules still returning the legacy
    payload-free ``Abstention.INSUFFICIENT_DATA``.
    """

    rule_id: str
    title: str
    missing: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class SuppressedDiagnosis:
    """A diagnosis removed by conflict resolution (only when suppression is on)."""

    diagnosis: Diagnosis
    suppressed_by: str
    reason: str


@dataclass(frozen=True)
class RuleError:
    """A rule that raised. ``error`` holds the exception class name only.

    The message and traceback are deliberately omitted: they can embed customer
    metric values that would leak if a user pasted the report into a public
    issue. Run with ``strict=True`` to re-raise and see full detail locally.
    """

    rule_id: str
    error: str


@dataclass(frozen=True)
class DiagnosisReport:
    """The complete, ranked output of one diagnosis run."""

    diagnoses: tuple[RankedDiagnosis, ...]
    insufficient_data: tuple[InsufficientDataNote, ...]
    suppressed: tuple[SuppressedDiagnosis, ...]
    errors: tuple[RuleError, ...]
    warnings: tuple[str, ...]
    rules_run: int
    engine_version: str = ENGINE_VERSION


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_diagnosis(
    dx: DiagnosisInput,
    *,
    strict: bool = False,
    enable_corroboration_boost: bool = False,
    enable_conflict_suppression: bool = False,
) -> DiagnosisReport:
    """Run every registered rule against ``dx`` and return a ranked report.

    Args:
        dx: the canonical, already-validated input. Treated as read-only.
        strict: re-raise rule exceptions, contract violations, and non-finite
            input instead of degrading gracefully. Intended for CI and tests.
        enable_corroboration_boost: allow corroboration to raise the confidence
            scalar. OFF by default (annotation only) until validated on real data.
        enable_conflict_suppression: remove conflict losers from the ranking.
            OFF by default (annotation only) for the same reason.
    """
    warnings = tuple(_scan_non_finite(dx))
    if warnings and strict:
        raise ValueError(f"Non-finite values in input: {warnings}")

    fired, insufficient, errors = _fire_rules(dx, strict=strict)

    if enable_conflict_suppression:
        survivors, suppressed = _resolve_conflicts(fired)
    else:
        survivors, suppressed = fired, ()

    ranked = _rank(survivors, enable_boost=enable_corroboration_boost)

    return DiagnosisReport(
        diagnoses=ranked,
        insufficient_data=tuple(insufficient),
        suppressed=suppressed,
        errors=tuple(errors),
        warnings=warnings,
        rules_run=len(ALL_RULES),
    )


# ---------------------------------------------------------------------------
# Stage 1 — fire rules, partition results
# ---------------------------------------------------------------------------

def _fire_rules(
    dx: DiagnosisInput, *, strict: bool
) -> tuple[list[Diagnosis], list[InsufficientDataNote], list[RuleError]]:
    """Evaluate every rule, partitioning results by outcome.

    ``BELOW_THRESHOLD`` is dropped silently, per the rule contract. Exceptions
    and contract violations become ``RuleError``s (or re-raise under ``strict``)
    so a single bad rule cannot void the rest of the run.
    """
    fired: list[Diagnosis] = []
    insufficient: list[InsufficientDataNote] = []
    errors: list[RuleError] = []

    for rule_cls in ALL_RULES:
        try:
            result = rule_cls().evaluate(dx)
        except Exception as exc:  # noqa: BLE001 — boundary; never catch BaseException
            if strict:
                raise
            errors.append(RuleError(rule_cls.rule_id, type(exc).__name__))
            continue

        if isinstance(result, Diagnosis):
            fired.append(result)
        elif isinstance(result, InsufficientData):
            insufficient.append(
                InsufficientDataNote(
                    rule_cls.rule_id, rule_cls.title, result.missing, result.reason
                )
            )
        elif result is Abstention.INSUFFICIENT_DATA:
            # Legacy payload-free abstention (rules not yet migrated).
            insufficient.append(InsufficientDataNote(rule_cls.rule_id, rule_cls.title))
        elif result is Abstention.BELOW_THRESHOLD:
            continue  # silent, by contract
        else:
            if strict:
                raise TypeError(
                    f"{rule_cls.rule_id} returned {type(result).__name__}, "
                    f"which is not a RuleResult."
                )
            errors.append(RuleError(rule_cls.rule_id, "ContractViolation"))

    return fired, insufficient, errors


# ---------------------------------------------------------------------------
# Stage 2 — conflict resolution (opt-in)
# ---------------------------------------------------------------------------

def _resolve_conflicts(
    fired: list[Diagnosis],
) -> tuple[list[Diagnosis], tuple[SuppressedDiagnosis, ...]]:
    """Within each conflict set, keep the highest-confidence diagnosis.

    Ties break on ``rule_id`` ascending so the winner is deterministic.
    Suppression uses the rule's self-reported confidence (the engine has not
    adjusted anything yet at this stage).
    """
    by_id = {d.rule_id: d for d in fired}
    present = set(by_id)
    suppressed_by: dict[str, str] = {}

    for group in relations.CONFLICT_SETS:
        members = group & present
        if len(members) < 2:
            continue
        ranked = sorted(members, key=lambda rid: (-by_id[rid].confidence, rid))
        winner = ranked[0]
        for loser in ranked[1:]:
            suppressed_by.setdefault(loser, winner)

    survivors = [d for d in fired if d.rule_id not in suppressed_by]
    suppressed = tuple(
        SuppressedDiagnosis(
            diagnosis=by_id[rid],
            suppressed_by=suppressed_by[rid],
            reason=f"conflicts with higher-confidence {suppressed_by[rid]}",
        )
        for rid in sorted(suppressed_by)
    )
    return survivors, suppressed


# ---------------------------------------------------------------------------
# Stage 3 — annotate and rank
# ---------------------------------------------------------------------------

def _rank(survivors: list[Diagnosis], *, enable_boost: bool) -> tuple[RankedDiagnosis, ...]:
    """Annotate each survivor with its relations and sort into ranked order.

    Sort key ``(-adjusted_confidence, rule_id)`` is a total order, so the
    output is identical regardless of the order rules fired.
    """
    present_ids = {d.rule_id for d in survivors}

    adjusted = [
        (
            d,
            adjust(
                d,
                relations.corroborators_of(d.rule_id, present_ids),
                relations.conflicts_of(d.rule_id, present_ids),
                enable_boost=enable_boost,
            ),
        )
        for d in survivors
    ]
    adjusted.sort(key=lambda pair: (-pair[1].adjusted_confidence, pair[0].rule_id))

    return tuple(
        RankedDiagnosis(
            rank=i + 1,
            diagnosis=d,
            adjusted_confidence=adj.adjusted_confidence,
            corroborated_by=adj.corroborated_by,
            conflicts_with=adj.conflicts_with,
            adjustment_reason=adj.reason,
        )
        for i, (d, adj) in enumerate(adjusted)
    )


# ---------------------------------------------------------------------------
# Input hardening
# ---------------------------------------------------------------------------

def _scan_non_finite(dx: DiagnosisInput) -> list[str]:
    """Report dotted paths to any non-finite float in the input.

    As of schema 1.5.0 the contract itself rejects NaN/inf at construction
    (``allow_inf_nan=False``). This scan stays as defense-in-depth for
    inputs that bypass validation (``model_construct``, deserialized state from
    pre-1.5.0 versions): non-finite values break the engine's deterministic
    ranking, so naming them explicitly is still worth the walk.

    Note the ``Diagnosis`` contract already rejects a NaN confidence, so a
    poisoned field cannot silently reach the ranking sort; it surfaces as a
    rule error instead. This scan exists to name the bad input explicitly.
    """
    bad: list[str] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, bool):
            return  # bool is an int subclass; never non-finite
        if isinstance(value, float):
            if not math.isfinite(value):
                bad.append(path or "<root>")
        elif isinstance(value, dict):
            for key, val in value.items():
                walk(val, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, val in enumerate(value):
                walk(val, f"{path}[{index}]")

    walk(dx.model_dump(), "")
    return [f"non-finite value at {path}" for path in bad]


__all__ = [
    "ENGINE_VERSION",
    "RankedDiagnosis",
    "InsufficientDataNote",
    "SuppressedDiagnosis",
    "RuleError",
    "DiagnosisReport",
    "run_diagnosis",
]
