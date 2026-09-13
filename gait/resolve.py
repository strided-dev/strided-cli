"""resolve — map the abstract fix onto the actual thing in the user's environment.

This is the highest-risk stage: it is the one most likely to confidently do the
wrong thing, so it is built to refuse rather than guess. It looks up the registered
fix for the diagnosis, asks the declared :class:`ConfigTarget` where that param
lives, and returns ``Resolved`` only on an unambiguous, single hit. On AMBIGUOUS or
NOT_FOUND it returns :class:`Abstained` with a payload naming the param (and, for
ambiguous, the candidates it saw). It never picks one — asking the human here is
cheap; guessing wrong here is the whole ballgame.

READ-ONLY. Safe to run anywhere, any number of times.
"""

from __future__ import annotations

from gait.fixes import fix_for
from gait.state import Diagnosed, Resolved
from gait.targets.base import ConfigTarget, ResolveOutcome
from gait.types import Abstained, AbstentionReason, RecommendedChange


def resolve(s: Diagnosed, target: ConfigTarget) -> Resolved | Abstained:
    """Locate the diagnosis's fix param within ``target``; abstain if unsure."""
    spec = fix_for(s.diagnosis.rule_id)
    if spec is None:
        return Abstained(
            AbstentionReason.NO_FIX_MAPPING,
            {
                "rule_id": s.diagnosis.rule_id,
                "detail": f"no registered config fix for rule {s.diagnosis.rule_id!r}",
            },
        )

    # Stage gate: is this snapshot even the shape the fix applies to?
    abstain = spec.applicable(s.diagnosis, s.snapshot)
    if abstain is not None:
        return abstain

    loc = target.locate(spec.param)

    if loc.outcome is ResolveOutcome.NOT_FOUND:
        return Abstained(
            AbstentionReason.PARAM_NOT_FOUND,
            {
                "param": spec.param,
                "target": target.ref,
                "detail": (
                    f"param {spec.param!r} not found in {target.ref}"
                    + (f" ({loc.note})" if loc.note else "")
                ),
            },
        )

    if loc.outcome is ResolveOutcome.AMBIGUOUS:
        return Abstained(
            AbstentionReason.AMBIGUOUS_CONFIG,
            {
                "param": spec.param,
                "target": target.ref,
                "candidates": list(loc.candidates),
                "detail": (
                    f"param {spec.param!r} is ambiguous in {target.ref}: "
                    f"{len(loc.candidates)} candidates {list(loc.candidates)}"
                ),
            },
        )

    recommended = RecommendedChange(
        param=spec.param,
        proposed_value=spec.propose_value(loc.value),
        rationale=spec.rationale,
    )
    return Resolved(
        diagnosis=s.diagnosis,
        snapshot=s.snapshot,
        target=target,
        param=spec.param,
        current_value=loc.value,
        recommended=recommended,
    )


__all__ = ["resolve"]
