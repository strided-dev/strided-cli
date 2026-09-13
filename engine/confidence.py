"""Relational confidence policy for the engine.

Per-rule confidence is the rule's own responsibility (see ``rules/base.py``):
the scalar a rule returns is authoritative, and the engine never recomputes it.
What the engine owns is the *relational* adjustment — how a rule's confidence
should move when other rules corroborate or conflict with it.

v1 policy, chosen deliberately for an open-source MVP and for the Day-60
accuracy kill criterion:

    The engine RANKS and ANNOTATES. It does not INVENT.

- Corroboration and conflict are always recorded as annotations.
- The confidence *scalar* changes only when ``enable_boost`` is True, which is
  OFF by default. An invented boost can promote a wrong diagnosis above a right
  one; we do not spend that headroom until real customer dumps validate it.

When the boost is enabled it is bounded and transparent: ``+STEP`` per
corroborating rule, capped at ``CEILING`` (strictly below 1.0, so absolute
certainty stays impossible by construction). This honours r01's self-imposed
0.9 ceiling, which exists precisely to reserve headroom for engine-level
corroboration.
"""

from __future__ import annotations

from dataclasses import dataclass

from rules.base import Diagnosis

# Each corroborating peer adds this much confidence when the boost is enabled.
CORROBORATION_STEP: float = 0.05

# Hard upper bound on engine-adjusted confidence. Strictly below 1.0.
CONFIDENCE_CEILING: float = 0.97


@dataclass(frozen=True)
class ConfidenceAdjustment:
    """How the engine moved (or did not move) a rule's self-reported confidence.

    Both numbers are kept so provenance is never lost. In the v1 default
    (boost off) ``adjusted_confidence == base_confidence`` and the annotations
    still carry the relationships, letting a consumer render
    "71% (corroborated by r03)" without overstating certainty.
    """

    base_confidence: float
    adjusted_confidence: float
    corroborated_by: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    reason: str

    @property
    def delta(self) -> float:
        return self.adjusted_confidence - self.base_confidence


def adjust(
    diagnosis: Diagnosis,
    corroborated_by: tuple[str, ...],
    conflicts_with: tuple[str, ...],
    *,
    enable_boost: bool = False,
) -> ConfidenceAdjustment:
    """Compute the relational adjustment for one diagnosis.

    Pure and deterministic. ``corroborated_by`` and ``conflicts_with`` are the
    already-resolved peer rule_ids for this diagnosis (see ``engine.relations``).
    """
    base = diagnosis.confidence
    if enable_boost and corroborated_by:
        adjusted = min(CONFIDENCE_CEILING, base + CORROBORATION_STEP * len(corroborated_by))
    else:
        adjusted = base

    return ConfidenceAdjustment(
        base_confidence=base,
        adjusted_confidence=adjusted,
        corroborated_by=corroborated_by,
        conflicts_with=conflicts_with,
        reason=_describe(base, adjusted, corroborated_by, conflicts_with),
    )


def _describe(
    base: float,
    adjusted: float,
    corroborated_by: tuple[str, ...],
    conflicts_with: tuple[str, ...],
) -> str:
    """Human-readable, audit-friendly summary of the adjustment."""
    parts: list[str] = []
    if corroborated_by:
        peers = ", ".join(corroborated_by)
        if adjusted > base:
            parts.append(f"corroborated by {peers} (+{adjusted - base:.2f})")
        else:
            parts.append(f"corroborated by {peers}")
    if conflicts_with:
        parts.append(f"conflicts with {', '.join(conflicts_with)}")
    return "; ".join(parts) if parts else "no adjustment"


__all__ = [
    "CORROBORATION_STEP",
    "CONFIDENCE_CEILING",
    "ConfidenceAdjustment",
    "adjust",
]
