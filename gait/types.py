"""Core value types for the ``gait`` agent.

These are the nouns the state machine passes around: the abstract change a fix
recommends, the prediction recorded *before* acting, the approval token that makes
``apply`` reachable, the change record that makes it reversible, and the verdict
``verify`` returns. The state objects themselves live in :mod:`gait.state`.

Two invariants are enforced *here*, at the type level, rather than by discipline:

* ``Approval`` cannot be constructed outside the approval gate. Its ``__init__``
  demands a module-private mint key that only :mod:`gait.approve` holds.
* Every "I stopped" is an :class:`Abstained` carrying a typed reason and a payload
  that names *what* was missing — never a bare ``None``.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal

from schema import DiagnosisInput

# ``gait`` only ever speaks canonical schema. The snapshot it reasons over is a
# ``DiagnosisInput`` — the same object the engine fired on. The alias keeps the
# package's vocabulary ("snapshot") without inventing a parallel type.
Snapshot = DiagnosisInput


# --------------------------------------------------------------------------- #
# Abstention — a first-class result of every read-only stage
# --------------------------------------------------------------------------- #

class AbstentionReason(Enum):
    """Why ``gait`` stopped. Every value pairs with a payload naming the gap."""

    NO_FIX_MAPPING = "no_fix_mapping"
    """The diagnosis has no registered config fix gait knows how to apply."""

    DATA_MISSING = "data_missing"
    """The snapshot lacks a field needed to size or justify the fix."""

    PARAM_NOT_FOUND = "param_not_found"
    """resolve: the param does not exist in the declared config surface."""

    AMBIGUOUS_CONFIG = "ambiguous_config"
    """resolve: several candidate locations; gait refuses to guess one."""

    NO_OP = "no_op"
    """propose: the recommended value equals the current value — nothing to do."""

    DECLINED = "declined"
    """approve: the human said no at the gate."""

    BELOW_CONFIDENCE_FOR_AUTO = "below_confidence_for_auto"
    """approve: ``--yes`` is refused for a diagnosis below the confidence bar."""

    APPROVAL_MISMATCH = "approval_mismatch"
    """apply: the approval token does not match the proposal it accompanies."""


@dataclass(frozen=True)
class Abstained:
    """A typed, terminal stop. ``payload`` always names what was missing or why."""

    reason: AbstentionReason
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        detail = self.payload.get("detail")
        if detail:
            return f"{self.reason.value}: {detail}"
        named = {k: v for k, v in self.payload.items() if k != "detail"}
        return f"{self.reason.value}: {named}" if named else self.reason.value


# --------------------------------------------------------------------------- #
# RecommendedChange — the abstract fix, before it is located in the environment
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecommendedChange:
    """What a fix wants to do, in the abstract: a param and a target value."""

    param: str
    proposed_value: Any
    rationale: str


# --------------------------------------------------------------------------- #
# Prediction — the checkable commitment, recorded BEFORE acting
# --------------------------------------------------------------------------- #

class Comparison(Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    def holds(self, value: float, threshold: float) -> bool:
        if self is Comparison.LT:
            return value < threshold
        if self is Comparison.LE:
            return value <= threshold
        if self is Comparison.GT:
            return value > threshold
        return value >= threshold


@dataclass(frozen=True)
class PredictionCheck:
    """One concrete, checkable effect: ``field comparison threshold``.

    ``advisory`` marks a soft, second-order signal: ``verify`` reports it in the
    before/after but does **not** gate the verdict on it, so a real primary win is
    not masked by an advisory clause that legitimately did not move.
    """

    field: str  # dotted path into the snapshot, e.g. "kv_cache_fragmentation"
    comparison: Comparison
    threshold: float
    description: str
    advisory: bool = False

    def holds(self, value: float) -> bool:
        return self.comparison.holds(value, self.threshold)


@dataclass(frozen=True)
class Prediction:
    """The set of effects ``verify`` will check the fresh snapshot against.

    Recorded at ``propose`` time so verification tests a commitment made in
    advance, not a story told afterward.
    """

    checks: tuple[PredictionCheck, ...]
    summary: str


# --------------------------------------------------------------------------- #
# Approval — minted ONLY by the gate
# --------------------------------------------------------------------------- #

# Module-private. The only holders are this module and gait.approve. Importing it
# elsewhere to forge an Approval is possible in Python, but it is no longer an
# accident: you have to reach for a clearly private symbol on purpose.
_APPROVAL_MINT_KEY = object()

ApprovalMode = Literal["interactive", "explicit_flag", "auto_above_threshold"]


@dataclass(frozen=True)
class Approval:
    """Proof a human authorized a specific proposal. ``apply`` requires one.

    The constructor refuses to build an ``Approval`` unless handed the private
    mint key, which only :func:`gait.approve.approve` (and its ``--yes`` sibling)
    possesses. This makes the approval gate unbypassable *by accident*: a new call
    site cannot reach ``apply`` without going through the gate.

    It is an accident-prevention guardrail, **not** a security boundary against an
    in-process adversary — Python has no true private state (the mint key is
    importable, and ``object.__setattr__`` defeats ``frozen``). Don't market it as
    a control against malicious in-process code.
    """

    proposal_id: str
    approved_at: datetime
    mode: ApprovalMode
    _mint: InitVar[Any] = None

    def __post_init__(self, _mint: Any) -> None:
        if _mint is not _APPROVAL_MINT_KEY:
            raise PermissionError(
                "Approval can only be minted by the approval gate "
                "(gait.approve.approve / approve_auto), never constructed directly."
            )


# --------------------------------------------------------------------------- #
# Change — reversible by construction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Change:
    """A mutation that was made, with the bound undo to reverse it."""

    change_id: str
    param: str
    prior_value: Any
    new_value: Any
    target_ref: str
    rollback: Callable[[], None]


# --------------------------------------------------------------------------- #
# Verdict — the four-way, honest result of verification
# --------------------------------------------------------------------------- #

class Verdict(Enum):
    CONFIRMED = "confirmed"
    """The recorded prediction materialized. Report before/after numbers."""

    NO_CHANGE = "no_change"
    """No improvement, or a regression. Say so; offer rollback."""

    INCONCLUSIVE = "inconclusive"
    """Signal moved but traffic shifted (or confidence too low to attribute)."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Could not collect a clean verifying snapshot, or it lacked the fields."""


@dataclass(frozen=True)
class HumanDecision:
    """The human's answer at the approval gate."""

    approved: bool
    note: str = ""


__all__ = [
    "Snapshot",
    "AbstentionReason",
    "Abstained",
    "RecommendedChange",
    "Comparison",
    "PredictionCheck",
    "Prediction",
    "Approval",
    "ApprovalMode",
    "Change",
    "Verdict",
    "HumanDecision",
]
