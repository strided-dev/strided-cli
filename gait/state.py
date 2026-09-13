"""The state machine's states — one frozen data object per node.

``gait`` is modeled as an explicit state machine, not a call chain. Each state is an
immutable snapshot of "where we are"; each transition (in :mod:`gait.resolve`,
:mod:`gait.propose`, …) is a function ``State -> NextState | Abstained``, with the
single exception of ``apply`` which additionally requires an ``Approval`` (carried
inside the ``Approved`` state, so it is unreachable otherwise).

The line between ``Proposed`` and ``Approved`` is the only place read-only becomes
mutating. Everything at or above ``Proposed`` is safe to run anywhere, any number of
times — in CI, a dry run, a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rules.base import Diagnosis

from gait.targets.base import ConfigTarget
from gait.types import (
    Approval,
    Change,
    Prediction,
    RecommendedChange,
    Snapshot,
    Verdict,
)


@dataclass(frozen=True)
class Diagnosed:
    """Entry state: a single diagnosis from the engine plus the snapshot it fired on."""

    diagnosis: Diagnosis
    snapshot: Snapshot


@dataclass(frozen=True)
class Resolved:
    """The abstract fix mapped to a concrete place in the user's environment."""

    diagnosis: Diagnosis
    snapshot: Snapshot
    target: ConfigTarget
    param: str
    current_value: Any
    recommended: RecommendedChange


@dataclass(frozen=True)
class Proposed:
    """A fully-described pending change. Nothing has been mutated."""

    diagnosis: Diagnosis
    snapshot: Snapshot
    target: ConfigTarget
    param: str
    current_value: Any
    proposed_value: Any
    prediction: Prediction
    preview: str
    proposal_id: str


@dataclass(frozen=True)
class Approved:
    """A proposal a human authorized. Carries the (gate-minted) approval token."""

    proposal: Proposed
    approval: Approval


@dataclass(frozen=True)
class Applied:
    """The change has been made; ``change`` holds the prior value and bound undo."""

    proposal: Proposed
    change: Change


@dataclass(frozen=True)
class Verified:
    """The four-way verdict of comparing a fresh snapshot to the prediction."""

    applied: Applied
    verdict: Verdict
    before: dict[str, Any]
    after: dict[str, Any]
    detail: str


@dataclass(frozen=True)
class RolledBack:
    """Terminal: the prior value was restored."""

    change: Change
    restored_value: Any


__all__ = [
    "Diagnosed",
    "Resolved",
    "Proposed",
    "Approved",
    "Applied",
    "Verified",
    "RolledBack",
]
