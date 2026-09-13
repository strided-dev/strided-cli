"""gait — the human-in-the-loop diagnosis-to-fix agent.

A gait is the disciplined, repeatable sequence of steps by which strided walks a
diagnosis to a verified fix. The package is an explicit state machine that advances
``Diagnosed → Resolved → Proposed → Approved → Applied → Verified`` one deliberate
step at a time, checking its footing before each commit, and dropping to a typed
``Abstained`` from any read-only stage that cannot proceed.

Non-negotiables this package exists to protect:

1. The human is the actor; gait is the assistant. A human approves every state change.
2. ``apply`` is physically unreachable without an ``Approval`` — enforced by types.
3. ``verify`` must be willing to say it failed or that it can't tell.
4. Every "I stopped" carries a structured ``Abstained(reason, payload)``.
5. gait only ever speaks canonical schema (the seam stays at ``DiagnosisInput``).

Dependencies flow one way: gait imports from ``schema``, ``engine``, ``rules``; none
of those import gait. ``cli`` imports gait to expose ``strided fix`` / ``strided undo``.
"""

from gait.apply import apply
from gait.approve import DEFAULT_CONFIDENCE_THRESHOLD, approve, approve_auto
from gait.fixes import FIX_REGISTRY, FixSpec, fix_for
from gait.journal import Journal, JournalEntry, default_journal_path
from gait.propose import propose
from gait.resolve import resolve
from gait.rollback import rollback
from gait.state import (
    Applied,
    Approved,
    Diagnosed,
    Proposed,
    Resolved,
    RolledBack,
    Verified,
)
from gait.targets import ABSENT, ConfigTarget, Location, ResolveOutcome, VllmArgsTarget
from gait.types import (
    Abstained,
    AbstentionReason,
    Approval,
    Change,
    Comparison,
    HumanDecision,
    Prediction,
    PredictionCheck,
    RecommendedChange,
    Snapshot,
    Verdict,
)
from gait.verify import verify

__all__ = [
    # transitions
    "resolve",
    "propose",
    "approve",
    "approve_auto",
    "apply",
    "verify",
    "rollback",
    # states
    "Diagnosed",
    "Resolved",
    "Proposed",
    "Approved",
    "Applied",
    "Verified",
    "RolledBack",
    # value types
    "Abstained",
    "AbstentionReason",
    "Approval",
    "Change",
    "Comparison",
    "HumanDecision",
    "Prediction",
    "PredictionCheck",
    "RecommendedChange",
    "Snapshot",
    "Verdict",
    # targets
    "ABSENT",
    "ConfigTarget",
    "Location",
    "ResolveOutcome",
    "VllmArgsTarget",
    # fixes / journal
    "FixSpec",
    "FIX_REGISTRY",
    "fix_for",
    "Journal",
    "JournalEntry",
    "default_journal_path",
    "DEFAULT_CONFIDENCE_THRESHOLD",
]
