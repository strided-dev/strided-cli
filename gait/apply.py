"""apply — the one mutating transition. Reachable only with an ``Approval``.

``apply`` takes an ``Approved`` state; the only way to obtain one is through the
approval gate, which mints the ``Approval`` it carries. The hard requirement here is
that undo state is captured *before* mutating: ``apply`` reads the current value,
writes a *pending* journal record, performs the write, then marks it *applied* and
returns a :class:`Change` whose ``rollback`` is already bound to restore the prior
value. A crash between the pending record and the write leaves a recoverable trail.
"""

from __future__ import annotations

import secrets

from gait.journal import Journal
from gait.state import Applied, Approved
from gait.targets.base import ABSENT, ResolveOutcome


def apply(s: Approved, *, journal: Journal | None = None) -> Applied:
    """Perform the approved change, reversibly, recording it to the journal first."""
    journal = journal or Journal.default()
    proposal = s.proposal
    target = proposal.target

    # Sanity: the approval must be for *this* proposal. Cheap, and catches a token
    # being shuffled between proposals.
    if s.approval.proposal_id != proposal.proposal_id:
        raise ValueError(
            "approval/proposal mismatch: "
            f"{s.approval.proposal_id!r} != {proposal.proposal_id!r}"
        )

    # Capture undo state freshly from the live surface (not the cached current_value).
    # A param resolved from an implicit default is recorded as ABSENT, so rollback
    # removes the flag instead of writing the default back as an explicit one.
    loc = target.locate(proposal.param)
    if loc.outcome is ResolveOutcome.RESOLVED:
        prior_value = ABSENT if loc.implicit else loc.value
    else:
        prior_value = proposal.current_value
    new_value = proposal.proposed_value
    change_id = secrets.token_hex(6)

    # Journal BEFORE mutating — recoverable on a mid-apply crash.
    journal.record_pending(
        change_id,
        rule_id=proposal.diagnosis.rule_id,
        param=proposal.param,
        prior_value=prior_value,
        new_value=new_value,
        target_ref=target.ref,
    )

    target.write(proposal.param, new_value)
    journal.mark_applied(change_id)

    def _rollback() -> None:
        target.write(proposal.param, prior_value)
        journal.mark_rolled_back(change_id)

    from gait.types import Change  # local import keeps types <-> apply decoupled

    change = Change(
        change_id=change_id,
        param=proposal.param,
        prior_value=prior_value,
        new_value=new_value,
        target_ref=target.ref,
        rollback=_rollback,
    )
    return Applied(proposal=proposal, change=change)


__all__ = ["apply"]
