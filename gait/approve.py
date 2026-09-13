"""approve — the hard gate. The only place an ``Approval`` is minted.

A human approves every state change. There is no autonomous mode in v1 except an
explicit, opt-in ``--yes`` that is off by default and **never available for
low-confidence diagnoses**. Because :func:`gait.apply.apply` requires an ``Approved``
state, ``Approved`` requires an ``Approval``, and an ``Approval`` can only be
constructed with the private mint key held here, the gate cannot be bypassed.

READ-ONLY (it authorizes; it does not mutate config).
"""

from __future__ import annotations

from datetime import datetime, timezone

from gait.state import Approved, Proposed
from gait.types import (
    _APPROVAL_MINT_KEY,
    Abstained,
    AbstentionReason,
    Approval,
    HumanDecision,
)

# Default bar below which ``--yes`` auto-approval is refused. A diagnosis under this
# confidence always requires a human keystroke.
DEFAULT_CONFIDENCE_THRESHOLD = 0.80


def approve(s: Proposed, decision: HumanDecision) -> Approved | Abstained:
    """Interactive gate: an explicit human yes/no on a rendered proposal."""
    if not decision.approved:
        return Abstained(
            AbstentionReason.DECLINED,
            {
                "param": s.param,
                "proposal_id": s.proposal_id,
                "note": decision.note,
                "detail": "human declined the proposal at the approval gate",
            },
        )
    return Approved(proposal=s, approval=_mint(s.proposal_id, "interactive"))


def approve_auto(
    s: Proposed, *, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Approved | Abstained:
    """``--yes`` gate: auto-approve only when confidence clears the bar.

    Refused for any diagnosis below ``confidence_threshold`` — high-stakes or
    low-confidence changes always require a human keystroke.
    """
    confidence = s.diagnosis.confidence
    if confidence < confidence_threshold:
        return Abstained(
            AbstentionReason.BELOW_CONFIDENCE_FOR_AUTO,
            {
                "confidence": confidence,
                "threshold": confidence_threshold,
                "proposal_id": s.proposal_id,
                "detail": (
                    f"--yes refused: diagnosis confidence {confidence:.0%} is below the "
                    f"{confidence_threshold:.0%} auto-approval bar; human approval required"
                ),
            },
        )
    return Approved(proposal=s, approval=_mint(s.proposal_id, "auto_above_threshold"))


def _mint(proposal_id: str, mode: str) -> Approval:
    """Mint an Approval. The private key makes this the only path to one."""
    return Approval(
        proposal_id=proposal_id,
        approved_at=datetime.now(timezone.utc),
        mode=mode,  # type: ignore[arg-type]
        _mint=_APPROVAL_MINT_KEY,
    )


__all__ = ["approve", "approve_auto", "DEFAULT_CONFIDENCE_THRESHOLD"]
