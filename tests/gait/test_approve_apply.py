"""The hard gate, the unforgeable Approval, apply's reversibility, the journal."""

from __future__ import annotations

import dataclasses

import pytest

from gait import (
    ABSENT,
    Abstained,
    AbstentionReason,
    Approval,
    Approved,
    Diagnosed,
    HumanDecision,
    VllmArgsTarget,
    apply,
    approve,
    approve_auto,
    propose,
    resolve,
    rollback,
)
from gait.journal import STATUS_APPLIED, STATUS_PENDING, STATUS_ROLLED_BACK
from gait.types import _APPROVAL_MINT_KEY

from gait_builders import make_diagnosis, make_snapshot


def _proposed(target, diagnosis, snapshot):
    return propose(resolve(Diagnosed(diagnosis, snapshot), target))


# -- the gate --------------------------------------------------------------- #

def test_decline_abstains(target, diagnosis, snapshot):
    p = _proposed(target, diagnosis, snapshot)
    out = approve(p, HumanDecision(approved=False, note="not now"))
    assert isinstance(out, Abstained)
    assert out.reason is AbstentionReason.DECLINED


def test_approve_yes_mints_interactive_approval(target, diagnosis, snapshot):
    p = _proposed(target, diagnosis, snapshot)
    out = approve(p, HumanDecision(approved=True))
    assert isinstance(out, Approved)
    assert out.approval.mode == "interactive"
    assert out.approval.proposal_id == p.proposal_id


def test_yes_flag_refused_below_threshold(target, snapshot):
    p = _proposed(target, make_diagnosis(confidence=0.65), snapshot)
    out = approve_auto(p, confidence_threshold=0.80)
    assert isinstance(out, Abstained)
    assert out.reason is AbstentionReason.BELOW_CONFIDENCE_FOR_AUTO


# -- reversibility: an implicit-default param restores to absent ------------- #

def test_rollback_restores_implicit_param_to_absent(journal):
    # A command without --block-size: the param resolves to its default, so apply
    # must record the prior as ABSENT and rollback must *remove* the flag, leaving
    # the command byte-identical — not an explicit --block-size 16 that wasn't there.
    original = "python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B --max-num-seqs 256"
    target = VllmArgsTarget.from_command(original)
    p = _proposed(target, make_diagnosis(confidence=0.90), make_snapshot())
    applied = apply(approve(p, HumanDecision(True)), journal=journal)

    assert applied.change.prior_value is ABSENT
    assert "--block-size 8" in target.command()
    # The ABSENT sentinel survives the JSONL round-trip.
    assert journal.get(applied.change.change_id).prior_value is ABSENT

    rollback(applied)
    assert "block-size" not in target.command()
    assert target.command() == original


def test_yes_flag_allowed_above_threshold(target, snapshot):
    p = _proposed(target, make_diagnosis(confidence=0.90), snapshot)
    out = approve_auto(p, confidence_threshold=0.80)
    assert isinstance(out, Approved)
    assert out.approval.mode == "auto_above_threshold"


# -- Approval cannot be forged --------------------------------------------- #

def test_approval_cannot_be_constructed_directly():
    with pytest.raises(PermissionError):
        Approval(proposal_id="x", approved_at=None, mode="interactive")


def test_apply_is_unreachable_without_an_approval(target, diagnosis, snapshot, journal):
    """apply requires an Approved state; Approved requires an Approval; Approval can
    only be minted by the gate. There is no way to call apply on a bare Proposed."""
    p = _proposed(target, diagnosis, snapshot)
    with pytest.raises((TypeError, AttributeError)):
        apply(p, journal=journal)  # type: ignore[arg-type]  — Proposed is not Approved


def test_apply_rejects_mismatched_approval(target, diagnosis, snapshot, journal):
    p = _proposed(target, diagnosis, snapshot)
    approved = approve(p, HumanDecision(approved=True))
    # Forge a *valid* approval (via the gate) but for a different proposal id, then
    # smuggle it onto this proposal — apply must catch the mismatch.
    wrong = Approval(
        proposal_id="different",
        approved_at=approved.approval.approved_at,
        mode="interactive",
        _mint=_APPROVAL_MINT_KEY,
    )
    tampered = dataclasses.replace(approved, approval=wrong)
    with pytest.raises(ValueError):
        apply(tampered, journal=journal)


# -- apply + journal + rollback -------------------------------------------- #

def test_apply_mutates_and_journals(target, diagnosis, snapshot, journal):
    p = _proposed(target, diagnosis, snapshot)
    applied = apply(approve(p, HumanDecision(True)), journal=journal)

    assert target.locate("block-size").value == 8
    assert applied.change.prior_value == 16
    assert applied.change.new_value == 8

    statuses = [e.status for e in journal.entries() if e.change_id == applied.change.change_id]
    # pending recorded BEFORE the write, then applied.
    assert statuses == [STATUS_PENDING, STATUS_APPLIED]


def test_journal_records_pending_before_mutation(target, diagnosis, snapshot, journal):
    """If write fails, the pending record must already be on disk (crash recovery)."""
    p = _proposed(target, diagnosis, snapshot)

    original_write = target.write

    def explode(param, value):
        raise RuntimeError("disk full")

    target.write = explode  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError):
            apply(approve(p, HumanDecision(True)), journal=journal)
    finally:
        target.write = original_write  # type: ignore[method-assign]

    # The pending record survived the crash; it is recoverable.
    entries = journal.entries()
    assert entries and entries[-1].status == STATUS_PENDING
    assert entries[-1].prior_value == 16


def test_rollback_restores_and_marks_journal(target, diagnosis, snapshot, journal):
    p = _proposed(target, diagnosis, snapshot)
    applied = apply(approve(p, HumanDecision(True)), journal=journal)
    rolled = rollback(applied)

    assert rolled.restored_value == 16
    assert target.locate("block-size").value == 16
    assert journal.current_status(applied.change.change_id) == STATUS_ROLLED_BACK


def test_last_applied_skips_rolled_back(target, diagnosis, snapshot, journal):
    p = _proposed(target, diagnosis, snapshot)
    applied = apply(approve(p, HumanDecision(True)), journal=journal)
    assert journal.last_applied().change_id == applied.change.change_id
    rollback(applied)
    assert journal.last_applied() is None
