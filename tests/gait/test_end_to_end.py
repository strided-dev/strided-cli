"""The spine: one rule (r03) walked all the way through, honestly.

Drives the full state machine from a real engine diagnosis to a verdict, and proves
the structural invariants hold across the whole walk:
  * read-only stages mutate nothing until approval,
  * apply is reachable only with a minted Approval,
  * verify is willing to return the unflattering verdict.
"""

from __future__ import annotations

from engine import run_diagnosis

from gait import (
    Applied,
    Approved,
    Diagnosed,
    HumanDecision,
    Proposed,
    Resolved,
    Verdict,
    Verified,
    apply,
    approve,
    propose,
    resolve,
    verify,
)

from gait_builders import make_snapshot


def _r03_from_engine(snapshot):
    report = run_diagnosis(snapshot)
    ranked = next(r for r in report.diagnoses if r.diagnosis.rule_id == "r03")
    return ranked.diagnosis


def test_full_walk_diagnosed_to_verified(target, journal):
    snapshot = make_snapshot()
    diagnosis = _r03_from_engine(snapshot)

    # Diagnosed → Resolved (read-only)
    resolved = resolve(Diagnosed(diagnosis, snapshot), target)
    assert isinstance(resolved, Resolved)
    assert target.locate("block-size").value == 16  # nothing mutated yet

    # Resolved → Proposed (read-only)
    proposed = propose(resolved)
    assert isinstance(proposed, Proposed)
    assert target.locate("block-size").value == 16  # still nothing mutated

    # Proposed → Approved (the gate)
    approved = approve(proposed, HumanDecision(approved=True))
    assert isinstance(approved, Approved)

    # Approved → Applied (the one mutation)
    applied = apply(approved, journal=journal)
    assert isinstance(applied, Applied)
    assert target.locate("block-size").value == 8

    # Applied → Verified. Re-collecting an improved snapshot → honest verdict.
    # r03's real confidence is uncalibrated-capped (~0.65), so a single clean pair
    # is INCONCLUSIVE, not CONFIRMED — the honest result.
    improved = make_snapshot(frag=0.05, util=0.60)
    verified = verify(applied, lambda: improved)
    assert isinstance(verified, Verified)
    assert verified.verdict in (Verdict.INCONCLUSIVE, Verdict.CONFIRMED)
    assert verified.verdict is Verdict.INCONCLUSIVE  # low-confidence r03 cannot claim causation


def test_read_only_stages_never_touch_config(target, journal):
    """Everything up to (not including) apply leaves the surface byte-identical."""
    before = target.command()
    snapshot = make_snapshot()
    diagnosis = _r03_from_engine(snapshot)

    resolved = resolve(Diagnosed(diagnosis, snapshot), target)
    proposed = propose(resolved)
    approve(proposed, HumanDecision(approved=False))  # declined

    assert target.command() == before  # not one character changed
