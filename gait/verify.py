"""verify — the stage that must never lie.

After a change, re-collect a fresh snapshot and compare it against the
:class:`Prediction` recorded at propose time. The verdict is four-way and ``gait``
must be willing to return the unflattering ones:

* ``CONFIRMED``          — the prediction materialized.
* ``NO_CHANGE``          — no improvement, or a regression.
* ``INCONCLUSIVE``       — the signal moved but traffic shifted (or the diagnosis was
                           too low-confidence to attribute a single before/after pair).
* ``INSUFFICIENT_DATA``  — could not collect a clean verifying snapshot.

A confident-but-wrong "it worked" is the single worst thing this product can do, so
abstaining here is correct behavior, not a bug. Under shifting live traffic you
cannot prove causation from one before/after pair, so ``verify`` samples the traffic
indicators on each side and returns ``INCONCLUSIVE`` when they moved materially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from gait.approve import DEFAULT_CONFIDENCE_THRESHOLD
from gait.state import Applied, Verified
from gait.types import Prediction, Snapshot, Verdict

# How far a traffic indicator may drift (relative) before we refuse to attribute a
# metric change to our edit rather than to the workload moving underneath us.
DEFAULT_TRAFFIC_TOLERANCE = 0.25

# Fields we treat as "traffic": if these moved, the comparison is contaminated.
_TRAFFIC_FIELDS = ("request_throughput", "seq_len_distribution.mean")

Collector = Callable[[], Optional[Snapshot]]


@dataclass(frozen=True)
class _TrafficCheck:
    shifted: bool
    detail: str
    measured: bool = False  # was at least one traffic indicator present on both sides?


def verify(
    s: Applied,
    collector: Collector,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    traffic_tolerance: float = DEFAULT_TRAFFIC_TOLERANCE,
) -> Verified:
    """Re-collect, compare to the recorded prediction, return an honest verdict."""
    before = s.proposal.snapshot
    prediction = s.proposal.prediction

    # 1. Re-collect. A failed or empty collection is INSUFFICIENT_DATA, never a guess.
    try:
        after = collector()
    except Exception as exc:  # noqa: BLE001 — boundary; report, don't crash
        return _verified(
            s, Verdict.INSUFFICIENT_DATA, before, None, prediction,
            f"re-collection raised {type(exc).__name__}: {exc}",
        )
    if after is None:
        return _verified(
            s, Verdict.INSUFFICIENT_DATA, before, None, prediction,
            "re-collection returned no snapshot",
        )

    # 2. Did the verifying snapshot even carry the fields we predicted on?
    missing = [c.field for c in prediction.checks if _resolve_field(after, c.field) is None]
    if missing:
        return _verified(
            s, Verdict.INSUFFICIENT_DATA, before, after, prediction,
            f"verifying snapshot missing predicted field(s): {missing}",
        )

    # 3. Traffic-shift guard. Can't attribute a metric move if the workload moved.
    traffic = _traffic_shifted(before, after, traffic_tolerance)
    if traffic.shifted:
        return _verified(
            s, Verdict.INCONCLUSIVE, before, after, prediction,
            f"traffic shifted between samples ({traffic.detail}); cannot attribute",
        )

    # Gate only on the primary (non-advisory) checks. Advisory checks are reported
    # in before/after but must not mask a real primary win (e.g. a defrag that holds
    # while utilization legitimately stays high).
    all_hold = all(
        c.holds(_resolve_field(after, c.field))
        for c in prediction.checks
        if not c.advisory
    )

    # 4. Low-confidence diagnoses cannot earn a CONFIRMED from a single pair: even a
    #    clean before/after is not strong enough evidence. Bias toward INCONCLUSIVE.
    confidence = s.proposal.diagnosis.confidence
    if all_hold and confidence < confidence_threshold:
        return _verified(
            s, Verdict.INCONCLUSIVE, before, after, prediction,
            (
                f"prediction held, but diagnosis confidence {confidence:.0%} is below "
                f"the {confidence_threshold:.0%} attribution bar; not claiming causation"
            ),
        )

    # 5. Don't claim CONFIRMED if traffic stability could not be verified at all: with
    #    no measurable traffic indicator on both sides we cannot rule out the workload
    #    moving underneath us, so attribute nothing.
    if all_hold and not traffic.measured:
        return _verified(
            s, Verdict.INCONCLUSIVE, before, after, prediction,
            "prediction held, but no traffic indicator was measurable on both "
            "sides; cannot rule out a workload shift, so not claiming causation",
        )

    if all_hold:
        return _verified(
            s, Verdict.CONFIRMED, before, after, prediction,
            "all predicted effects materialized",
        )

    return _verified(
        s, Verdict.NO_CHANGE, before, after, prediction,
        "predicted effect did not materialize; consider rollback",
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _verified(
    s: Applied,
    verdict: Verdict,
    before: Snapshot,
    after: Optional[Snapshot],
    prediction: Prediction,
    detail: str,
) -> Verified:
    fields = [c.field for c in prediction.checks] + list(_TRAFFIC_FIELDS)
    before_nums = {f: _resolve_field(before, f) for f in fields}
    after_nums = {f: (_resolve_field(after, f) if after is not None else None) for f in fields}
    return Verified(
        applied=s,
        verdict=verdict,
        before=before_nums,
        after=after_nums,
        detail=detail,
    )


def _traffic_shifted(before: Snapshot, after: Snapshot, tolerance: float) -> _TrafficCheck:
    """Flag if any traffic indicator drifted beyond ``tolerance`` (relative).

    Also reports whether *any* indicator was measurable on both sides. An absent
    indicator can't be claimed shifted — but unknown traffic is not the same as
    stable traffic, so a caller about to award CONFIRMED treats "nothing measurable"
    as a reason to stay INCONCLUSIVE rather than attribute a change to our edit.
    """
    measured_any = False
    for f in _TRAFFIC_FIELDS:
        b = _resolve_field(before, f)
        a = _resolve_field(after, f)
        if b is None or a is None:
            continue
        measured_any = True
        if b == 0:
            if a != 0:
                return _TrafficCheck(True, f"{f} 0 → {a}", measured=True)
            continue
        rel = abs(a - b) / abs(b)
        if rel > tolerance:
            return _TrafficCheck(True, f"{f} {b} → {a} ({rel:.0%} drift)", measured=True)
    detail = "traffic indicators stable" if measured_any else "no traffic indicators measurable"
    return _TrafficCheck(False, detail, measured=measured_any)


def _resolve_field(snapshot: Optional[Snapshot], dotted: str) -> Optional[Any]:
    """Walk a dotted path into the snapshot; None if any hop is missing/None."""
    if snapshot is None:
        return None
    cur: Any = snapshot
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


__all__ = ["verify", "Collector", "DEFAULT_TRAFFIC_TOLERANCE"]
