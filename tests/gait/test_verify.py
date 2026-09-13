"""verify — the four-way verdict that must be willing to say it can't tell."""

from __future__ import annotations

from gait import (
    Diagnosed,
    HumanDecision,
    Verdict,
    apply,
    approve,
    propose,
    resolve,
    verify,
)

from gait_builders import make_diagnosis, make_snapshot


def _applied(target, journal, *, confidence=0.65):
    snap = make_snapshot()
    p = propose(resolve(Diagnosed(make_diagnosis(confidence=confidence), snap), target))
    return apply(approve(p, HumanDecision(True)), journal=journal)


def test_confirmed_when_prediction_materializes(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    improved = make_snapshot(frag=0.10, util=0.70, throughput=10.0)
    v = verify(applied, lambda: improved, confidence_threshold=0.80)
    assert v.verdict is Verdict.CONFIRMED


def test_no_change_when_prediction_fails(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    same = make_snapshot(frag=0.47, util=0.86)  # no improvement
    v = verify(applied, lambda: same, confidence_threshold=0.80)
    assert v.verdict is Verdict.NO_CHANGE


def test_inconclusive_when_traffic_shifts(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    # Metrics improved, but request throughput tripled → cannot attribute.
    shifted = make_snapshot(frag=0.10, util=0.70, throughput=30.0)
    v = verify(applied, lambda: shifted, confidence_threshold=0.80)
    assert v.verdict is Verdict.INCONCLUSIVE
    assert "traffic" in v.detail


def test_inconclusive_when_traffic_unmeasurable(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    # Prediction holds, but the fresh snapshot carries no traffic indicator at all →
    # we can't rule out a workload shift, so we must not award CONFIRMED.
    blind_traffic = make_snapshot(frag=0.10, util=0.70, throughput=None, seqlen_mean=None)
    v = verify(applied, lambda: blind_traffic, confidence_threshold=0.80)
    assert v.verdict is Verdict.INCONCLUSIVE
    assert "traffic" in v.detail


def test_confirmed_when_only_advisory_check_fails(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    # Fragmentation (primary) drops below floor; utilization (advisory) legitimately
    # stays high because the workload needs the cache. A real win must read CONFIRMED,
    # not NO_CHANGE — the advisory clause does not gate the verdict.
    win = make_snapshot(frag=0.10, util=0.88, throughput=10.0)
    v = verify(applied, lambda: win, confidence_threshold=0.80)
    assert v.verdict is Verdict.CONFIRMED


def test_inconclusive_when_confidence_too_low(target, journal):
    applied = _applied(target, journal, confidence=0.65)
    improved = make_snapshot(frag=0.10, util=0.70, throughput=10.0)
    # Prediction holds, but a low-confidence diagnosis cannot earn CONFIRMED.
    v = verify(applied, lambda: improved, confidence_threshold=0.80)
    assert v.verdict is Verdict.INCONCLUSIVE


def test_insufficient_data_when_no_snapshot(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    v = verify(applied, lambda: None, confidence_threshold=0.80)
    assert v.verdict is Verdict.INSUFFICIENT_DATA


def test_insufficient_data_when_collector_raises(target, journal):
    applied = _applied(target, journal, confidence=0.90)

    def boom():
        raise RuntimeError("scrape failed")

    v = verify(applied, boom, confidence_threshold=0.80)
    assert v.verdict is Verdict.INSUFFICIENT_DATA


def test_insufficient_data_when_predicted_field_absent(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    # Fresh snapshot lacks the predicted fields (fragmentation/util are None).
    blind = make_snapshot(frag=None, util=None)
    v = verify(applied, lambda: blind, confidence_threshold=0.80)
    assert v.verdict is Verdict.INSUFFICIENT_DATA


def test_verified_reports_before_and_after(target, journal):
    applied = _applied(target, journal, confidence=0.90)
    improved = make_snapshot(frag=0.10, util=0.70)
    v = verify(applied, lambda: improved, confidence_threshold=0.80)
    assert v.before["kv_cache_fragmentation"] == 0.47
    assert v.after["kv_cache_fragmentation"] == 0.10
