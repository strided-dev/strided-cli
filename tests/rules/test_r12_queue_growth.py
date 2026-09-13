"""r12 — Queue growth (super-linear waiting-queue trend).

The four mandatory categories: fires (super-linear), abstains (linear / bursty /
drained), InsufficientData (one-shot / too-few / too-short), and the metamorphic
properties (monotonicity, boundary). Fixtures are synthetic ThroughputSample
series; calibration against real GPU spirals is the open sprint work (thresholds
are seeds while THRESHOLDS_UNCALIBRATED is True).
"""

from __future__ import annotations

import pytest

from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r12_queue_growth import QueueGrowthRule
from schema import DiagnosisInput
from schema.diagnosis_input import ThroughputSample

_BASE = dict(model_name="m", gpu_type="H100", inference_engine="vllm")
_DT = 20.0  # seconds per tick


def _dx(
    waiting: list[float],
    *,
    queue_ms: list[float] | None = None,
    dt: float = _DT,
    kv: float | None = None,
) -> DiagnosisInput:
    hist = []
    for i, w in enumerate(waiting):
        q = queue_ms[i] if queue_ms is not None else None
        hist.append(
            ThroughputSample(t=i * dt, token_throughput_gen=500.0, num_requests_waiting=w, queue_time_ms=q)
        )
    return DiagnosisInput(**_BASE, throughput_history=hist, kv_cache_util=kv)


@pytest.fixture
def rule() -> QueueGrowthRule:
    return QueueGrowthRule()


# Accelerating backlog: growth rate itself increases (a spiral).
_SPIRAL = [0, 1, 2, 4, 7, 11, 16, 22]
# Steady overload: backlog rises at a constant rate (absorbed, not a spiral).
_LINEAR = [0, 2, 4, 6, 8, 10, 12, 14]


class TestAcceptanceMatrix:
    def test_1_fires_on_super_linear_backlog(self, rule: QueueGrowthRule) -> None:
        res = rule.evaluate(_dx(_SPIRAL))
        assert isinstance(res, Diagnosis)
        assert res.rule_id == "r12"
        assert res.evidence["acceleration_ratio"] > 1.5

    def test_2_abstains_on_linear_backlog(self, rule: QueueGrowthRule) -> None:
        # Rising but not accelerating — steady overload, not a spiral.
        assert rule.evaluate(_dx(_LINEAR)) is Abstention.BELOW_THRESHOLD

    def test_3_abstains_on_bursty_mean_reverting(self, rule: QueueGrowthRule) -> None:
        # No monotonic trend — Mann-Kendall must reject it.
        assert rule.evaluate(_dx([3, 0, 4, 1, 3, 0, 4, 1])) is Abstention.BELOW_THRESHOLD

    def test_4_abstains_when_queue_drained(self, rule: QueueGrowthRule) -> None:
        # Oscillates around zero — no real queue, below WAITING_FLOOR.
        assert rule.evaluate(_dx([0, 0, 1, 0, 0, 1, 0, 0])) is Abstention.BELOW_THRESHOLD


class TestInsufficientData:
    def test_one_shot_has_no_history(self, rule: QueueGrowthRule) -> None:
        res = rule.evaluate(DiagnosisInput(**_BASE))
        assert isinstance(res, InsufficientData)
        assert "throughput_history" in res.missing

    def test_too_few_samples(self, rule: QueueGrowthRule) -> None:
        assert isinstance(rule.evaluate(_dx([0, 1, 4])), InsufficientData)

    def test_window_too_short(self, rule: QueueGrowthRule) -> None:
        # Enough samples but packed into < MIN_WINDOW_S.
        assert isinstance(rule.evaluate(_dx(_SPIRAL, dt=2.0)), InsufficientData)

    def test_no_waiting_series(self, rule: QueueGrowthRule) -> None:
        # History present, but the backlog gauge is absent from every sample.
        hist = [ThroughputSample(t=i * _DT, token_throughput_gen=500.0) for i in range(8)]
        res = rule.evaluate(DiagnosisInput(**_BASE, throughput_history=hist))
        assert isinstance(res, InsufficientData)
        assert any("num_requests_waiting" in m for m in res.missing)


class TestMetamorphic:
    def test_uncalibrated_cap(self, rule: QueueGrowthRule) -> None:
        res = rule.evaluate(_dx(_SPIRAL))
        assert isinstance(res, Diagnosis)
        assert res.confidence <= 0.65  # capped while THRESHOLDS_UNCALIBRATED

    def test_monotonicity_sharper_spiral_not_less_confident(self, rule: QueueGrowthRule) -> None:
        # A strictly sharper acceleration must not lower signal_strength.
        gentle = rule.evaluate(_dx([0, 1, 2, 4, 6, 9, 12, 16]))
        sharp = rule.evaluate(_dx([0, 1, 2, 4, 8, 15, 25, 40]))
        assert isinstance(gentle, Diagnosis) and isinstance(sharp, Diagnosis)
        assert sharp.confidence_breakdown.signal_strength >= gentle.confidence_breakdown.signal_strength

    def test_strictly_increasing_timestamp_gate(self, rule: QueueGrowthRule) -> None:
        # #15 port: a duplicate-timestamp sample is dropped, not counted as a
        # trend point. Injecting a stalled scrape must not change the verdict.
        clean = rule.evaluate(_dx(_SPIRAL))
        stalled = list(_SPIRAL)
        stalled.insert(4, _SPIRAL[3])  # a repeated backlog reading...
        hist = []
        prev_t = None
        for i, w in enumerate(_SPIRAL):
            hist.append(ThroughputSample(t=i * _DT, token_throughput_gen=500.0, num_requests_waiting=w))
            if i == 3:  # ...arriving at a non-advancing timestamp
                hist.append(ThroughputSample(t=3 * _DT, token_throughput_gen=500.0, num_requests_waiting=w))
        res = rule.evaluate(DiagnosisInput(**_BASE, throughput_history=hist))
        assert isinstance(clean, Diagnosis) and isinstance(res, Diagnosis)
        assert res.evidence["samples"] == clean.evidence["samples"]  # stalled sample dropped

    def test_queue_time_corroborator_populates_completeness(self, rule: QueueGrowthRule) -> None:
        with_q = rule.evaluate(_dx(_SPIRAL, queue_ms=[10, 20, 35, 60, 100, 160, 240, 340]))
        without_q = rule.evaluate(_dx(_SPIRAL))
        assert isinstance(with_q, Diagnosis) and isinstance(without_q, Diagnosis)
        assert with_q.confidence_breakdown.data_completeness > without_q.confidence_breakdown.data_completeness


class TestKvHeadroomSelfGuard:
    """An accelerating backlog with ample KV headroom is R-SCHED-CAP's signature
    (artificial ceiling — raise it), the opposite remedy to r12's shed/route.
    The guard must abstain on headroom, fire on genuine pressure, and fire
    (with reduced completeness) when the gauge is absent."""

    def test_abstains_on_kv_headroom(self, rule: QueueGrowthRule) -> None:
        assert rule.evaluate(_dx(_SPIRAL, kv=0.5)) is Abstention.BELOW_THRESHOLD

    def test_fires_under_genuine_kv_pressure(self, rule: QueueGrowthRule) -> None:
        res = rule.evaluate(_dx(_SPIRAL, kv=0.92))
        assert isinstance(res, Diagnosis)
        assert res.evidence["kv_cache_util"] == 0.92

    def test_fires_without_gauge_at_reduced_completeness(self, rule: QueueGrowthRule) -> None:
        no_kv = rule.evaluate(_dx(_SPIRAL))
        with_kv = rule.evaluate(_dx(_SPIRAL, kv=0.92))
        assert isinstance(no_kv, Diagnosis) and isinstance(with_kv, Diagnosis)
        assert no_kv.confidence_breakdown.data_completeness < with_kv.confidence_breakdown.data_completeness
