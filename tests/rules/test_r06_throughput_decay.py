"""
Tests for r06 — throughput decay over a sustained run.

The acceptance matrix for r06 plus gate,
mechanism, and confidence coverage. The abstentions are the point: a downtrend
alone never fires. The rule must stay silent when throughput falls because load
tapered (the backlog drained), when nothing explains the fall as work-limited,
and when a clock drops while the GPU is *cooling* (idle down-clock, not a
throttle) — and it must fire when a real, attributable mechanism is building.
"""

from __future__ import annotations

import random

import pytest

from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r06_throughput_decay import (
    ThroughputDecayRule,
    DECAY_MIN,
    MIN_SAMPLES,
    _UNCALIBRATED_CEILING,
)
from schema import DiagnosisInput, ThroughputSample


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _series(
    *,
    n: int = 12,
    dt: float = 20.0,
    start: float = 120.0,
    slope: float = -4.5,        # tok/s per tick
    noise: float = 0.0,
    preempt_start: float | None = None,
    preempt_step: float = 0.0,
    queue_start: float | None = None,
    queue_step: float = 0.0,
    waiting: float | None = 8.0,
    clock_start: float | None = None,
    clock_step: float = 0.0,
    temp_start: float | None = None,
    temp_step: float = 0.0,
    seed: int = 0,
) -> list[ThroughputSample]:
    rng = random.Random(seed)
    out: list[ThroughputSample] = []
    for i in range(n):
        tput = max(0.0, start + slope * i + (rng.uniform(-noise, noise) if noise else 0.0))
        kw: dict = {}
        if preempt_start is not None:
            kw["preemption_rate"] = preempt_start + preempt_step * i
        if queue_start is not None:
            kw["queue_time_ms"] = queue_start + queue_step * i
        if waiting is not None:
            kw["num_requests_waiting"] = waiting
        if clock_start is not None:
            kw["sm_clock_mhz"] = clock_start + clock_step * i
        if temp_start is not None:
            kw["gpu_temp_c"] = temp_start + temp_step * i
        out.append(ThroughputSample(t=i * dt, token_throughput_gen=tput, **kw))
    return out


def _input(history: list[ThroughputSample], engine: str = "vllm") -> DiagnosisInput:
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-8B",
        gpu_type="RTX-3050",
        inference_engine=engine,
        throughput_history=history,
    )


@pytest.fixture
def rule() -> ThroughputDecayRule:
    return ThroughputDecayRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_fires_on_memory_pressure_decay(self, rule: ThroughputDecayRule) -> None:
        # Throughput slides while preemptions AND queue time climb, backlog held.
        result = rule.evaluate(_input(_series(
            preempt_start=0.0, preempt_step=0.004, queue_start=5.0, queue_step=3.0)))
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r06"
        assert result.evidence["mechanism"] == "memory_pressure"
        # vLLM fix is the memory-relief one, not a thermal one.
        assert "gpu-memory-utilization" in result.fix
        assert result.confidence <= _UNCALIBRATED_CEILING

    def test_2_fires_via_queue_time_only(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(_input(_series(queue_start=5.0, queue_step=4.0)))
        assert isinstance(result, Diagnosis)
        assert result.evidence["mechanism"] == "memory_pressure"
        assert result.evidence["queue_time_ms_last"] is not None

    def test_3_fires_on_thermal_throttle(self, rule: ThroughputDecayRule) -> None:
        # No vLLM pressure signals; clock falls as temperature climbs into the band.
        result = rule.evaluate(_input(_series(
            waiting=None, clock_start=1900.0, clock_step=-45.0,
            temp_start=62.0, temp_step=3.0)))
        assert isinstance(result, Diagnosis)
        assert result.evidence["mechanism"] == "thermal_throttle"
        assert "cooling" in result.fix or "thermal" in result.fix.lower()
        assert result.evidence["sm_clock_last"] < result.evidence["sm_clock_first"]

    def test_4_abstains_on_benign_drain(self, rule: ThroughputDecayRule) -> None:
        # Same decline, but the queue drained to zero → demand-driven, not a fault.
        result = rule.evaluate(_input(_series(
            waiting=0.0, preempt_start=0.0, preempt_step=0.004)))
        assert result is Abstention.BELOW_THRESHOLD

    def test_5_abstains_on_flat_throughput(self, rule: ThroughputDecayRule) -> None:
        # No downtrend (slope ~0, noisy) even with rising pressure → no decay.
        result = rule.evaluate(_input(_series(
            slope=0.0, noise=4.0, preempt_start=0.0, preempt_step=0.004)))
        assert result is Abstention.BELOW_THRESHOLD

    def test_6_abstains_on_unattributable_decay(self, rule: ThroughputDecayRule) -> None:
        # Real decline but nothing explains it as work-limited (no pressure/thermal,
        # backlog unknown) → most likely the load tapered. Do not invent a cause.
        result = rule.evaluate(_input(_series(waiting=None)))
        assert result is Abstention.BELOW_THRESHOLD

    def test_7_abstains_on_idle_downclock(self, rule: ThroughputDecayRule) -> None:
        # Clock falls but the GPU is COOLING — an idle down-clock, not a throttle.
        result = rule.evaluate(_input(_series(
            waiting=None, clock_start=1900.0, clock_step=-45.0,
            temp_start=70.0, temp_step=-3.0)))
        assert result is Abstention.BELOW_THRESHOLD

    def test_8_insufficient_data_no_history(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(DiagnosisInput(
            model_name="m", gpu_type="g", inference_engine="vllm"))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("throughput_history",)

    def test_9_insufficient_data_too_few_samples(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(_input(_series(n=MIN_SAMPLES - 1)))
        assert isinstance(result, InsufficientData)
        assert "sample" in result.reason

    def test_10_insufficient_data_window_too_short(self, rule: ThroughputDecayRule) -> None:
        # Enough samples, but compressed into < MIN_WINDOW_S of wall-clock.
        result = rule.evaluate(_input(_series(n=10, dt=3.0,
                                              preempt_start=0.0, preempt_step=0.01)))
        assert isinstance(result, InsufficientData)
        assert "span" in result.reason

    def test_11_abstains_when_no_generation(self, rule: ThroughputDecayRule) -> None:
        # Median throughput is zero (server idle) → nothing to call a decay.
        result = rule.evaluate(_input(_series(start=0.0, slope=0.0,
                                              preempt_start=0.0, preempt_step=0.01)))
        assert result is Abstention.BELOW_THRESHOLD

    def test_12_fires_on_spiky_windowed_preemption_series(self, rule: ThroughputDecayRule) -> None:
        # Schema 1.4.0 feeds *windowed* co-samples: per-window delta preemption
        # rates are spiky (zeros interleaved with bursts) rather than a smooth
        # lifetime ramp. Mann–Kendall's tie-corrected variance must still call
        # the rising envelope significant so the mechanism leg opens. Pinned
        # BEFORE the GPU re-validation so a failure here is a logic bug, not a
        # rig problem.
        base = _series()  # decaying throughput, no co-samples
        spiky = [0.0, 0.0, 0.3, 0.0, 0.6, 0.4, 1.1, 0.0, 0.9, 1.5, 1.2, 1.8]
        history = [
            s.model_copy(update={"preemption_rate": spiky[i]})
            for i, s in enumerate(base)
        ]
        result = rule.evaluate(_input(history))
        assert isinstance(result, Diagnosis)
        assert result.evidence["mechanism"] == "memory_pressure"


# --------------------------------------------------------------------------- #
# Confidence & evidence
# --------------------------------------------------------------------------- #

class TestConfidenceAndEvidence:
    def test_strong_signal_hits_uncalibrated_cap(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(_input(_series(
            start=200.0, slope=-12.0, preempt_start=0.0, preempt_step=0.02,
            queue_start=5.0, queue_step=20.0)))
        assert isinstance(result, Diagnosis)
        assert result.confidence == pytest.approx(_UNCALIBRATED_CEILING)

    def test_gentle_decay_is_less_confident_than_steep(self, rule: ThroughputDecayRule) -> None:
        gentle = rule.evaluate(_input(_series(
            start=100.0, slope=-1.8, preempt_start=0.0, preempt_step=0.0008,
            queue_start=5.0, queue_step=0.5)))
        steep = rule.evaluate(_input(_series(
            start=200.0, slope=-12.0, preempt_start=0.0, preempt_step=0.02,
            queue_start=5.0, queue_step=20.0)))
        assert isinstance(gentle, Diagnosis) and isinstance(steep, Diagnosis)
        assert gentle.confidence_breakdown.signal_strength < steep.confidence_breakdown.signal_strength

    def test_evidence_populated(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(_input(_series(
            preempt_start=0.0, preempt_step=0.004, queue_start=5.0, queue_step=3.0)))
        assert isinstance(result, Diagnosis)
        ev = result.evidence
        assert ev["samples"] == 12
        assert ev["window_seconds"] == pytest.approx(220.0)
        assert ev["decay_frac_per_min"] >= DECAY_MIN
        assert 0.0 < ev["throughput_drop_frac"] <= 1.0
        assert ev["mann_kendall_p"] < 0.10
        assert ev["throughput_first"] > ev["throughput_last"]

    def test_unknown_engine_uses_generic_memory_fix(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(_input(
            _series(preempt_start=0.0, preempt_step=0.004), engine="unknown"))
        assert isinstance(result, Diagnosis)
        assert "gpu-memory-utilization" not in result.fix
        assert "KV cache" in result.fix

    def test_uncalibrated_note_present(self, rule: ThroughputDecayRule) -> None:
        result = rule.evaluate(_input(_series(queue_start=5.0, queue_step=4.0)))
        assert isinstance(result, Diagnosis)
        assert "uncalibrated" in result.confidence_breakdown.notes

    def test_closed_leg_contributes_no_strength(self, rule: ThroughputDecayRule) -> None:
        # A strongly falling SM clock whose thermal gate never opened (the GPU
        # is cooling, not hot) must contribute nothing to mechanism strength:
        # the run must score identically to the same run with no clock/temp
        # series at all. The queue trend is a sawtooth (MK z≈2.2, significant
        # but below Z_STRONG) so a leaked clock z (monotone, ≈4.5, clamps to
        # full strength) would visibly inflate signal_strength.
        def history(with_closed_thermal_leg: bool) -> list[ThroughputSample]:
            queue = [14.0, 11.0, 15.0, 12.0, 16.0, 13.0,
                     17.0, 14.0, 18.0, 15.0, 19.0, 16.0]
            out: list[ThroughputSample] = []
            for i, q in enumerate(queue):
                kw: dict = {"queue_time_ms": q, "num_requests_waiting": 8.0}
                if with_closed_thermal_leg:
                    kw["sm_clock_mhz"] = 1900.0 - 45.0 * i  # falling, max |z|
                    kw["gpu_temp_c"] = 60.0 - 2.0 * i       # cooling: gate shut
                out.append(ThroughputSample(
                    t=i * 20.0, token_throughput_gen=120.0 - 4.5 * i, **kw))
            return out

        bare = rule.evaluate(_input(history(False)))
        with_leg = rule.evaluate(_input(history(True)))
        assert isinstance(bare, Diagnosis) and isinstance(with_leg, Diagnosis)
        # The unopened leg is absent from the label ...
        assert with_leg.evidence["mechanism"] == "memory_pressure"
        # ... and from the strength: the queue's sub-Z_STRONG trend sets it,
        # not the closed thermal leg's saturated clock z.
        assert with_leg.evidence["signal_strength"] == bare.evidence["signal_strength"]
        assert with_leg.confidence == bare.confidence
