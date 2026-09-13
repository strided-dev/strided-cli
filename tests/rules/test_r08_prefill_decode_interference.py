"""
Tests for r08 — Prefill↔decode interference (timeline).

The acceptance matrix plus helper-level coverage. The abstentions are the point:
r08 must stay silent on a well-chunked workload (the dominant false positive) and
must not invent a budget when the trace carried no token counts. The engine-level
test pins the live r02↔r08 corroboration — the first relation strided ships.
"""

from __future__ import annotations

import pytest

from engine import run_diagnosis
from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r08_prefill_decode_interference import (
    INTERFERENCE_MIN,
    TAIL_MIN,
    PrefillDecodeInterferenceRule,
    _percentile,
    _TimelineStats,
)
from schema import (
    DiagnosisInput,
    Distribution,
    EngineStep,
    NsysTimeline,
    PhaseMetrics,
    VllmServingMetrics,
)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def _input(
    *,
    n_decode: int = 32,
    n_prefill: int = 8,
    decode_ms: float = 10.0,
    prefill_ms: float = 120.0,
    prefill_phase: str = "prefill",
    prefill_tokens: int | None = 8192,
    decode_seqs: int | None = 16,
    include_timeline: bool = True,
    trace_duration: bool = True,
    ttft: bool = False,
) -> DiagnosisInput:
    """Build a DiagnosisInput with a synthetic timeline.

    Per-phase stats are order-independent, so decode steps then prefill-bearing
    steps back-to-back gives the same tail/interference as an interleaved trace.
    """
    timeline = None
    if include_timeline:
        steps: list[EngineStep] = []
        t = 0.0
        for _ in range(n_decode):
            steps.append(EngineStep(start_ms=t, end_ms=t + decode_ms, phase="decode",
                                    num_decode_seqs=decode_seqs))
            t += decode_ms
        for _ in range(n_prefill):
            steps.append(EngineStep(start_ms=t, end_ms=t + prefill_ms, phase=prefill_phase,
                                    num_prefill_tokens=prefill_tokens))
            t += prefill_ms
        timeline = NsysTimeline(steps=steps, trace_duration_ms=(t if trace_duration else None))
    return DiagnosisInput(
        model_name="Llama-3-8B",
        gpu_type="H100-SXM",
        inference_engine="vllm",
        nsys_timeline=timeline,
        ttft_ms=(Distribution(mean=300.0, p50=150.0, p99=900.0) if ttft else None),
    )


@pytest.fixture
def rule() -> PrefillDecodeInterferenceRule:
    return PrefillDecodeInterferenceRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_true_positive_chunking_off(self, rule) -> None:
        result = rule.evaluate(_input())  # 32 decode@10ms, 8 prefill@120ms, no mixed
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r08"
        ev = result.evidence
        assert ev["step_tail_ratio"] == pytest.approx(12.0)
        assert ev["interference_fraction"] == pytest.approx(0.688, abs=1e-2)
        assert ev["chunking_active"] is False
        # Budget = prefill rate (68.3 tok/ms) × the ~5ms slack beyond one 10ms
        # decode step, rounded to 128 — sized so a fused step lands within ~1.5×
        # a decode step. See TestBudgetConvergence for why this is the slack, not
        # the whole target step.
        assert ev["recommended_max_num_batched_tokens"] == 384
        assert result.confidence == pytest.approx(0.65)  # uncalibrated cap
        assert "off" in result.fix.lower()
        assert "--max-num-batched-tokens" in result.fix

    def test_2_true_positive_chunking_on_mistuned(self, rule) -> None:
        # mixed steps present (chunking on) but still 120ms → budget too large.
        result = rule.evaluate(_input(prefill_phase="mixed"))
        assert isinstance(result, Diagnosis)
        assert result.evidence["chunking_active"] is True
        assert "already on" in result.fix.lower()
        assert "lower" in result.fix.lower()

    def test_3_well_chunked_abstains(self, rule) -> None:
        # mixed steps bounded near the decode baseline → tail 1.2× < 2× floor.
        result = rule.evaluate(_input(n_decode=20, n_prefill=20, prefill_phase="mixed",
                                       prefill_ms=12.0))
        assert result is Abstention.BELOW_THRESHOLD

    def test_4_no_timeline_insufficient(self, rule) -> None:
        result = rule.evaluate(_input(include_timeline=False))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("nsys_timeline.steps",)

    def test_5_too_few_steps_insufficient(self, rule) -> None:
        result = rule.evaluate(_input(n_decode=6, n_prefill=2))  # 8 < MIN_STEPS
        assert isinstance(result, InsufficientData)
        assert "timeline steps" in result.reason

    def test_6_decode_only_abstains(self, rule) -> None:
        result = rule.evaluate(_input(n_decode=25, n_prefill=0))
        assert result is Abstention.BELOW_THRESHOLD

    def test_7_prefill_only_abstains(self, rule) -> None:
        result = rule.evaluate(_input(n_decode=0, n_prefill=25))
        assert result is Abstention.BELOW_THRESHOLD

    def test_8_too_few_decode_steps_insufficient(self, rule) -> None:
        # 21 steps total (clears MIN_STEPS) but only 3 decode-only → no baseline.
        result = rule.evaluate(_input(n_decode=3, n_prefill=18))
        assert isinstance(result, InsufficientData)
        assert "decode-only steps" in result.reason

    def test_9_spiky_but_low_interference_abstains(self, rule) -> None:
        # tail 2.2× clears the floor, but two short prefills → interference < 0.15.
        result = rule.evaluate(_input(n_decode=38, n_prefill=2, prefill_ms=22.0))
        assert result is Abstention.BELOW_THRESHOLD

    def test_10_no_token_counts_gives_ratio_target(self, rule) -> None:
        result = rule.evaluate(_input(prefill_tokens=None))
        assert isinstance(result, Diagnosis)
        assert result.evidence["recommended_max_num_batched_tokens"] is None
        assert "within ~1.5×" in result.fix or "within ~1.5x" in result.fix
        assert "no per-step token counts" in result.fix


# --------------------------------------------------------------------------- #
# Budget convergence — the honest acceptance criterion for a rule whose OUTPUT
# is a setting. Feed r08's own recommendation back and it must abstain.
# --------------------------------------------------------------------------- #

class TestBudgetConvergence:
    def test_recommended_budget_is_a_one_shot_fixed_point(self, rule) -> None:
        """Apply r08's recommended budget, re-capture, re-run: it must NOT fire.

        r08's output is a `--max-num-batched-tokens` value an operator sets, so
        the rule has to converge in one shot — the earlier arithmetic sized the
        chunk to the whole target step instead of the slack beyond one decode
        step, recommending a budget ~3× too large, so re-running after applying
        it fired r08 again. This pins convergence directly, not the number.
        """
        dx = _input()  # 32 decode@10ms + 8 prefill@120ms @8192 tok → fires
        fired = rule.evaluate(dx)
        assert isinstance(fired, Diagnosis)
        budget = fired.evidence["recommended_max_num_batched_tokens"]
        assert budget is not None

        stats = _TimelineStats(list(dx.nsys_timeline.steps))
        rate = stats.prefill_tokens_per_ms          # measured prefill tok/ms
        decode_ms = stats.decode_baseline_ms

        # Re-capture the same workload with chunked prefill at `budget`: each
        # prefill of P tokens becomes ceil(P / budget) mixed steps, each running
        # its decode work + (chunk / rate) ms of prefill.
        steps = [s for s in dx.nsys_timeline.steps if s.phase == "decode"]
        t = steps[-1].end_ms
        for orig in [s for s in dx.nsys_timeline.steps if s.phase == "prefill"]:
            remaining = orig.num_prefill_tokens
            while remaining > 0:
                chunk = min(budget, remaining)
                dur = decode_ms + chunk / rate
                steps.append(EngineStep(start_ms=t, end_ms=t + dur, phase="mixed",
                                        num_prefill_tokens=chunk, num_decode_seqs=16))
                t += dur
                remaining -= chunk

        refired = rule.evaluate(
            dx.model_copy(update={"nsys_timeline": NsysTimeline(steps=steps)})
        )
        assert not isinstance(refired, Diagnosis), (
            f"r08 still fires after applying its own budget={budget}: "
            f"{getattr(refired, 'evidence', refired)}"
        )


# --------------------------------------------------------------------------- #
# Evidence & confidence
# --------------------------------------------------------------------------- #

class TestEvidence:
    def test_signal_strength_tracks_breakdown(self, rule) -> None:
        result = rule.evaluate(_input())
        assert isinstance(result, Diagnosis)
        bd = result.confidence_breakdown
        assert bd.signal_strength == pytest.approx(result.evidence["signal_strength"], abs=1e-6)
        assert 0.0 <= bd.data_completeness <= 1.0

    def test_data_completeness_higher_with_tokens(self, rule) -> None:
        with_tokens = rule.evaluate(_input(ttft=True))
        without = rule.evaluate(_input(prefill_tokens=None, trace_duration=False))
        assert isinstance(with_tokens, Diagnosis) and isinstance(without, Diagnosis)
        assert with_tokens.confidence_breakdown.data_completeness > \
            without.confidence_breakdown.data_completeness


# --------------------------------------------------------------------------- #
# Helper-level unit tests
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_percentile_endpoints_and_interp(self) -> None:
        assert _percentile([], 0.95) == 0.0
        assert _percentile([10.0], 0.95) == 10.0
        assert _percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)
        assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)

    def test_timeline_stats_match_hand_computation(self) -> None:
        dx = _input()
        stats = _TimelineStats(list(dx.nsys_timeline.steps))
        assert stats.num_decode == 32 and stats.num_prefill_bearing == 8
        assert stats.decode_baseline_ms == pytest.approx(10.0)
        assert stats.step_tail_ratio == pytest.approx(12.0)
        # excess = 8*(120-10)=880; total = 320+960=1280.
        assert stats.interference_fraction == pytest.approx(880 / 1280)
        budget, target = stats.recommended_budget()
        # rate 68.27 tok/ms × slack (15 − 10 = 5ms) = 341 → round-128 → 384.
        assert budget == 384 and target == pytest.approx(15.0)

    def test_recommended_budget_none_without_tokens(self) -> None:
        dx = _input(prefill_tokens=None)
        stats = _TimelineStats(list(dx.nsys_timeline.steps))
        assert stats.prefill_tokens_per_ms is None
        assert stats.recommended_budget()[0] is None

    def test_thresholds_are_seeds(self) -> None:
        # Guard against accidental threshold drift in review.
        assert TAIL_MIN == 2.0 and INTERFERENCE_MIN == 0.15


# --------------------------------------------------------------------------- #
# Engine integration — the live r02 ↔ r08 corroboration
# --------------------------------------------------------------------------- #

class TestCorroboration:
    def test_r02_and_r08_corroborate_when_both_fire(self) -> None:
        """A colocated dump with BOTH a contention fingerprint and a spiky timeline
        fires r02 and r08; the engine annotates each as corroborated by the other."""
        base = _input()  # spiky timeline → r08
        dx = base.model_copy(update={
            # r02's serving fingerprint (mirrors its own test_1 true positive).
            "prefill": PhaseMetrics(),
            "decode": PhaseMetrics(),
            "vllm_serving": VllmServingMetrics(
                num_preemptions_total=4,
                request_success_total=200,
                prompt_tokens_total=300,
                generation_tokens_total=700,
                # r02 gate G4: contention requires concurrent
                # residency, so r02 now needs this gauge and abstains without it.
                # 32 is the field-measured contended mean.
                num_requests_running=32,
            ),
            "tpot_ms": Distribution(mean=10.0, p50=10.0, p99=24.0),
            "model_params_b": 13.0,
            "num_gpus": 1,
        })
        report = run_diagnosis(dx)
        fired = {d.diagnosis.rule_id: d for d in report.diagnoses}
        assert "r02" in fired and "r08" in fired
        assert fired["r02"].corroborated_by == ("r08",)
        assert fired["r08"].corroborated_by == ("r02",)
        # Boost is off by default → annotation only, scalar unchanged.
        assert fired["r08"].adjusted_confidence == fired["r08"].diagnosis.confidence
