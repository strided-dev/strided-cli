"""
Tests for r03 — KV cache fragmentation.

The acceptance matrix for r03, plus gate,
evidence, and confidence coverage. The abstentions are the point: they stop the
rule from telling a vLLM engineer to "enable PagedAttention" (which they already
run) or to fix fragmentation that costs nothing because the cache has headroom.
"""

from __future__ import annotations

import pytest

from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r03_kv_fragmentation import (
    KvCacheFragmentationRule,
    _UNCALIBRATED_CEILING,
)
from schema import DiagnosisInput, Distribution, VllmServingMetrics


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def _input(
    *,
    frag: float | None = 0.40,
    util: float | None = 0.92,
    engine: str = "vllm",
    block_size: int | None = 16,
    num_blocks_used: int | None = None,
    seq_len_mean: float | None = None,
    preemptions: int | None = None,
    successes: int | None = None,
    queue_ms_mean: float | None = None,
    pressure_window: str | None = None,
) -> DiagnosisInput:
    seq = Distribution(mean=seq_len_mean) if seq_len_mean is not None else None
    serving = None
    if preemptions is not None or successes is not None:
        serving = VllmServingMetrics(
            num_preemptions_total=preemptions,
            request_success_total=successes,
        )
    queue = Distribution(mean=queue_ms_mean) if queue_ms_mean is not None else None
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-70B",
        gpu_type="H100-SXM",
        inference_engine=engine,
        kv_cache_fragmentation=frag,
        kv_cache_util=util,
        kv_block_size=block_size,
        kv_num_blocks_used=num_blocks_used,
        seq_len_distribution=seq,
        vllm_serving=serving,
        queue_time_ms=queue,
        pressure_window=pressure_window,
    )


@pytest.fixture
def rule() -> KvCacheFragmentationRule:
    return KvCacheFragmentationRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_fires_vllm_recommends_block_tuning(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=0.40, util=0.92, engine="vllm"))
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r03"
        # vLLM users already have PagedAttention — the fix must not say "enable" it.
        assert "block-size" in result.fix or "block_size" in result.fix
        assert "enable PagedAttention" not in result.fix
        # Uncalibrated cap holds.
        assert result.confidence <= _UNCALIBRATED_CEILING

    def test_2_fires_unknown_engine_recommends_adopting_paged(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(engine="unknown"))
        assert isinstance(result, Diagnosis)
        assert "PagedAttention" in result.fix

    def test_3_fires_trtllm_recommends_paged_kv_cache(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(engine="trt-llm"))
        assert isinstance(result, Diagnosis)
        assert "paged_kv_cache" in result.fix

    def test_4_boundary_confidence_near_floor(self, rule: KvCacheFragmentationRule) -> None:
        # Just past both thresholds → signal ~0 → confidence ~ floor.
        result = rule.evaluate(_input(frag=0.21, util=0.81))
        assert isinstance(result, Diagnosis)
        assert 0.50 <= result.confidence <= 0.60

    def test_5_abstains_below_frag_threshold(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=0.10, util=0.92))
        assert result is Abstention.BELOW_THRESHOLD

    def test_6_abstains_when_util_low(self, rule: KvCacheFragmentationRule) -> None:
        # Fragmentation present but the cache has headroom → harmless.
        result = rule.evaluate(_input(frag=0.40, util=0.50))
        assert result is Abstention.BELOW_THRESHOLD

    def test_7_abstains_at_exact_threshold(self, rule: KvCacheFragmentationRule) -> None:
        # Firing condition is strict; boundary values must not fire.
        result = rule.evaluate(_input(frag=0.20, util=0.80))
        assert result is Abstention.BELOW_THRESHOLD

    def test_8_insufficient_data_no_fragmentation(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=None))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("kv_cache_fragmentation",)

    def test_9_insufficient_data_no_pressure_signal(self, rule: KvCacheFragmentationRule) -> None:
        # Fragmentation present, but NO way to assess pressure: util, preemptions,
        # and queue time are all absent → the rule cannot judge cost → abstain.
        # (util=None alone is no longer fatal; the other two must be absent too.)
        result = rule.evaluate(
            _input(util=None, preemptions=None, successes=None, queue_ms_mean=None)
        )
        assert result is Abstention.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# Confidence & evidence
# --------------------------------------------------------------------------- #

class TestConfidenceAndEvidence:
    def test_strong_signal_hits_uncalibrated_cap(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=0.95, util=0.99))
        assert isinstance(result, Diagnosis)
        assert result.confidence == pytest.approx(_UNCALIBRATED_CEILING)

    def test_no_decorative_measured_source(self, rule: KvCacheFragmentationRule) -> None:
        # Fragmentation is always the block-size estimate; there is no measured
        # block-accounting recompute, so the rule must NOT stamp a "measured"
        # fragmentation_source. kv_num_blocks_used still counts toward
        # data_completeness, but it may not masquerade as a measured value.
        result = rule.evaluate(_input(num_blocks_used=1000))
        assert isinstance(result, Diagnosis)
        assert "fragmentation_source" not in result.evidence

    def test_evidence_populated(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=0.40, util=0.92, block_size=16))
        assert isinstance(result, Diagnosis)
        ev = result.evidence
        assert ev["kv_cache_fragmentation"] == pytest.approx(0.40)
        assert ev["kv_cache_util"] == pytest.approx(0.92)
        assert ev["inference_engine"] == "vllm"
        assert ev["kv_block_size"] == 16

    def test_signal_strength_and_data_completeness_in_range(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(num_blocks_used=1000, seq_len_mean=128.0))
        assert isinstance(result, Diagnosis)
        bd = result.confidence_breakdown
        assert bd.signal_strength == pytest.approx(result.evidence["signal_strength"], abs=0.02)
        # All four corroborators present → completeness 1.0.
        assert bd.data_completeness == pytest.approx(1.0)

    def test_data_completeness_drops_with_fewer_corroborators(self, rule: KvCacheFragmentationRule) -> None:
        # No block_size, unknown engine, no block accounting, no seq-len dist.
        result = rule.evaluate(
            _input(engine="unknown", block_size=None, num_blocks_used=None, seq_len_mean=None)
        )
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == pytest.approx(0.0)
        # Cause must omit the block-size clause when block size is absent.
        assert "block size" not in result.cause.lower()

    def test_cause_includes_block_size_when_present(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(block_size=32))
        assert isinstance(result, Diagnosis)
        assert "32" in result.cause


# --------------------------------------------------------------------------- #
# Pressure corroboration — the real-GPU finding (2026-06-12)
#
# The instantaneous kv_cache_usage gauge sawtooths, so a single customer scrape
# routinely reads low even on a pressured cache. These tests pin the fix: the
# cumulative preemption rate corroborates pressure when the gauge is in its
# trough, WITHOUT firing on a genuinely idle cache.
# --------------------------------------------------------------------------- #

class TestPreemptionPressure:
    def test_fires_via_preemptions_when_util_in_trough(self, rule: KvCacheFragmentationRule) -> None:
        # Falsified case: frag real, gauge low (trough), but cache provably hit
        # capacity (80% preemption rate). Old hard util gate abstained here.
        result = rule.evaluate(_input(frag=0.40, util=0.60, preemptions=80, successes=100))
        assert isinstance(result, Diagnosis)
        assert result.evidence["pressure_source"] == "preemptions"
        assert result.evidence["preemption_rate"] == pytest.approx(0.80)

    def test_fires_via_preemptions_when_util_absent(self, rule: KvCacheFragmentationRule) -> None:
        # util None is no longer fatal if preemptions establish pressure.
        result = rule.evaluate(_input(frag=0.40, util=None, preemptions=80, successes=100))
        assert isinstance(result, Diagnosis)
        assert result.evidence["pressure_source"] == "preemptions"

    def test_abstains_low_util_and_preemptions_below_mild(self, rule: KvCacheFragmentationRule) -> None:
        # Trough gauge AND a sub-1% preemption rate = no corroborated pressure.
        result = rule.evaluate(_input(frag=0.40, util=0.60, preemptions=0, successes=1000))
        assert result is Abstention.BELOW_THRESHOLD

    def test_no_false_alarm_low_frag_despite_preemptions(self, rule: KvCacheFragmentationRule) -> None:
        # Pressure is real but fragmentation is not — fragmentation stays the
        # primary gate, so the rule must stay silent (the P=30 property).
        result = rule.evaluate(_input(frag=0.06, util=0.95, preemptions=80, successes=100))
        assert result is Abstention.BELOW_THRESHOLD

    def test_pressure_source_lists_all_active_signals(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=0.40, util=0.92, preemptions=80, successes=100))
        assert isinstance(result, Diagnosis)
        src = result.evidence["pressure_source"]
        assert "utilization" in src and "preemptions" in src


# --------------------------------------------------------------------------- #
# Queue-time pressure — the V1 correction (real GPU, 2026-06-13)
#
# vLLM V1 queues instead of preempting: under genuine capacity pressure
# num_preemptions stays 0 and the util gauge sits in its trough, but the
# cumulative request_queue_time_seconds histogram has a nonzero mean. These pin
# that queue time fires the rule where preemptions and util both go silent.
# --------------------------------------------------------------------------- #

class TestQueueTimePressure:
    def test_fires_via_queue_time_when_preempt_zero_and_util_trough(self, rule: KvCacheFragmentationRule) -> None:
        # The exact falsified shape: util 0.60 (trough), no preemptions, but a
        # 21ms mean queue time — the cache is provably at capacity.
        result = rule.evaluate(_input(frag=0.40, util=0.60, preemptions=0, successes=1000, queue_ms_mean=21.0))
        assert isinstance(result, Diagnosis)
        assert "queue" in result.evidence["pressure_source"]
        assert result.evidence["queue_time_ms"] == pytest.approx(21.0)

    def test_abstains_when_queue_time_below_min(self, rule: KvCacheFragmentationRule) -> None:
        # Trough util, no preemptions, negligible queue (5ms < 10ms) → no pressure.
        result = rule.evaluate(_input(frag=0.40, util=0.60, preemptions=0, successes=1000, queue_ms_mean=5.0))
        assert result is Abstention.BELOW_THRESHOLD

    def test_no_false_alarm_low_frag_despite_queue(self, rule: KvCacheFragmentationRule) -> None:
        result = rule.evaluate(_input(frag=0.06, util=0.95, queue_ms_mean=50.0))
        assert result is Abstention.BELOW_THRESHOLD


# --------------------------------------------------------------------------- #
# Pressure window — single-scrape lifetime vs two-scrape delta
#
# A cumulative metric (queue time, preemptions) survives one scrape but its mean
# is a LIFETIME average that dilutes under fresh pressure. The parser sets
# pressure_window="delta" only when two scrapes were differenced to the current
# window; r03 keeps lifetime-only firings capped and records which path it used.
# --------------------------------------------------------------------------- #

class TestPressureWindow:
    def test_default_is_lifetime(self, rule: KvCacheFragmentationRule) -> None:
        # A hand-built / single-scrape input carries no window provenance and is
        # treated as lifetime.
        result = rule.evaluate(_input(frag=0.40, util=0.92))
        assert isinstance(result, Diagnosis)
        assert result.evidence["pressure_window"] == "lifetime"

    def test_delta_window_flagged(self, rule: KvCacheFragmentationRule) -> None:
        # Two-scrape window: the falsified shape (util trough + queue pressure)
        # fires and is flagged as a current-window reading.
        result = rule.evaluate(
            _input(frag=0.40, util=0.60, preemptions=0, successes=1000,
                   queue_ms_mean=21.0, pressure_window="delta")
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["pressure_window"] == "delta"
        assert "queue" in result.evidence["pressure_source"]

    def test_lifetime_pressure_stays_capped(self, rule: KvCacheFragmentationRule) -> None:
        # Even a maximal signal stays under the cap when pressure is a lifetime
        # proxy — the conservative guard that survives flipping the calibration
        # flag (here it coincides with the uncalibrated cap).
        result = rule.evaluate(
            _input(frag=0.95, util=0.99, queue_ms_mean=500.0, pressure_window="lifetime")
        )
        assert isinstance(result, Diagnosis)
        assert result.confidence <= _UNCALIBRATED_CEILING
        assert "lifetime" in result.confidence_breakdown.notes
