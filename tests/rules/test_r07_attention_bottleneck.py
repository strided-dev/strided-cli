"""
Tests for r07 — attention bottleneck (unfused attention path).

The acceptance matrix for r07 plus
confidence and evidence coverage. The abstentions are the point: the rule must
stay silent when fused attention kernels are already running (the advice is
taken — a fused-dominant trace is r09's future long-context scope), when a
healthy paged decode merely reads memory-bound, and when the softmax present is
just the tiny sampling/logits pass every fused trace carries.
"""

from __future__ import annotations

import pytest

from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r07_attention_bottleneck import (
    AttentionBottleneckRule,
    ATTN_DOMINANT,
    FUSED_MINOR,
    MIN_KERNELS,
    SOFTMAX_SHARE_FIRE,
    _UNCALIBRATED_CEILING,
)
from schema import DiagnosisInput, LayerMetrics


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _layer(
    name: str,
    ms: float | None,
    *,
    layer_type: str | None = None,
    occ: float | None = None,
    hbm: float | None = None,
    roofline: str | None = None,
) -> LayerMetrics:
    """One kernel row. layer_type defaults to what the Nsight parser would tag
    (attention/mlp/softmax/... by name) via explicit values in the tests, so the
    tests read as traces, not as classifier exercises."""
    return LayerMetrics(
        layer_name=name,
        layer_type=layer_type,
        duration_ms=ms,
        sm_occupancy=occ,
        hbm_bandwidth_util=hbm,
        roofline_position=roofline,
    )


def _gemms(total_ms: float, n: int = 4) -> list[LayerMetrics]:
    """N generic GEMM kernels (the way naive attention's Q.K^T / P.V and the
    MLP both appear) summing to total_ms."""
    return [
        _layer(f"ampere_fp16_s16816gemm_{i}", total_ms / n, layer_type="mlp")
        for i in range(n)
    ]


def _input(layers: list[LayerMetrics] | None) -> DiagnosisInput:
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-8B",
        gpu_type="H100-SXM",
        inference_engine="unknown",   # Nsight dumps do not identify the engine
        layers=layers,
    )


@pytest.fixture
def rule() -> AttentionBottleneckRule:
    return AttentionBottleneckRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_fires_on_softmax_signature(self, rule: AttentionBottleneckRule) -> None:
        # Naive path: GEMMs hold most of the time (attention's matmuls hiding
        # among them by name), standalone softmax at 15%, no fused kernels.
        layers = _gemms(70.0) + [
            _layer("softmax_warp_forward", 15.0, layer_type="softmax"),
            _layer("rmsnorm_kernel", 15.0, layer_type="norm"),
        ]
        result = rule.evaluate(_input(layers))
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r07"
        assert result.evidence["firing_leg"] == "softmax_signature"
        assert result.confidence <= _UNCALIBRATED_CEILING
        assert "unfused" in result.cause

    def test_2_fires_on_unfused_attention_dominance(self, rule: AttentionBottleneckRule) -> None:
        # A custom/legacy attention kernel (attention-named, not fused-named)
        # holholding half the step; softmax minor.
        layers = [
            _layer("my_attention_score_kernel", 50.0, layer_type="attention"),
            _layer("softmax_warp_forward", 2.0, layer_type="softmax"),
        ] + _gemms(48.0)
        result = rule.evaluate(_input(layers))
        assert isinstance(result, Diagnosis)
        assert result.evidence["firing_leg"] == "unfused_attention_dominant"
        assert "my_attention_score_kernel" in result.cause

    def test_3_abstains_when_flash_already_dominant(self, rule: AttentionBottleneckRule) -> None:
        # FlashAttention running and dominating = long-context cost, r09's
        # future territory. "Enable FlashAttention" would be wrong advice.
        layers = [
            _layer("flash_fwd_kernel", 55.0, layer_type="attention"),
        ] + _gemms(45.0)
        assert rule.evaluate(_input(layers)) is Abstention.BELOW_THRESHOLD

    def test_4_abstains_on_sampling_softmax_in_fused_trace(self, rule: AttentionBottleneckRule) -> None:
        # Every healthy fused trace has a tiny logits/sampling softmax; the
        # share bar (not presence) keeps it silent.
        layers = [
            _layer("flash_fwd_splitkv_kernel", 40.0, layer_type="attention"),
            _layer("softmax_warp_forward", 3.0, layer_type="softmax"),
        ] + _gemms(57.0)
        assert rule.evaluate(_input(layers)) is Abstention.BELOW_THRESHOLD

    def test_5_abstains_on_healthy_memory_bound_paged_decode(self, rule: AttentionBottleneckRule) -> None:
        # Paged decode is inherently memory-bound (Kwon 2023) — roofline must
        # never gate, and paged_attention_* counts as fused.
        layers = [
            _layer("paged_attention_v1_kernel", 35.0, layer_type="attention",
                   occ=0.25, hbm=0.9, roofline="memory_bound"),
        ] + _gemms(65.0)
        assert rule.evaluate(_input(layers)) is Abstention.BELOW_THRESHOLD

    def test_6_abstains_when_both_legs_below_bar(self, rule: AttentionBottleneckRule) -> None:
        # No fused kernels, but softmax 5% (< 8%) and attention-named 20% (< 40%).
        layers = [
            _layer("legacy_attention_kernel", 20.0, layer_type="attention"),
            _layer("softmax_warp_forward", 5.0, layer_type="softmax"),
        ] + _gemms(75.0)
        assert rule.evaluate(_input(layers)) is Abstention.BELOW_THRESHOLD

    def test_7_abstains_on_partial_migration(self, rule: AttentionBottleneckRule) -> None:
        # Fused kernels present (10% > FUSED_MINOR) AND a big softmax (12%):
        # mixed/partially-migrated stack — abstain conservatively (doc Open
        # questions) rather than advise enabling what is already half-enabled.
        layers = [
            _layer("flash_fwd_kernel", 10.0, layer_type="attention"),
            _layer("softmax_warp_forward", 12.0, layer_type="softmax"),
        ] + _gemms(78.0)
        assert rule.evaluate(_input(layers)) is Abstention.BELOW_THRESHOLD

    def test_8_insufficient_data_without_layers(self, rule: AttentionBottleneckRule) -> None:
        result = rule.evaluate(_input(None))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("layers",)

    def test_9_insufficient_data_when_no_durations(self, rule: AttentionBottleneckRule) -> None:
        # Layers present but nothing timed (a metrics-only capture).
        layers = [
            _layer("softmax_warp_forward", None, layer_type="softmax", occ=0.2),
            _layer("gemm_k", None, layer_type="mlp", occ=0.7),
        ]
        result = rule.evaluate(_input(layers))
        assert isinstance(result, InsufficientData)
        assert "timed kernels" in result.reason

    def test_10_insufficient_data_on_tiny_capture(self, rule: AttentionBottleneckRule) -> None:
        # 3 kernels / 0.4 ms total: shares over a handful of launches are noise.
        layers = [
            _layer("softmax_warp_forward", 0.2, layer_type="softmax"),
            _layer("gemm_a", 0.1, layer_type="mlp"),
            _layer("gemm_b", 0.1, layer_type="mlp"),
        ]
        result = rule.evaluate(_input(layers))
        assert isinstance(result, InsufficientData)

    def test_11_fires_on_duration_only_trace_with_reduced_completeness(
        self, rule: AttentionBottleneckRule
    ) -> None:
        # The nsys/torch-profiler fallback: durations only, no occupancy or
        # roofline anywhere. Both legs still work; completeness says how
        # degraded the capture was; corroboration reads "not evaluated" (None).
        layers = _gemms(75.0) + [
            _layer("softmax_warp_forward", 20.0, layer_type="softmax"),
            _layer("rmsnorm_kernel", 5.0, layer_type="norm"),
        ]
        result = rule.evaluate(_input(layers))
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == 0.0
        assert result.evidence["attention_path_memory_bound_frac"] is None
        assert "not evaluated" in result.confidence_breakdown.notes


# --------------------------------------------------------------------------- #
# Confidence & evidence
# --------------------------------------------------------------------------- #

class TestConfidenceAndEvidence:
    def _fire(self, rule: AttentionBottleneckRule, softmax_ms: float,
              roofline: str | None = None) -> Diagnosis:
        layers = _gemms(100.0 - softmax_ms) + [
            _layer("softmax_warp_forward", softmax_ms, layer_type="softmax",
                   occ=0.2 if roofline else None,
                   hbm=0.9 if roofline else None,
                   roofline=roofline),
        ]
        result = rule.evaluate(_input(layers))
        assert isinstance(result, Diagnosis)
        return result

    def test_severe_softmax_hits_uncalibrated_cap(self, rule: AttentionBottleneckRule) -> None:
        result = self._fire(rule, softmax_ms=30.0)   # past SOFTMAX_SHARE_SEVERE
        assert result.confidence == pytest.approx(_UNCALIBRATED_CEILING)
        assert result.confidence_breakdown.signal_strength == pytest.approx(1.0)

    def test_signal_strength_monotone_in_softmax_share(self, rule: AttentionBottleneckRule) -> None:
        gentle = self._fire(rule, softmax_ms=10.0)
        steep = self._fire(rule, softmax_ms=20.0)
        assert (steep.confidence_breakdown.signal_strength
                > gentle.confidence_breakdown.signal_strength)

    def test_memory_bound_corroboration_boosts_vs_no_roofline_twin(
        self, rule: AttentionBottleneckRule
    ) -> None:
        # Same trace with and without roofline data on the softmax kernel.
        # While THRESHOLDS_UNCALIBRATED the 0.65 cap absorbs the +0.05 scalar
        # bump (floor 0.5 + scale 0.8 puts nearly every firing at the cap), so
        # the observable contract is: never *less* confident, the boost named
        # in the notes, and the measured fraction in evidence.
        plain = self._fire(rule, softmax_ms=10.0)
        corroborated = self._fire(rule, softmax_ms=10.0, roofline="memory_bound")
        assert corroborated.confidence >= plain.confidence
        assert "corroborated" in corroborated.confidence_breakdown.notes
        assert "corroborated" not in plain.confidence_breakdown.notes
        assert corroborated.evidence["attention_path_memory_bound_frac"] == pytest.approx(1.0)

    def test_evidence_populated(self, rule: AttentionBottleneckRule) -> None:
        result = self._fire(rule, softmax_ms=15.0)
        ev = result.evidence
        assert ev["softmax_share"] == pytest.approx(0.15)
        assert ev["fused_attention_share"] == 0.0
        assert ev["mlp_share"] == pytest.approx(0.85)
        assert ev["firing_leg"] == "softmax_signature"
        assert ev["total_kernel_time_ms"] == pytest.approx(100.0)
        assert ev["num_kernels"] == 5
        assert ev["top_offender"] == "softmax_warp_forward"
        assert ev["top_offender_ms"] == pytest.approx(15.0)
        assert 0.0 <= ev["signal_strength"] <= 1.0

    def test_fix_names_every_stack(self, rule: AttentionBottleneckRule) -> None:
        # Nsight dumps do not identify the engine, so the fix must enumerate.
        result = self._fire(rule, softmax_ms=15.0)
        for needle in ("VLLM_ATTENTION_BACKEND", "scaled_dot_product_attention",
                       "flash_attention_2"):
            assert needle in result.fix

    def test_uncalibrated_note_present(self, rule: AttentionBottleneckRule) -> None:
        result = self._fire(rule, softmax_ms=15.0)
        assert "uncalibrated" in result.confidence_breakdown.notes

    def test_seed_relationships_hold(self) -> None:
        # The guard bar must sit below both firing bars, or the self-guard
        # could never be outrun by a legitimate firing.
        assert FUSED_MINOR < SOFTMAX_SHARE_FIRE < ATTN_DOMINANT
        assert MIN_KERNELS >= 3
