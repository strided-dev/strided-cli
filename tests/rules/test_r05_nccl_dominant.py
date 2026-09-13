"""
Tests for r05 — NCCL collective dominates step time.

The acceptance matrix for r05, plus helper-level
coverage of the topology inference and the r04 self-guard, and an Nsight parser →
rule integration check. The abstentions are the point: a healthy collective share,
an inferred multi-node fabric whose NCCL% is inherently higher, and — critically —
a dump where a single straggler (r04's territory) is inflating the local NCCL time
must NOT produce a confident "your network is the problem" claim.
"""

from __future__ import annotations

import pytest

from parsers.nsight import parse_nsight_csv
from rules.base import Abstention, Diagnosis
from rules.r04_tp_imbalance import TpRankImbalanceRule
from rules.r05_nccl_dominant import (
    FIRE_MULTI_NODE,
    FIRE_SINGLE_NODE,
    NcclCollectiveDominantRule,
    _UNCALIBRATED_CEILING,
    infer_topology_scale,
    straggler_present,
)
from schema import DiagnosisInput, TpRankSample


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def _input(
    nccl_time_pct: float | None,
    *,
    tensor_parallel_size: int | None = None,
    num_gpus: int | None = None,
    tp_rank_sm_clocks: list[float] | None = None,
) -> DiagnosisInput:
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-70B",
        gpu_type="H100-SXM",
        nccl_time_pct=nccl_time_pct,
        tensor_parallel_size=tensor_parallel_size,
        num_gpus=num_gpus,
        tp_rank_sm_clocks=tp_rank_sm_clocks,
    )


@pytest.fixture
def rule() -> NcclCollectiveDominantRule:
    return NcclCollectiveDominantRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_fires_single_node(self, rule: NcclCollectiveDominantRule) -> None:
        # TP=8 (one NVLink box), 45% of step in NCCL — well above the single-node bar.
        result = rule.evaluate(_input(0.45, tensor_parallel_size=8, num_gpus=8))
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r05"
        assert result.evidence["inferred_topology"] == "single_node"
        assert result.evidence["threshold_used"] == FIRE_SINGLE_NODE
        assert result.confidence <= _UNCALIBRATED_CEILING
        assert "45%" in result.cause

    def test_2_multi_node_moderate_abstains(self, rule: NcclCollectiveDominantRule) -> None:
        # TP=16 (multi-node): 30% NCCL is below the higher inherent bar.
        result = rule.evaluate(_input(0.30, tensor_parallel_size=16, num_gpus=16))
        assert result is Abstention.BELOW_THRESHOLD

    def test_3_multi_node_high_fires(self, rule: NcclCollectiveDominantRule) -> None:
        result = rule.evaluate(_input(0.60, tensor_parallel_size=16, num_gpus=16))
        assert isinstance(result, Diagnosis)
        assert result.evidence["inferred_topology"] == "multi_node"
        assert result.evidence["threshold_used"] == FIRE_MULTI_NODE

    def test_4_single_node_low_abstains(self, rule: NcclCollectiveDominantRule) -> None:
        result = rule.evaluate(_input(0.10, tensor_parallel_size=8))
        assert result is Abstention.BELOW_THRESHOLD

    def test_5_straggler_defers_to_r04(self, rule: NcclCollectiveDominantRule) -> None:
        # High NCCL fraction, but one rank is a clear outlier — the dominance is most
        # likely that rank busy-waiting at the barrier. Defer; r04 owns this.
        result = rule.evaluate(
            _input(0.60, tensor_parallel_size=8, num_gpus=8,
                   tp_rank_sm_clocks=[2000, 2000, 2000, 1500])  # rank 3 down-clocked
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_6_none_is_insufficient(self, rule: NcclCollectiveDominantRule) -> None:
        assert rule.evaluate(_input(None)) is Abstention.INSUFFICIENT_DATA

    def test_7_unknown_scale_fires_with_lower_confidence(
        self, rule: NcclCollectiveDominantRule
    ) -> None:
        # No TP/num_gpus → scale unknown → conservative bar + confidence penalty.
        unknown = rule.evaluate(_input(0.60))
        single = rule.evaluate(_input(0.60, tensor_parallel_size=8))
        assert isinstance(unknown, Diagnosis)
        assert isinstance(single, Diagnosis)
        assert unknown.evidence["inferred_topology"] == "unknown"
        assert unknown.confidence < single.confidence

    def test_balanced_rank_data_still_fires(self, rule: NcclCollectiveDominantRule) -> None:
        # Rank data present but balanced → self-guard does not trip.
        result = rule.evaluate(
            _input(0.45, tensor_parallel_size=8, num_gpus=8,
                   tp_rank_sm_clocks=[1990, 2000, 1980, 1990])
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["straggler_suspected"] is False

    def test_straggler_suspected_none_when_no_clocks(
        self, rule: NcclCollectiveDominantRule
    ) -> None:
        # No rank data: the self-guard could not run, so straggler_suspected is None
        # (not evaluated) — distinct from False (ran and ruled a straggler out).
        result = rule.evaluate(_input(0.60, tensor_parallel_size=8, num_gpus=8))
        assert isinstance(result, Diagnosis)
        assert result.evidence["straggler_suspected"] is None
        assert "deferral not evaluated" in result.confidence_breakdown.notes


# --------------------------------------------------------------------------- #
# Confidence & evidence
# --------------------------------------------------------------------------- #

class TestConfidenceAndEvidence:
    def test_evidence_populated(self, rule: NcclCollectiveDominantRule) -> None:
        result = rule.evaluate(_input(0.45, tensor_parallel_size=8, num_gpus=8))
        assert isinstance(result, Diagnosis)
        ev = result.evidence
        assert ev["nccl_time_pct"] == pytest.approx(0.45)
        assert ev["tensor_parallel_size"] == 8
        assert ev["num_gpus"] == 8
        assert "signal_strength" in ev

    def test_signal_strength_tracks_breakdown(self, rule: NcclCollectiveDominantRule) -> None:
        result = rule.evaluate(_input(0.60, tensor_parallel_size=8))
        assert isinstance(result, Diagnosis)
        bd = result.confidence_breakdown
        assert bd.signal_strength == pytest.approx(result.evidence["signal_strength"], abs=0.02)
        assert 0.0 <= bd.data_completeness <= 1.0

    def test_data_completeness_full_with_context(
        self, rule: NcclCollectiveDominantRule
    ) -> None:
        result = rule.evaluate(
            _input(0.45, tensor_parallel_size=8, num_gpus=8,
                   tp_rank_sm_clocks=[1990, 2000, 1980, 1990])
        )
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == pytest.approx(1.0)

    def test_data_completeness_drops_without_context(
        self, rule: NcclCollectiveDominantRule
    ) -> None:
        result = rule.evaluate(_input(0.60))  # nothing but the required field
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Helper-level unit tests
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_infer_scale_prefers_tp_size(self) -> None:
        dx = _input(0.5, tensor_parallel_size=8, num_gpus=16)
        assert infer_topology_scale(dx) == "single_node"  # TP wins over num_gpus

    def test_infer_scale_falls_back_to_num_gpus(self) -> None:
        assert infer_topology_scale(_input(0.5, num_gpus=16)) == "multi_node"

    def test_infer_scale_unknown(self) -> None:
        assert infer_topology_scale(_input(0.5)) == "unknown"

    def test_straggler_present_true(self) -> None:
        # straggler_present reads raw SM clocks (rank 3 down-clocked → slow outlier).
        assert straggler_present([2000, 2000, 2000, 1500]) is True

    def test_straggler_absent_when_balanced(self) -> None:
        assert straggler_present([1990, 2000, 1980, 1990]) is False

    def test_straggler_absent_below_min_ranks(self) -> None:
        assert straggler_present([1500, 2000]) is False

    def test_straggler_absent_when_none_or_corrupt(self) -> None:
        assert straggler_present(None) is False
        assert straggler_present([2000, 0, 1900]) is False  # non-positive clock


class TestSelfGuardHonoursPersistence:
    """The self-guard must not defer to a straggler r04 would itself reject.

    r04's persistence gate (schema 1.6.0) makes it abstain when the outlier rank
    changes tick to tick — a DVFS transient, not a throttle. If r05 kept deferring
    on the snapshot alone it would step aside for a diagnosis that no longer
    exists, silencing a real interconnect finding. Both rules read the same
    `outlier_persists` helper from `rules/_stats`, so neither imports the other.
    """

    _SUSTAINED = [2000.0, 2000.0, 2000.0, 1500.0]   # rank 3 slow, every tick

    @staticmethod
    def _history(series: list[list[float]]) -> list[TpRankSample]:
        return [TpRankSample(t=float(i * 3), sm_clocks=c) for i, c in enumerate(series)]

    def test_defers_to_a_sustained_straggler(self) -> None:
        hist = self._history([self._SUSTAINED] * 3)
        assert straggler_present(self._SUSTAINED, hist) is True

    def test_does_not_defer_to_a_transient(self) -> None:
        # A different rank dips each tick — nothing for r04 to blame.
        hist = self._history([
            [1500.0, 2000.0, 2000.0, 2000.0],
            [2000.0, 1500.0, 2000.0, 2000.0],
            [2000.0, 2000.0, 2000.0, 1500.0],
        ])
        assert straggler_present([2000.0, 2000.0, 2000.0, 1500.0], hist) is False

    def test_no_history_keeps_snapshot_behaviour(self) -> None:
        # The dump path carries no history; the guard is unchanged there.
        assert straggler_present(self._SUSTAINED, None) is True

    def test_guard_stays_broader_than_r04_firing(self) -> None:
        # A lag that clears r05's guard but not r04's z-score bar must still be
        # deferred to — persistence narrows *when*, never *how strict*.
        clocks = [2000.0, 1900.0, 1800.0, 1700.0]   # graded spread, no isolated outlier
        hist = self._history([clocks] * 3)
        assert straggler_present(clocks, hist) is True
        assert TpRankImbalanceRule().evaluate(
            _input(0.70, tensor_parallel_size=4, tp_rank_sm_clocks=clocks).model_copy(
                update={"tp_rank_history": hist}
            )
        ) is Abstention.BELOW_THRESHOLD

    def test_rule_speaks_when_the_straggler_was_only_a_transient(self) -> None:
        # End to end: a genuinely interconnect-bound run whose current tick catches
        # one DVFS dip must still produce the NCCL diagnosis.
        hist = self._history([
            [1500.0, 2000.0, 2000.0, 2000.0],
            [2000.0, 1500.0, 2000.0, 2000.0],
        ])
        dx = _input(
            0.70, tensor_parallel_size=4, tp_rank_sm_clocks=[2000.0, 1500.0, 2000.0, 2000.0]
        ).model_copy(update={"tp_rank_history": hist})
        assert isinstance(NcclCollectiveDominantRule().evaluate(dx), Diagnosis)


# --------------------------------------------------------------------------- #
# Nsight parser → rule integration
# --------------------------------------------------------------------------- #

class TestNsightIntegration:
    def _csv(self, kernels: dict[str, float]) -> str:
        # raw-page format: Kernel Name, Metric Name, Metric Value (duration in ns).
        rows = ["Kernel Name,Metric Name,Metric Value"]
        for name, dur_ns in kernels.items():
            rows.append(f"{name},gpu__time_duration.sum,{dur_ns}")
        return "\n".join(rows)

    def test_nccl_dominant_dump_fires(self, rule: NcclCollectiveDominantRule) -> None:
        # 6 ms in an all-reduce kernel vs 4 ms of GEMM → nccl_time_pct = 0.6.
        # parse_nsight_csv sets no TP/num_gpus → unknown scale, bar 0.45 → fires.
        csv = self._csv(
            {"ncclAllReduceRingLLKernel": 6_000_000, "ampere_sgemm_128x128": 4_000_000}
        )
        dx = parse_nsight_csv(csv, gpu_type="H100-SXM")
        assert dx.nccl_time_pct == pytest.approx(0.6)

        result = rule.evaluate(dx)
        assert isinstance(result, Diagnosis)
        assert result.evidence["inferred_topology"] == "unknown"

    def test_low_nccl_dump_abstains(self, rule: NcclCollectiveDominantRule) -> None:
        # 0.5 ms all-reduce vs 9.5 ms GEMM → nccl_time_pct = 0.05 → below every bar.
        csv = self._csv(
            {"ncclAllReduceRingLLKernel": 500_000, "ampere_sgemm_128x128": 9_500_000}
        )
        dx = parse_nsight_csv(csv, gpu_type="H100-SXM")
        assert dx.nccl_time_pct == pytest.approx(0.05)
        assert rule.evaluate(dx) is Abstention.BELOW_THRESHOLD
