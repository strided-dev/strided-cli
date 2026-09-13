"""
Tests for r04 — Tensor-parallel rank imbalance.

The acceptance matrix for r04, plus helper-level
coverage of the robust-outlier statistics and a DCGM parser → rule integration
check. The abstentions are the point: balanced clusters, uniform gradients, and
too-few-ranks must NOT produce a confident "one GPU is broken" claim.

r04 reads the RAW per-rank SM clocks (`tp_rank_sm_clocks`) and derives the
slowness ratio (max_clock / clock) itself — the schema carries telemetry, the
rule owns the derivation. Inputs here are therefore clock lists (MHz); the
lowest-clock rank is the slowest.
"""

from __future__ import annotations

import json

import pytest

from parsers.dcgm import parse_dcgm_json
from rules.base import Abstention, Diagnosis
from rules.r04_tp_imbalance import (
    TpRankImbalanceRule,
    _UNCALIBRATED_CEILING,
    compute_imbalance,
    slowness_from_sm_clocks,
)
from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def _input(
    sm_clocks: list[float] | None,
    *,
    tensor_parallel_size: int | None = None,
    num_gpus: int | None = None,
    temps: list[float] | None = None,
) -> DiagnosisInput:
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-70B",
        gpu_type="H100-SXM",
        tp_rank_sm_clocks=sm_clocks,
        tp_rank_temps=temps,
        tensor_parallel_size=tensor_parallel_size,
        num_gpus=num_gpus,
    )


@pytest.fixture
def rule() -> TpRankImbalanceRule:
    return TpRankImbalanceRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_fires_on_single_slow_outlier(self, rule: TpRankImbalanceRule) -> None:
        # 8 ranks, rank 3 down-clocked ~20% below the (tightly-clustered) pack.
        result = rule.evaluate(
            _input([2000, 1980, 2010, 1650, 1990, 2000, 1985, 2005],
                   tensor_parallel_size=8, num_gpus=8)
        )
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r04"
        assert result.evidence["slowest_rank"] == 3
        assert "rank 3" in result.cause
        assert "rank 3" in result.fix
        assert result.confidence <= _UNCALIBRATED_CEILING

    def test_2_perfect_outlier_shows_capped_z(self, rule: TpRankImbalanceRule) -> None:
        # Peers identical → MAD 0 → modified z-score is infinite, displayed capped.
        result = rule.evaluate(_input([2000, 2000, 2000, 1500]))
        assert isinstance(result, Diagnosis)
        assert result.evidence["modified_z_score"] == pytest.approx(99.9)
        # An infinite z-score is rendered as a capped "z-score >NN" marker.
        assert "z-score >" in result.cause

    def test_3_balanced_cluster_abstains(self, rule: TpRankImbalanceRule) -> None:
        # A few-MHz spread is healthy jitter, not a straggler.
        result = rule.evaluate(_input([1980, 1970, 2000, 1985]))
        assert result is Abstention.BELOW_THRESHOLD

    def test_4_uniform_gradient_abstains_on_z_gate(self, rule: TpRankImbalanceRule) -> None:
        # Lag exceeds 10% but no isolated outlier (an even gradient) → z-score gate stops it.
        result = rule.evaluate(_input([2000, 1850, 1700, 1550]))
        assert result is Abstention.BELOW_THRESHOLD

    def test_5_small_lag_abstains_on_lag_gate(self, rule: TpRankImbalanceRule) -> None:
        # ~6% lag: a clean outlier statistically, but not materially slow.
        result = rule.evaluate(_input([2000, 2000, 1887, 2000, 2000, 2000]))
        assert result is Abstention.BELOW_THRESHOLD

    def test_6_none_is_insufficient(self, rule: TpRankImbalanceRule) -> None:
        assert rule.evaluate(_input(None)) is Abstention.INSUFFICIENT_DATA

    def test_7_two_ranks_is_insufficient(self, rule: TpRankImbalanceRule) -> None:
        assert rule.evaluate(_input([2000, 1500])) is Abstention.INSUFFICIENT_DATA

    def test_8_degenerate_values_are_insufficient(self, rule: TpRankImbalanceRule) -> None:
        # A non-positive clock (corrupt scrape) yields no usable slowness signal.
        assert rule.evaluate(_input([0.0, 0.0, 0.0])) is Abstention.INSUFFICIENT_DATA
        assert rule.evaluate(_input([2000, -500, 1900])) is Abstention.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# Confidence & evidence
# --------------------------------------------------------------------------- #

class TestConfidenceAndEvidence:
    def test_evidence_populated(self, rule: TpRankImbalanceRule) -> None:
        clocks = [2000, 1980, 2010, 1650, 1990, 2000, 1985, 2005]
        result = rule.evaluate(_input(clocks, tensor_parallel_size=8, num_gpus=8))
        assert isinstance(result, Diagnosis)
        ev = result.evidence
        assert ev["slowest_rank"] == 3
        assert ev["num_ranks"] == 8
        assert ev["relative_lag"] > 0.10
        # The derived slowness ratio is reported (in-rule, not a schema field) along
        # with the raw clocks it came from.
        assert ev["tp_rank_slowness"] == [round(t, 3) for t in slowness_from_sm_clocks(clocks)]
        assert ev["tp_rank_sm_clocks"] == [round(float(c), 1) for c in clocks]

    def test_signal_strength_tracks_breakdown(self, rule: TpRankImbalanceRule) -> None:
        result = rule.evaluate(_input([2000, 1980, 2010, 1650, 1990, 2000, 1985, 2005]))
        assert isinstance(result, Diagnosis)
        bd = result.confidence_breakdown
        assert bd.signal_strength == pytest.approx(result.evidence["signal_strength"], abs=0.02)
        assert 0.0 <= bd.data_completeness <= 1.0

    def test_data_completeness_full_with_context(self, rule: TpRankImbalanceRule) -> None:
        # >=4 ranks + tp_size + num_gpus + temps → all four corroborators present.
        # (The raw clocks are required to fire, so they are not counted as "context".)
        result = rule.evaluate(
            _input(
                [2250, 2250, 2250, 1800],
                tensor_parallel_size=4,
                num_gpus=4,
                temps=[60, 60, 60, 72],
            )
        )
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == pytest.approx(1.0)

    def test_data_completeness_drops_without_context(self, rule: TpRankImbalanceRule) -> None:
        # Exactly 3 ranks, no tp_size, no num_gpus, no temps → no corroborators.
        result = rule.evaluate(_input([2000, 2000, 1500]))
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Temperature corroboration (booster, never a gate)
# --------------------------------------------------------------------------- #

class TestTemperatureCorroboration:
    # A weak-but-firing outlier (slowness ≈ 1.105, MAD=0) keeps confidence below the
    # uncalibrated 0.65 ceiling, so the +0.05 corroboration bonus is observable.
    # clocks: three peers at 2000, rank 3 at 2000/1.105 ≈ 1810.
    _WEAK = [2000, 2000, 2000, 1810]

    def test_hot_slowest_rank_corroborates(self, rule: TpRankImbalanceRule) -> None:
        result = rule.evaluate(_input(self._WEAK, temps=[60, 60, 60, 72]))
        assert isinstance(result, Diagnosis)
        assert result.evidence["temp_corroborated"] is True
        assert result.evidence["slowest_temp_c"] == pytest.approx(72.0)
        assert "thermal throttling" in result.cause

    def test_corroboration_raises_confidence(self, rule: TpRankImbalanceRule) -> None:
        cool = rule.evaluate(_input(self._WEAK, temps=[60, 60, 60, 61]))
        hot = rule.evaluate(_input(self._WEAK, temps=[60, 60, 60, 72]))
        assert isinstance(cool, Diagnosis) and isinstance(hot, Diagnosis)
        assert cool.evidence["temp_corroborated"] is False
        assert hot.confidence > cool.confidence

    def test_cool_slowest_rank_does_not_corroborate(self, rule: TpRankImbalanceRule) -> None:
        # Slowest (lowest-clock) rank is not the hottest → no thermal corroboration.
        result = rule.evaluate(_input(self._WEAK, temps=[70, 71, 72, 61]))
        assert isinstance(result, Diagnosis)
        assert result.evidence["temp_corroborated"] is False
        assert "thermal throttling" not in result.cause


# --------------------------------------------------------------------------- #
# Helper-level unit tests
# --------------------------------------------------------------------------- #

class TestSlownessFromClocks:
    def test_ratio_inverts_clock(self) -> None:
        # max_clock / clock: the lowest-clock rank gets the largest ratio.
        out = slowness_from_sm_clocks([2000, 2000, 1500, 2000])
        assert out == pytest.approx([1.0, 1.0, 2000 / 1500, 1.0])
        assert out.index(max(out)) == 2

    def test_none_on_empty_or_nonpositive(self) -> None:
        assert slowness_from_sm_clocks(None) is None
        assert slowness_from_sm_clocks([]) is None
        assert slowness_from_sm_clocks([2000, 0, 1900]) is None
        assert slowness_from_sm_clocks([2000, -5, 1900]) is None


class TestComputeImbalance:
    def test_none_below_min_ranks(self) -> None:
        assert compute_imbalance([1.0, 0.7]) is None

    def test_locates_slowest_index(self) -> None:
        stats = compute_imbalance([0.8, 1.0, 0.81, 0.79])
        assert stats is not None
        assert stats.slowest_index == 1

    def test_relative_lag_math(self) -> None:
        stats = compute_imbalance([0.8, 0.8, 0.8, 1.0])
        assert stats is not None
        # median peer 0.8 → (1.0 - 0.8) / 0.8 = 0.25
        assert stats.relative_lag == pytest.approx(0.25)

    def test_perfect_outlier_is_infinite_z(self) -> None:
        stats = compute_imbalance([0.8, 0.8, 0.8, 1.0])
        assert stats is not None
        assert stats.modified_z == float("inf")


# --------------------------------------------------------------------------- #
# DCGM parser → rule integration
# --------------------------------------------------------------------------- #

class TestDcgmIntegration:
    def test_util_only_dcgm_dump_abstains(self, rule: TpRankImbalanceRule) -> None:
        # Util-only dump (no SM clocks): rank 0 is pegged at 99% util, but the parser
        # captures no SM clocks because gpu_util's straggler direction is ambiguous
        # under the NCCL barrier (fast ranks spin-wait high). So r04 has no signal and
        # abstains rather than finger the wrong rank — the resolution of the open
        # util-direction question (see dcgm.py / r04 docstring).
        dump = json.dumps(
            {"DCGM_FI_DEV_GPU_UTIL": {"0": 99, "1": 80, "2": 81, "3": 79}}
        )
        dx = parse_dcgm_json(dump, gpu_type="H100-SXM")
        assert dx.tp_rank_sm_clocks is None
        assert rule.evaluate(dx) is Abstention.INSUFFICIENT_DATA

    def test_balanced_dcgm_dump_abstains(self, rule: TpRankImbalanceRule) -> None:
        # Balanced SM clocks → no outlier → below-threshold abstention.
        dump = json.dumps({
            "DCGM_FI_DEV_SM_CLOCK": {"0": 1980, "1": 1985, "2": 1975, "3": 1982},
            "DCGM_FI_DEV_GPU_UTIL": {"0": 88, "1": 90, "2": 89, "3": 91},
        })
        dx = parse_dcgm_json(dump, gpu_type="H100-SXM")
        assert rule.evaluate(dx) is Abstention.BELOW_THRESHOLD

    def test_clock_straggler_fires_where_util_is_blind(self, rule: TpRankImbalanceRule) -> None:
        # The point of the SM-clock signal: under the all-reduce barrier util is flat
        # (busy-wait masks the laggard), but rank 2 is down-clocked. r04 derives the
        # slowness ratio from the clocks and fingers rank 2 — and temperature corroborates.
        dump = json.dumps({
            "DCGM_FI_DEV_GPU_UTIL": {"0": 99, "1": 99, "2": 99, "3": 99},
            "DCGM_FI_DEV_SM_CLOCK": {"0": 1980, "1": 1980, "2": 1400, "3": 1980},
            "DCGM_FI_DEV_GPU_TEMP": {"0": 60, "1": 61, "2": 78, "3": 60},
        })
        dx = parse_dcgm_json(dump, gpu_type="H100-SXM")
        result = rule.evaluate(dx)
        assert isinstance(result, Diagnosis)
        assert result.evidence["slowest_rank"] == 2
        assert result.evidence["tp_rank_sm_clocks"] == [1980.0, 1980.0, 1400.0, 1980.0]
        assert result.evidence["temp_corroborated"] is True


class TestPersistenceGate:
    """Schema 1.6.0 `tp_rank_history`: a straggler must persist across ticks.

    Field testing measured r04 firing on 16/100 ticks
    against a HEALTHY pack from single snapshots, because an idle-downclocked
    rank and a thermally-throttled one look identical in one sample. These pin
    the gate that separates them — and, equally, pin that the one-shot path is
    left alone (a static dump has no DVFS transient to confuse).
    """

    _STRAGGLER = [1980.0, 1975.0, 1985.0, 1200.0]   # rank 3 clearly slow

    @staticmethod
    def _with_history(clocks_series: list[list[float]]):
        from collect.window import TpRankHistory
        hist = TpRankHistory()
        h = []
        for i, c in enumerate(clocks_series):
            h = hist.update(float(i * 3), c)
        return _input(clocks_series[-1]).model_copy(update={"tp_rank_history": h})

    def test_sustained_straggler_fires(self, rule: TpRankImbalanceRule) -> None:
        # Same rank slow on every tick — a real throttle.
        dx = self._with_history([self._STRAGGLER] * 3)
        result = rule.evaluate(dx)
        assert isinstance(result, Diagnosis)
        assert result.evidence["slowest_rank"] == 3

    def test_transient_outlier_abstains(self, rule: TpRankImbalanceRule) -> None:
        # A DIFFERENT rank dips each tick — the idle-downclock signature that
        # produced 16/100 false positives on real healthy hardware.
        dx = self._with_history([
            [1200.0, 1980.0, 1975.0, 1985.0],   # rank 0 dips
            [1980.0, 1200.0, 1975.0, 1985.0],   # rank 1 dips
            [1980.0, 1975.0, 1985.0, 1200.0],   # rank 3 dips
        ])
        assert rule.evaluate(dx) is Abstention.BELOW_THRESHOLD

    def test_one_shot_path_unaffected(self, rule: TpRankImbalanceRule) -> None:
        # No history (a static dump): single-snapshot behaviour is preserved.
        result = rule.evaluate(_input(self._STRAGGLER))
        assert isinstance(result, Diagnosis)
        assert result.evidence["slowest_rank"] == 3

    def test_short_history_does_not_silence_the_first_tick(
        self, rule: TpRankImbalanceRule
    ) -> None:
        # Below PERSISTENCE_TICKS there is nothing to persist across yet; a live
        # watch must not go blind on startup. Transients die at the 2nd sample.
        dx = self._with_history([self._STRAGGLER])
        assert isinstance(rule.evaluate(dx), Diagnosis)
