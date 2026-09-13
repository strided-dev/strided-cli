"""Schema 1.5.0 contract hardening: no non-finite float crosses the boundary.

Comparison-based validators are blind to NaN (every comparison is False), so
before 1.5.0 `Distribution(mean=nan)` passed the `v < 0` check and adversarial
testing demonstrated a fired r02 diagnosis carrying `tpot_tail_ratio: nan` into
the final report. These tests pin the fix: NaN/Inf are rejected at construction
on every schema model, the finite clamp still clamps, and the two poison inputs
can no longer be built.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schema import DiagnosisInput
from schema.diagnosis_input import (
    Distribution,
    LayerMetrics,
    PhaseMetrics,
    ThroughputSample,
    VllmServingMetrics,
)

NAN = float("nan")
INF = float("inf")
_BASE = dict(model_name="m", gpu_type="H100-SXM", inference_engine="vllm")


@pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
class TestRejectedAtConstruction:
    def test_distribution_mean(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            Distribution(mean=bad)

    def test_distribution_percentiles_previously_unguarded(self, bad: float) -> None:
        # The first poison input's entry point: p99 had no validator at all before 1.5.0.
        with pytest.raises(ValidationError):
            Distribution(mean=1.0, p99=bad)

    def test_top_level_scalars(self, bad: float) -> None:
        for field in ("kv_cache_util", "kv_cache_fragmentation", "nccl_time_pct",
                      "model_params_b", "request_throughput"):
            with pytest.raises(ValidationError):
                DiagnosisInput(**_BASE, **{field: bad})

    def test_tp_rank_lists(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            DiagnosisInput(**_BASE, tp_rank_sm_clocks=[1980.0, bad])

    def test_throughput_sample_unconstrained_field(self, bad: float) -> None:
        # gpu_temp_c carried no ge= bound, so nothing rejected NaN before.
        with pytest.raises(ValidationError):
            ThroughputSample(t=0.0, token_throughput_gen=1.0, gpu_temp_c=bad)

    def test_layer_metrics(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            LayerMetrics(layer_name="k", achieved_flops=bad)

    def test_phase_metrics_utilisation_not_clamped_to_one(self, bad: float) -> None:
        # Pre-1.5.0, min/max ordering clamped NaN to 1.0 — garbage read as full
        # utilisation (r01 fired at 0.90 confidence on it). Now it rejects.
        with pytest.raises(ValidationError):
            PhaseMetrics(hbm_bandwidth_util=bad)


class TestFiniteBehaviorUnchanged:
    def test_clamp_still_clamps_finite_overshoot(self) -> None:
        pm = PhaseMetrics(hbm_bandwidth_util=1.01, sm_occupancy=-0.02)
        assert pm.hbm_bandwidth_util == 1.0
        assert pm.sm_occupancy == 0.0

    def test_ordinary_construction_unaffected(self) -> None:
        dx = DiagnosisInput(
            **_BASE,
            kv_cache_util=0.93,
            tpot_ms=Distribution(mean=12.0, p50=10.0, p99=45.0),
            vllm_serving=VllmServingMetrics(num_preemptions_total=3),
        )
        assert dx.tpot_ms.p99 == 45.0


class TestHarnessFindingsClosed:
    """The poison inputs can no longer be constructed at all."""

    def test_f1_r02_nan_tail_input_unbuildable(self) -> None:
        with pytest.raises(ValidationError):
            DiagnosisInput(**_BASE, tpot_ms=Distribution(mean=12.0, p50=10.0, p99=NAN))

    def test_f2_r03_inf_queue_input_unbuildable(self) -> None:
        with pytest.raises(ValidationError):
            DiagnosisInput(**_BASE, kv_cache_fragmentation=0.55,
                           queue_time_ms=Distribution(mean=INF))
