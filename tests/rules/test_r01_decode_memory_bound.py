"""
Tests for r01 — Decode memory-bound at low batch.

Three halves, which is one more than there were:

1. The acceptance matrix for r01 — clear
   signal, boundary, below-threshold on each leg, insufficient data, evidence.
   These run on the *collapsed* (DCGM-shaped) input and therefore also pin the
   fallback path.

2. **The scope-dilution finding regression**, built from the real field numbers of a capture (A100-PCIE-40GB, vLLM 0.10.2, Qwen2.5-7B, batch-1 decode). The
   shipped rule read a duration-weighted mean over every kernel — 0.595 against
   a 0.80 gate — and stayed silent on the textbook memory-bound case, while the
   kernel that actually carried the pathology sat at 0.795. That is a FALSE
   NEGATIVE, and `TestFieldRegressionFr01` pins it: the scoped read recovers
   0.795, the flat read that hid it still measures 0.595, and scoping is shown
   to flip the verdict on that dilution shape.

   It also pins the part this fix does NOT fix. 0.795 does not clear a strict
   0.80 gate, so the field capture itself remains silent even scoped — the locus
   fix removes 0.200 of the 0.205 shortfall and the last 0.005 belongs to the
   threshold, whose re-derivation is blocked on a paired hardware run. See
   `test_locus_fix_closes_the_gap_but_the_gate_still_blocks`, which is written to
   flip to a firing assertion the day that run happens.

3. **The occupancy leg as a declared FP guard** (`TestOccupancyFpGuard`) — the
   planned **third arm**. the scope-dilution finding's two arms measured occupancy at 0.124
   and 0.170, both far under the 0.30 gate, so the leg passed in both and
   separated nothing. That makes it *untested*, not useless: the mode it guards
   is high bandwidth **with** high occupancy — a saturated compute-bound kernel
   — and neither arm was high-occupancy. This class supplies the arm the field
   pair could not: a compute-bound, high-occupancy workload on which r01 must
   stay silent, plus the attribution test showing the bandwidth leg alone would
   have fired, so the silence is the guard's doing. It also pins the behavioural
   half of the reclassification: `signal_strength` is the bandwidth leg alone,
   and the guard no longer grades it.

4. **The guard's own fail-open** (`TestOccupancyFailOpen`). Scoping the bandwidth leg while the guard kept reading collapsed
   occupancy re-created the scope-dilution finding's dilution one level down, inside the guard: on
   a mixed capture idle kernels drag the aggregate under 0.30 while the kernel
   r01 actually read is busy, so the guard fails open on the one mode it exists
   to catch. This class pins that hole shut, and pins the sharper claim that
   makes it a defect rather than a nuance — **scoping one leg and leaving the
   other collapsed is worse than scoping neither.** On the fixture below,
   scoping nothing is silent, scoping only bandwidth FIRES on a healthily
   saturated kernel, and scoping both is silent. The middle state is the only
   wrong answer of the three.
"""

from __future__ import annotations

import pytest

from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r01_decode_memory_bound import (
    DecodeMemoryBoundRule,
    EV_OCC_COLLAPSED,
    EV_OCC_SCOPED,
    FP_GUARDS,
    KERNEL_SCOPE,
    LOCUS_COLLAPSED,
    LOCUS_SCOPED,
    OCCUPANCY_FP_GUARD,
    OCCUPANCY_LEG_ROLE,
    _HBM_THRESHOLD,
    _SM_THRESHOLD,
    _UNCALIBRATED_CEILING,
    apply_kernel_scope,
    flat_hbm_mean,
)
from schema import DiagnosisInput, LayerMetrics, PhaseMetrics


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _input(
    hbm_util: float | None = None,
    sm_occ: float | None = None,
    decode_present: bool = True,
    batch_size: int | None = None,
    layers: list[LayerMetrics] | None = None,
) -> DiagnosisInput:
    decode = (
        PhaseMetrics(hbm_bandwidth_util=hbm_util, sm_occupancy=sm_occ)
        if decode_present
        else None
    )
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-70B",
        gpu_type="H100-SXM",
        decode=decode,
        batch_size=batch_size,
        layers=layers,
    )


def _kernel(
    name: str,
    ms: float,
    hbm: float | None = None,
    occ: float | None = None,
) -> LayerMetrics:
    return LayerMetrics(
        layer_name=name, duration_ms=ms, hbm_bandwidth_util=hbm, sm_occupancy=occ
    )


def _flat_occ_mean(layers: list[LayerMetrics]) -> float:
    """The collapsed `decode.sm_occupancy` the Nsight parser would derive.

    Mirrors `parsers/nsight.py::_aggregate_phase` for occupancy, so a fixture's
    `decode` block is what the parser would actually hand r01 rather than a
    hand-picked number. This is the read the guard used to gate on.
    """
    num = den = 0.0
    for layer in layers:
        ms = layer.duration_ms
        if layer.sm_occupancy is not None and ms is not None and ms > 0:
            num += layer.sm_occupancy * ms
            den += ms
    return num / den


def _capture(
    layers: list[LayerMetrics], batch_size: int | None = None
) -> DiagnosisInput:
    """A capture shaped as `parsers/nsight.py` hands it to r01.

    The `decode` block is the collapsed aggregate DERIVED from `layers` — both
    legs — so a fixture can never quietly pair per-kernel records with a
    hand-picked phase scalar the parser would never have produced. Every guard fail-open
    test below turns on the collapsed occupancy being the honest duration-weighted
    mean of the very kernels it is being compared against.
    """
    return _input(
        hbm_util=flat_hbm_mean(layers),
        sm_occ=_flat_occ_mean(layers),
        batch_size=batch_size,
        layers=layers,
    )


# --------------------------------------------------------------------------- #
# The measured shape of a field capture, induce arm (batch-1 decode).
#
#   subset                          share of time   hbm_util
#   dominant decode GEMM alone          43.6%        0.795
#   all GEMM kernels                    82.6%        0.720
#   everything (what r01 read)         100.0%        0.595
#
# Reconstructed on a 1000 ms base: 436 ms of dominant GEMM at 0.795, 390 ms of
# other GEMMs at 0.6362 (the value that puts the GEMM subset at 0.720), and
# 174 ms of elementwise/norm/rotary work at ~0.002 — the 17.4% of step time that
# moves essentially no bandwidth. The flat duration-weighted mean of that is
# 0.595, reproducing the field reading to three decimals, which is the point:
# duration weighting is already in force and does not save the signal.
# --------------------------------------------------------------------------- #

_DOMINANT_KERNEL = "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_nn"
_FIELD_DOMINANT_HBM = 0.795
_FIELD_FLAT_HBM = 0.595
_FIELD_AGGREGATE_OCC = 0.124   # collapsed occupancy, both arms far under 0.30


def _field_layers(dominant_hbm: float = _FIELD_DOMINANT_HBM) -> list[LayerMetrics]:
    """The measured kernel *shape* of the induce arm.

    The 43.6 / 39.0 / 17.4 time split and the 0.6362 / 0.002 utilisations are
    the field measurement and never move. `dominant_hbm` defaults to the
    measured 0.795 and is parameterised only so the mechanism tests can ask
    "what if the carrying kernel had been inside the seed's firing band" without
    pretending the answer was measured.
    """
    layers = [
        _kernel(_DOMINANT_KERNEL, 436.0, hbm=dominant_hbm, occ=0.118),
        _kernel("ampere_fp16_s16816gemm_fp16_64x64_qkv", 195.0, hbm=0.6362, occ=0.09),
        _kernel("ampere_fp16_s16816gemm_fp16_64x64_o_proj", 195.0, hbm=0.6362, occ=0.09),
    ]
    # 17.4% of step time at ~0.002 utilisation: the dilution population.
    for name in (
        "rms_norm_kernel",
        "rotary_embedding_kernel",
        "act_and_mul_kernel",
        "fused_add_rms_norm_kernel",
        "reshape_and_cache_flash_kernel",
        "sampling_top_p_kernel",
    ):
        layers.append(_kernel(name, 29.0, hbm=0.002, occ=0.03))
    return layers


def _field_input(
    with_layers: bool = True,
    dominant_hbm: float = _FIELD_DOMINANT_HBM,
) -> DiagnosisInput:
    """The batch-1 decode capture as the Nsight parser hands it to r01.

    `decode` carries the collapsed aggregate the parser derives from `layers`
    (parsers/nsight.py::_aggregate_phase), so the fixture is faithful to the
    real input rather than a rule-shaped convenience: the collapsed HBM number
    is recomputed from the layers exactly as the parser would, and occupancy is
    the measured 0.124.
    """
    layers = _field_layers(dominant_hbm)
    collapsed = flat_hbm_mean(layers)
    return _input(
        hbm_util=collapsed,
        sm_occ=_FIELD_AGGREGATE_OCC,
        batch_size=1,
        layers=layers if with_layers else None,
    )


@pytest.fixture
def rule() -> DecodeMemoryBoundRule:
    return DecodeMemoryBoundRule()


# --------------------------------------------------------------------------- #
# The finding, pinned
# --------------------------------------------------------------------------- #

class TestFieldRegressionFr01:
    """the scope-dilution finding: batch-1 decode on an A100 must not read as compute-bound."""

    def test_flat_read_reproduces_the_false_negative(self) -> None:
        """The number that hid the pathology is still 0.595 — and still under the gate.

        Pinned as a *measurement*, not as accepted behaviour: if a future change
        to the fixture stops reproducing 0.595 the regression below stops
        testing the thing it claims to test.
        """
        flat = flat_hbm_mean(_field_layers())
        assert flat == pytest.approx(_FIELD_FLAT_HBM, abs=0.001)
        assert flat <= _HBM_THRESHOLD

    def test_scoped_read_recovers_the_carrying_kernel(self) -> None:
        scoped = apply_kernel_scope(_field_layers(), KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.hbm_util == pytest.approx(_FIELD_DOMINANT_HBM, abs=0.001)
        assert scoped.kernel_name == _DOMINANT_KERNEL
        assert scoped.kernel_share == pytest.approx(0.436, abs=0.001)
        # The six near-zero-bandwidth kernels are 2.9% of step time each.
        assert scoped.kernels_below_min_share == 6
        assert scoped.kernels_in_scope == 3

    def test_locus_fix_closes_the_gap_but_the_gate_still_blocks(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """**the scope-dilution finding is not fully closed by this fix, and here is the residue.**

        Read the arithmetic, because it is the whole state of the finding:

            flat read (what r01 gated on)          0.595
            gate                                   0.800
            shortfall                              0.205

            scoped read (dominant kernel)          0.795
            shortfall the locus fix removes        0.200   (97.6%)
            shortfall remaining                    0.005   (2.4%)

        Scoping recovers essentially the entire miss — the defect really was
        unscoped aggregation — and the capture *still* does not fire, because
        0.795 does not clear a strict 0.80 and never will. The last 0.005 is the
        threshold's, and re-deriving the threshold needs a paired hardware run
        with an nsys cross-check (ncu serialises kernel replay and may itself
        depress apparent bandwidth). That is deliberately out of this fix's
        scope.

        So this test asserts SILENCE on the field capture. It is not an
        endorsement of that silence: it is the honest record of exactly how far
        the locus fix goes. When a hardware run re-seats the gate, this assertion flips to
        `Diagnosis` and the finding closes — and if anyone re-seats the gate
        without a hardware run, this test is where they will have to argue for it.
        """
        scoped = apply_kernel_scope(_field_layers(), KERNEL_SCOPE)
        flat = flat_hbm_mean(_field_layers())
        assert scoped is not None and flat is not None

        recovered = scoped.hbm_util - flat
        residual = _HBM_THRESHOLD - scoped.hbm_util
        assert recovered == pytest.approx(0.200, abs=0.001)
        assert residual == pytest.approx(0.005, abs=0.001)
        assert recovered / (recovered + residual) > 0.97

        assert rule.evaluate(_field_input()) is Abstention.BELOW_THRESHOLD

    def test_same_capture_without_layers_still_stays_silent(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """Identical `decode` block, no per-kernel records: nothing to scope.

        The pre-fix read path, preserved for the DCGM locus — and the reason
        every diagnosis now labels which locus produced it.
        """
        assert rule.evaluate(_field_input(with_layers=False)) is Abstention.BELOW_THRESHOLD

    def test_scoping_flips_the_verdict_on_the_measured_dilution(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """The locus fix changes verdicts, on the dilution ratio we measured.

        Same 43.6 / 39.0 / 17.4 kernel shape as the field capture, with the
        carrying kernel moved to 0.86 — inside the seed's own documented firing
        band, and NOT a measured value. Everything else is the field's. The
        dilution drags the flat read to 0.62, under the gate: without `layers`
        the rule is silent, with them it fires. That is the false negative the
        fix removes, isolated from the threshold question the field capture
        also raises.
        """
        assert rule.evaluate(
            _field_input(with_layers=False, dominant_hbm=0.86)
        ) is Abstention.BELOW_THRESHOLD

        result = rule.evaluate(_field_input(dominant_hbm=0.86))
        assert isinstance(result, Diagnosis)
        assert result.evidence["hbm_read_locus"] == LOCUS_SCOPED
        assert result.evidence["layers.dominant.hbm_bandwidth_util"] == pytest.approx(0.86)
        assert result.evidence["dominant_kernel"] == _DOMINANT_KERNEL
        assert result.evidence["flat_hbm_mean_all_kernels"] == pytest.approx(
            0.6234, abs=0.001
        )

    def test_diagnosis_publishes_the_gap_that_hid_it(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """A reader must see, from the report alone, that a flat read would miss."""
        result = rule.evaluate(_field_input(dominant_hbm=0.86))
        assert isinstance(result, Diagnosis)
        assert "62%" in result.cause          # the flat mean, spelled out
        assert "scope dilution" in result.cause
        assert _DOMINANT_KERNEL in result.cause

    def test_occupancy_is_read_from_the_same_kernel_as_bandwidth(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """**the guard fail-open.** Both legs describe one kernel, and the evidence says so.

        The dominant kernel's own occupancy (0.118) differs from the phase
        aggregate (0.124). r01 reports the *kernel's*, under a key that names the
        locus, and publishes the aggregate beside it as a labelled contrast — the
        occupancy twin of `flat_hbm_mean_all_kernels`. On this capture the gap is
        small and changes nothing; the fixture in `TestOccupancyFailOpen` is where
        it changes the verdict.
        """
        result = rule.evaluate(_field_input(dominant_hbm=0.86))
        assert isinstance(result, Diagnosis)
        assert result.evidence[EV_OCC_SCOPED] == pytest.approx(0.118)
        assert result.evidence["occupancy_read_locus"] == LOCUS_SCOPED
        # The number the guard used to gate on, published but not read.
        assert result.evidence["collapsed_sm_occupancy"] == pytest.approx(
            _FIELD_AGGREGATE_OCC
        )
        # And the collapsed key is absent, so nobody can mistake one for the other.
        assert EV_OCC_COLLAPSED not in result.evidence

    def test_both_legs_report_the_same_locus_when_layers_are_present(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """One scope, one kernel, one verdict — asserted as a pair, not per leg."""
        result = rule.evaluate(_field_input(dominant_hbm=0.86))
        assert isinstance(result, Diagnosis)
        assert result.evidence["hbm_read_locus"] == LOCUS_SCOPED
        assert result.evidence["occupancy_read_locus"] == LOCUS_SCOPED
        assert result.evidence["read_loci_agree"] is True
        assert result.confidence_breakdown.data_completeness == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The occupancy leg, reclassified: a declared FP guard
# --------------------------------------------------------------------------- #

def _compute_bound_layers(hbm: float = 0.88, occ: float = 0.82) -> list[LayerMetrics]:
    """A healthily saturated compute-bound kernel mix.

    Same *shape* discipline as the field fixture — one dominant GEMM plus
    supporting kernels — but the regime is inverted: the SMs are busy. This is
    NOT a measurement. It is the planned third arm, written synthetically
    now so the guard has a test behind it before real hardware exists, and shaped
    to match what a hardware run must produce for the arm to count.
    """
    layers = [
        _kernel("large_batch_gemm_dominant", 500.0, hbm=hbm, occ=occ),
        _kernel("fused_attention_kernel", 250.0, hbm=hbm * 0.9, occ=occ * 0.95),
        _kernel("act_and_mul_kernel", 150.0, hbm=0.232, occ=0.799),
        _kernel("rms_norm_kernel", 100.0, hbm=0.15, occ=0.60),
    ]
    return layers


class TestOccupancyFpGuard:
    """**Planned third arm** — r01 must stay silent on saturated compute.

    the scope-dilution finding measured occupancy at 0.124 and 0.170. Both are far under the 0.30
    gate, so the leg passed in both arms and contributed no separation at all:
    every bit of discrimination in that pair came from the bandwidth signal.

    The tempting read is that the leg is dead weight. It is the wrong read, and
    for the same reason retuning r02's `prefill_share` against the inverted-arms finding's arms
    would have been wrong: **the pair never exercised the term.** The mode the
    occupancy leg guards is high bandwidth *with* high occupancy — a kernel
    moving bytes because it is genuinely busy — and neither r01 arm was
    high-occupancy. A term that passes in both arms of a two-arm study is
    untested, not useless. The remedy for untested is an arm, not a deletion.

    This class is that arm, run synthetically. On real hardware it becomes a real
    third condition ("Planned
    third arm"): a compute-bound, high-occupancy workload on which r01 must
    abstain. Until then these tests hold the claim to the same standard the
    fixture can support — and, crucially, they pin that the *bandwidth leg alone
    would have fired*, so the silence is attributable to the guard and to
    nothing else.
    """

    def test_declared_as_a_guard_not_a_detector(self) -> None:
        """The reclassification is part of the contract, not a comment."""
        assert OCCUPANCY_FP_GUARD.mode == "saturated_compute_bound_kernel"
        # The declared signal names the SCOPED locus first and the collapsed one
        # as the fallback (the guard fail-open). A guard whose declaration still claimed
        # `decode.sm_occupancy` outright would be documenting the fail-open.
        assert OCCUPANCY_FP_GUARD.signal.startswith(EV_OCC_SCOPED)
        assert EV_OCC_COLLAPSED in OCCUPANCY_FP_GUARD.signal
        # The behavioural half of the reclassification.
        assert OCCUPANCY_FP_GUARD.grades_confidence is False
        assert OCCUPANCY_FP_GUARD in FP_GUARDS

    def test_saturated_compute_bound_kernel_stays_silent(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """The guarded FP mode: high HBM **and** high occupancy → no diagnosis.

        Telling the operator of an already-saturated kernel to raise the batch
        is at best a no-op and at worst pushes them into KV pressure chasing a
        bottleneck that was never there. That is the wrong answer this leg
        exists to prevent.
        """
        layers = _compute_bound_layers(hbm=0.88, occ=0.82)
        dx = _capture(layers, batch_size=256)
        assert rule.evaluate(dx) is Abstention.BELOW_THRESHOLD
        # And the guard read the busy kernel, not the phase mean (the guard fail-open).
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.sm_occupancy == pytest.approx(0.82)

    def test_the_bandwidth_leg_alone_would_have_fired(self) -> None:
        """Attribution: the silence above is the guard's doing, not the gate's.

        Without this, the arm proves nothing — an abstention that the bandwidth
        leg would have produced anyway tests no guard.
        """
        layers = _compute_bound_layers(hbm=0.88, occ=0.82)
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.kernel_name == "large_batch_gemm_dominant"
        assert scoped.hbm_util > _HBM_THRESHOLD

        # Same capture, the SCOPED kernel's occupancy dropped under the gate and
        # nothing else touched: now it fires. The guard is the only difference.
        # (Dropping the collapsed aggregate instead would no longer flip it —
        # that is the point of guard fail-open, and `TestOccupancyFailOpen` pins it.)
        rule = DecodeMemoryBoundRule()
        fired = rule.evaluate(
            _capture(_compute_bound_layers(hbm=0.88, occ=0.12), batch_size=256)
        )
        assert isinstance(fired, Diagnosis)

    def test_guard_boundary_is_strict(self, rule: DecodeMemoryBoundRule) -> None:
        """Occupancy exactly at 0.30 is suppressed; the gate is `<`, not `<=`.

        Varied on the **dominant kernel's** occupancy, because that is what the
        guard now reads. Varying the collapsed aggregate would move a number the
        rule never looks at on this input, and the test would pass for no reason.
        """
        assert rule.evaluate(
            _capture(_compute_bound_layers(hbm=0.88, occ=_SM_THRESHOLD))
        ) is Abstention.BELOW_THRESHOLD
        assert isinstance(
            rule.evaluate(_capture(_compute_bound_layers(hbm=0.88, occ=0.2999))),
            Diagnosis,
        )

    def test_guard_does_not_grade_confidence(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """**The behavioural half of the reclassification.**

        `signal_strength` used to be `sqrt(hbm_strength * sm_strength)` — a
        geometric mean treating the two legs as co-equal detectors. Under that
        form the field pair would have scored occupancy 0.587 (induce) and 0.433
        (saturate) into the confidence: a large contribution from a signal that
        separated the arms by nothing, and a report in which a lower occupancy
        looks like a stronger diagnosis.

        A guard's contract is binary. Past the gate it must not move the score.
        """
        strengths = {
            occ: rule.evaluate(_input(hbm_util=0.90, sm_occ=occ))
            for occ in (0.01, 0.10, 0.20, 0.29)
        }
        for result in strengths.values():
            assert isinstance(result, Diagnosis)
        values = {
            r.confidence_breakdown.signal_strength for r in strengths.values()
        }
        assert len(values) == 1
        # And it is the detection leg's own value: (0.90 - 0.80) / 0.20.
        assert values.pop() == pytest.approx(0.50)

    def test_strength_still_tracks_the_detection_leg(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """Dropping the geometric mean must not flatten the score entirely."""
        ladder = []
        for hbm in (0.82, 0.86, 0.90, 0.98):
            result = rule.evaluate(_input(hbm_util=hbm, sm_occ=0.14))
            assert isinstance(result, Diagnosis)
            ladder.append(result.confidence_breakdown.signal_strength)
        assert ladder == sorted(ladder)
        assert ladder[0] < ladder[-1]

    def test_role_is_named_in_every_diagnosis(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """A reader must not have to infer that occupancy is a guard.

        Same principle as `hbm_read_locus`: the expensive part of the scope-dilution finding was a
        number whose meaning lived in a file nobody had open.
        """
        scoped = rule.evaluate(_field_input(dominant_hbm=0.86))
        collapsed = rule.evaluate(_input(hbm_util=0.88, sm_occ=0.15))
        for result in (scoped, collapsed):
            assert isinstance(result, Diagnosis)
            assert result.evidence["occupancy_leg_role"] == OCCUPANCY_LEG_ROLE
            assert result.evidence["occupancy_leg_role"].startswith("fp_guard:")
            notes = result.confidence_breakdown.notes
            assert "false-positive guard" in notes
            assert "co-equal detector" in notes


# --------------------------------------------------------------------------- #
# The guard's own fail-open
# --------------------------------------------------------------------------- #

def _diluted_occupancy_layers(dominant_occ: float = 0.82) -> list[LayerMetrics]:
    """A busy dominant kernel buried in idle ones — the exact guard fail-open shape.

    500 ms of large-batch GEMM at 0.88 HBM / 0.82 occupancy, plus 1000 ms spread
    across ten small kernels at ~0.002 HBM / 0.02 occupancy: the same
    near-zero-bandwidth dilution population the scope-dilution finding measured (rms_norm, rotary,
    act_and_mul, …), at the share a mixed capture produces.

    The arithmetic is the whole point:

        dominant kernel   500 ms   hbm 0.880   occ 0.820   (33% of kernel time)
        ten idle kernels 1000 ms   hbm 0.002   occ 0.020   (6.7% each)

        collapsed hbm  = 442 / 1500 = 0.2947    under the 0.80 gate
        collapsed occ  = 430 / 1500 = 0.2867    UNDER the 0.30 guard gate
        scoped    hbm  =              0.8800    over  the 0.80 gate
        scoped    occ  =              0.8200    over  the 0.30 guard gate

    A healthily saturated kernel — 82% occupancy, doing real work — whose phase
    aggregate reads *idle*. Not a measurement: the shape is the field capture's,
    the regime is inverted, and it is the third arm's workload seen through a
    mixed capture.
    """
    layers = [_kernel("large_batch_gemm_dominant", 500.0, hbm=0.88, occ=dominant_occ)]
    for i in range(10):
        layers.append(_kernel(f"idle_elementwise_{i}", 100.0, hbm=0.002, occ=0.02))
    return layers


class TestOccupancyFailOpen:
    """**The hole the guard fail-open names, pinned shut.**

    Scoping the bandwidth leg while the guard kept reading `decode.sm_occupancy`
    re-created the scope-dilution finding's dilution one level down — inside the guard. Idle kernels
    drag the aggregate under 0.30 while the kernel r01 actually read is busy, so
    the guard fails open on precisely the mode it exists to catch: a compute-bound
    high-occupancy kernel diluted into looking low-occupancy.

    The sharp form of the claim, and what this class exists to demonstrate:
    **scoping one leg and leaving the other collapsed is worse than scoping
    neither.** On this fixture, scoping nothing is silent, scoping both is silent,
    and the half-scoped state in between is the only one that produces a wrong
    answer. That is not a nuance of degree — a partial scope does not remove the
    dilution, it relocates it into the term nobody is looking at.
    """

    def test_fixture_has_the_shape_the_finding_describes(self) -> None:
        """Pinned as a measurement of the fixture, like the scope-dilution finding flat read.

        If these numbers drift, the regressions below stop testing guard fail-open and
        nobody would be able to tell from a green run.
        """
        layers = _diluted_occupancy_layers()
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None

        # The kernel r01 reads is busy, on both legs.
        assert scoped.kernel_name == "large_batch_gemm_dominant"
        assert scoped.hbm_util == pytest.approx(0.88)
        assert scoped.sm_occupancy == pytest.approx(0.82)
        assert scoped.kernel_share == pytest.approx(1 / 3, abs=0.001)
        assert scoped.kernels_below_min_share == 10

        # The phase it lives in looks idle.
        assert _flat_occ_mean(layers) == pytest.approx(0.2867, abs=0.001)
        assert _flat_occ_mean(layers) < _SM_THRESHOLD
        assert scoped.sm_occupancy > _SM_THRESHOLD

    def test_the_collapsed_read_would_have_fired_this_false_positive(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """**The regression.** The old read fires on a saturated kernel; this must not.

        Both terms of the pre-fix predicate hold on this capture — the scoped
        bandwidth leg clears 0.80 and the *collapsed* occupancy clears 0.30
        downwards — so the rule as it stood after the locus fix would have
        emitted a diagnosis telling the operator of an 82%-occupancy kernel to
        raise their batch. That is the single false positive OCCUPANCY_FP_GUARD
        exists to prevent, produced by the guard itself.
        """
        layers = _diluted_occupancy_layers()
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None

        # The old predicate, term by term: D1 on the scoped read, G1 on the
        # collapsed one. Both true ⇒ the pre-fix rule fires.
        assert scoped.hbm_util > _HBM_THRESHOLD
        assert _flat_occ_mean(layers) < _SM_THRESHOLD

        # And it is not silent for some unrelated reason: the scoped read now
        # suppresses it, on the same input.
        assert rule.evaluate(_capture(layers, batch_size=256)) is Abstention.BELOW_THRESHOLD

    def test_the_old_read_path_driven_deliberately_still_fires(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """The same claim as an execution, not an arithmetic argument.

        r01's mixed-locus fallback — scoped bandwidth, collapsed occupancy — is
        bit-for-bit the pre-fix read. Driving it deliberately (per-kernel
        occupancy withheld from `layers`, the phase aggregate the parser derived
        from the full capture left in place) reproduces the old behaviour through
        the documented fallback instead of by reverting the fix: it FIRES.

        Which is also the honest reading of that fallback today. When the
        selected kernel carries no occupancy counter the guard is back to gating
        on a diluted number, and the diagnosis says so — `read_loci_agree=False`,
        `data_completeness=0.75`, and the caveat in the notes — rather than
        presenting a two-locus verdict as a statement about one kernel.
        """
        full = _diluted_occupancy_layers()
        no_occ_counters = [
            _kernel(k.layer_name, k.duration_ms, hbm=k.hbm_bandwidth_util, occ=None)
            for k in full
        ]
        fired = rule.evaluate(
            _input(
                hbm_util=flat_hbm_mean(no_occ_counters),
                sm_occ=_flat_occ_mean(full),      # what the parser derived
                batch_size=256,
                layers=no_occ_counters,
            )
        )
        assert isinstance(fired, Diagnosis)
        assert fired.evidence["hbm_read_locus"] == LOCUS_SCOPED
        assert fired.evidence["occupancy_read_locus"] == LOCUS_COLLAPSED
        assert fired.evidence["read_loci_agree"] is False
        assert fired.confidence_breakdown.data_completeness == pytest.approx(0.75)
        assert "different loci" in fired.confidence_breakdown.notes
        assert "guard permissive" in fired.confidence_breakdown.notes

    def test_a_partial_scope_is_worse_than_no_scope(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """**the guard fail-open's sharpest claim, as three verdicts on one capture.**

        scope nothing  → silent (bandwidth diluted to 0.29: right answer, wrong
                         reason — the scope-dilution finding false negative, still there)
        scope bandwidth→ FIRES on an 82%-occupancy kernel: a false POSITIVE, and
                         a wrong answer the unscoped rule would not have given
        scope both     → silent, via the guard, on the correct reason

        Scoping half the predicate did not halve the dilution. It converted a
        false negative into a false positive, which is strictly worse: the false
        negative is silence, and the false positive is a confident instruction to
        raise a batch that is already saturated.
        """
        full = _diluted_occupancy_layers()
        collapsed_hbm = flat_hbm_mean(full)
        collapsed_occ = _flat_occ_mean(full)
        assert collapsed_hbm is not None

        # 1. Scope nothing — the DCGM-shaped read.
        assert rule.evaluate(
            _input(hbm_util=collapsed_hbm, sm_occ=collapsed_occ, layers=None)
        ) is Abstention.BELOW_THRESHOLD

        # 2. Scope the bandwidth leg only — the state guard fail-open indicts.
        half_scoped = rule.evaluate(
            _input(
                hbm_util=collapsed_hbm,
                sm_occ=collapsed_occ,
                layers=[
                    _kernel(k.layer_name, k.duration_ms,
                            hbm=k.hbm_bandwidth_util, occ=None)
                    for k in full
                ],
            )
        )
        assert isinstance(half_scoped, Diagnosis)

        # 3. Scope both — the fix.
        assert rule.evaluate(_capture(full)) is Abstention.BELOW_THRESHOLD

    def test_suppression_is_the_guards_doing_not_the_gates(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """Attribution, same standard the third arm is held to.

        An abstention the bandwidth leg would have produced anyway tests nothing.
        Here the scoped bandwidth read clears 0.80 comfortably, and dropping only
        the dominant kernel's occupancy under the gate — nothing else touched —
        flips the verdict.
        """
        busy = _diluted_occupancy_layers(dominant_occ=0.82)
        scoped = apply_kernel_scope(busy, KERNEL_SCOPE)
        assert scoped is not None and scoped.hbm_util > _HBM_THRESHOLD
        assert rule.evaluate(_capture(busy)) is Abstention.BELOW_THRESHOLD

        idle = _diluted_occupancy_layers(dominant_occ=0.12)
        fired = rule.evaluate(_capture(idle))
        assert isinstance(fired, Diagnosis)
        assert fired.evidence[EV_OCC_SCOPED] == pytest.approx(0.12)
        assert fired.evidence["read_loci_agree"] is True


# --------------------------------------------------------------------------- #
# Kernel scope mechanics
# --------------------------------------------------------------------------- #

class TestKernelScope:
    def test_declared_scope(self) -> None:
        """The declaration is part of the contract; `phase` especially.

        `phase="any"` is not a placeholder for `"decode"`. No parser tags
        prefill vs decode, and `LayerMetrics` has no phase field, so a
        `decode` claim would assert a precision the data does not carry.
        """
        assert KERNEL_SCOPE.select == "dominant_by_total_ms"
        assert KERNEL_SCOPE.phase == "any"
        assert KERNEL_SCOPE.aggregate == "none"
        assert KERNEL_SCOPE.min_share == 0.15

    def test_min_share_excludes_a_small_hot_kernel(self) -> None:
        """A 10%-of-time kernel at 0.99 must not become the reading.

        The guard against over-correcting: scoping fixed a false negative and
        must not manufacture a false positive out of a brief bandwidth spike.
        """
        layers = [
            _kernel("big_gemm", 900.0, hbm=0.40),
            _kernel("tiny_hot_kernel", 100.0, hbm=0.99),
        ]
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.kernel_name == "big_gemm"
        assert scoped.hbm_util == pytest.approx(0.40)
        assert scoped.kernels_below_min_share == 1
        assert scoped.kernels_in_scope == 1

    def test_min_share_boundary_is_inclusive(self) -> None:
        layers = [
            _kernel("big_gemm", 850.0, hbm=0.40),
            _kernel("exactly_15_pct", 150.0, hbm=0.99),
        ]
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.kernels_below_min_share == 0
        assert scoped.kernels_in_scope == 2
        # Still not selected — eligible, but not dominant.
        assert scoped.kernel_name == "big_gemm"

    def test_sub_min_share_kernel_cannot_fire_the_rule(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        dx = _input(
            hbm_util=0.46, sm_occ=0.10, batch_size=8,
            layers=[
                _kernel("big_gemm", 900.0, hbm=0.40),
                _kernel("tiny_hot_kernel", 100.0, hbm=0.99),
            ],
        )
        assert rule.evaluate(dx) is Abstention.BELOW_THRESHOLD

    def test_launches_of_one_kernel_are_summed_not_diluted(self) -> None:
        """`dominant_by_total_ms` means total, across a name's own launches.

        Averaging *within* one kernel name is not the aggregation the scope-dilution finding
        indicts: repeated launches of the same kernel do the same work at the
        same arithmetic intensity.
        """
        layers = [
            _kernel("gemm", 100.0, hbm=0.90),
            _kernel("gemm", 100.0, hbm=0.86),
            _kernel("other", 150.0, hbm=0.10),
        ]
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.kernel_name == "gemm"
        assert scoped.kernel_ms == pytest.approx(200.0)
        assert scoped.hbm_util == pytest.approx(0.88)

    def test_zero_duration_kernels_are_not_counted(self) -> None:
        layers = [
            _kernel("gemm", 100.0, hbm=0.90),
            _kernel("never_ran", 0.0, hbm=0.99),
            _kernel("untimed", None, hbm=0.99),
        ]
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.kernel_name == "gemm"
        assert scoped.kernels_named == 1

    def test_dominant_kernel_without_a_counter_is_skipped(self) -> None:
        """Duration-only records cannot be read, but still count against shares."""
        layers = [
            _kernel("no_counters_gemm", 600.0, hbm=None),
            _kernel("measured_gemm", 400.0, hbm=0.83),
        ]
        scoped = apply_kernel_scope(layers, KERNEL_SCOPE)
        assert scoped is not None
        assert scoped.kernel_name == "measured_gemm"
        # Share is measured against the FULL denominator, not the eligible subset.
        assert scoped.kernel_share == pytest.approx(0.40)

    def test_scope_not_applicable_returns_none(self) -> None:
        assert apply_kernel_scope(None, KERNEL_SCOPE) is None
        assert apply_kernel_scope([], KERNEL_SCOPE) is None
        # Timed, but no bandwidth counters anywhere.
        assert apply_kernel_scope([_kernel("k", 10.0, hbm=None)], KERNEL_SCOPE) is None
        # Counters, but no positive duration.
        assert apply_kernel_scope([_kernel("k", 0.0, hbm=0.9)], KERNEL_SCOPE) is None

    def test_exact_tie_is_broken_deterministically(self) -> None:
        a = [_kernel("b_gemm", 500.0, hbm=0.9), _kernel("a_gemm", 500.0, hbm=0.1)]
        b = list(reversed(a))
        first = apply_kernel_scope(a, KERNEL_SCOPE)
        second = apply_kernel_scope(b, KERNEL_SCOPE)
        assert first is not None and second is not None
        assert first.kernel_name == second.kernel_name == "a_gemm"

    def test_unsupported_scope_is_rejected(self) -> None:
        from rules.r01_decode_memory_bound import KernelScope

        bad = KernelScope(
            select="dominant_by_total_ms", phase="any",
            aggregate="unweighted_mean", min_share=0.15,
        )
        with pytest.raises(ValueError, match="aggregate"):
            apply_kernel_scope([_kernel("k", 1.0, hbm=0.9)], bad)


# --------------------------------------------------------------------------- #
# Fallback to the collapsed scalar (the DCGM path)
# --------------------------------------------------------------------------- #

class TestCollapsedFallback:
    def test_no_layers_reads_the_collapsed_scalar(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """`parsers/dcgm.py` supplies a device-level counter with no kernels."""
        result = rule.evaluate(_input(hbm_util=0.88, sm_occ=0.15, batch_size=4))
        assert isinstance(result, Diagnosis)
        assert result.evidence["decode.hbm_bandwidth_util"] == pytest.approx(0.88)
        assert "layers.dominant.hbm_bandwidth_util" not in result.evidence

    def test_fallback_is_recorded_in_the_evidence(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """A reader must be able to tell which locus produced the verdict.

        Not being able to is what made the scope-dilution finding expensive to find.
        """
        result = rule.evaluate(_input(hbm_util=0.88, sm_occ=0.15))
        assert isinstance(result, Diagnosis)
        assert result.evidence["hbm_read_locus"] == LOCUS_COLLAPSED
        notes = result.confidence_breakdown.notes
        assert "COLLAPSED" in notes and "scope dilution" in notes
        # And the honest completeness: one of the two loci was available.
        assert result.confidence_breakdown.data_completeness == pytest.approx(0.5)

    def test_unusable_layers_fall_back_rather_than_abstain(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """Duration-only `layers` must not disable a rule the DCGM path can run."""
        result = rule.evaluate(
            _input(
                hbm_util=0.88, sm_occ=0.15,
                layers=[_kernel("k", 10.0, hbm=None)],
            )
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["hbm_read_locus"] == LOCUS_COLLAPSED

    def test_scoped_path_reports_full_completeness(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        result = rule.evaluate(_field_input(dominant_hbm=0.86))
        assert isinstance(result, Diagnosis)
        assert result.confidence_breakdown.data_completeness == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Acceptance matrix (collapsed inputs)
# --------------------------------------------------------------------------- #

class TestDecodeMemoryBoundRule:
    def test_fires_on_clear_signal(self, rule: DecodeMemoryBoundRule) -> None:
        result = rule.evaluate(_input(hbm_util=0.89, sm_occ=0.14, batch_size=8))

        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r01"
        # Signal strength saturates well past the cap, so the cap is what shows.
        assert result.confidence == pytest.approx(_UNCALIBRATED_CEILING)
        # The detection leg alone: (0.89 - 0.80) / 0.20. Occupancy guards, it
        # does not grade — see TestOccupancyFpGuard.
        assert result.confidence_breakdown.signal_strength == pytest.approx(0.45)

        # batch_size present → cause string must include it
        assert "8" in result.cause

    def test_uncalibrated_cap_holds(self, rule: DecodeMemoryBoundRule) -> None:
        """r01 carries an open field finding against its own gate (the scope-dilution finding).

        Scoping fixed *where* the number is read; it did not validate *what the
        number should be*. Until a hardware run re-derives the 0.80 threshold this
        rule hedges like the other eight — it may not hold the library's
        highest confidence ceiling while its seed is contradicted by field data.
        """
        result = rule.evaluate(_input(hbm_util=0.999, sm_occ=0.001))
        assert isinstance(result, Diagnosis)
        assert result.confidence <= _UNCALIBRATED_CEILING
        assert "uncalibrated" in result.confidence_breakdown.notes

    def test_fires_on_clear_signal_no_batch_size(self, rule: DecodeMemoryBoundRule) -> None:
        result = rule.evaluate(_input(hbm_util=0.89, sm_occ=0.14, batch_size=None))

        assert isinstance(result, Diagnosis)
        # batch_size absent → no "batch size" clause in cause
        assert "batch size" not in result.cause.lower()

    def test_confidence_at_boundary(self, rule: DecodeMemoryBoundRule) -> None:
        # Just past threshold on both signals → confidence near the floor
        result = rule.evaluate(_input(hbm_util=0.81, sm_occ=0.29))

        assert isinstance(result, Diagnosis)
        assert 0.50 <= result.confidence <= 0.60

    def test_abstains_below_hbm_threshold(self, rule: DecodeMemoryBoundRule) -> None:
        result = rule.evaluate(_input(hbm_util=0.75, sm_occ=0.14))
        assert result is Abstention.BELOW_THRESHOLD

    def test_abstains_when_sm_too_high(self, rule: DecodeMemoryBoundRule) -> None:
        result = rule.evaluate(_input(hbm_util=0.89, sm_occ=0.45))
        assert result is Abstention.BELOW_THRESHOLD

    def test_scoped_read_does_not_bypass_the_occupancy_leg(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """A saturated-but-busy GPU stays silent even with a hot dominant kernel."""
        dx = _input(
            hbm_util=0.85, sm_occ=0.60,
            layers=[_kernel("gemm", 100.0, hbm=0.95)],
        )
        assert rule.evaluate(dx) is Abstention.BELOW_THRESHOLD

    def test_abstains_at_exact_threshold(self, rule: DecodeMemoryBoundRule) -> None:
        # Firing condition is strict inequality; boundary values must not fire.
        result = rule.evaluate(_input(hbm_util=0.80, sm_occ=0.30))
        assert result is Abstention.BELOW_THRESHOLD

    def test_insufficient_data_no_decode(self, rule: DecodeMemoryBoundRule) -> None:
        result = rule.evaluate(_input(decode_present=False))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("decode.hbm_bandwidth_util", "decode.sm_occupancy")

    def test_insufficient_data_missing_field(self, rule: DecodeMemoryBoundRule) -> None:
        # hbm_bandwidth_util absent; sm_occupancy present
        result = rule.evaluate(_input(hbm_util=None, sm_occ=0.14))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("decode.hbm_bandwidth_util",)

    def test_layers_alone_supply_both_legs(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """**Behaviour change, the guard fail-open.** Per-kernel records now carry occupancy.

        Before the guard was scoped, a capture with `layers` and no `decode`
        block was INSUFFICIENT_DATA: the bandwidth leg had a locus and the guard
        did not. Now both legs read the same kernel, so per-kernel records alone
        are a complete input — which is the honest consequence of the scope
        binding both legs rather than one.
        """
        result = rule.evaluate(
            _input(decode_present=False, layers=_field_layers(dominant_hbm=0.86))
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["occupancy_read_locus"] == LOCUS_SCOPED
        assert result.evidence[EV_OCC_SCOPED] == pytest.approx(0.118)
        # No `decode` block, so there is no collapsed contrast to publish.
        assert "collapsed_sm_occupancy" not in result.evidence

    def test_neither_locus_supplies_occupancy_is_still_insufficient(
        self, rule: DecodeMemoryBoundRule
    ) -> None:
        """Scoping the guard must not invent an occupancy the capture lacks.

        `layers` with bandwidth but no occupancy counters, and no `decode` block
        to fall back to: r01 abstains for missing data, named for the field a
        user can actually go supply.
        """
        result = rule.evaluate(
            _input(
                decode_present=False,
                layers=[_kernel("gemm", 100.0, hbm=0.95, occ=None)],
            )
        )
        assert isinstance(result, InsufficientData)
        assert result.missing == (EV_OCC_COLLAPSED,)

    def test_evidence_populated(self, rule: DecodeMemoryBoundRule) -> None:
        dx = _input(hbm_util=0.89, sm_occ=0.14, batch_size=16)
        result = rule.evaluate(dx)

        assert isinstance(result, Diagnosis)
        assert result.evidence["decode.hbm_bandwidth_util"] == pytest.approx(0.89)
        assert result.evidence["decode.sm_occupancy"] == pytest.approx(0.14)
        assert result.evidence["batch_size"] == 16
