"""
r01 — Decode memory-bound at low batch.

Fires when HBM bandwidth utilisation is high and SM occupancy is low during
decode, signalling that the kernel sits in the memory-bound regime of the
roofline. The fix is increasing batch size to raise arithmetic intensity.

WHERE the bandwidth number is read
----------------------------------
Field testing (A100-PCIE-40GB, vLLM 0.10.2, Qwen2.5-7B, batch-1 decode — the
textbook memory-bound case this rule exists to detect) measured three different
HBM utilisations off the SAME capture:

    0.795   the dominant decode GEMM alone      (43.6% of profiled kernel time)
    0.720   all GEMM kernels together           (82.6% of profiled kernel time)
    0.595   every kernel averaged together      (100%)

Reading the flat average (0.595) against the 0.80 gate leaves r01 SILENT on a
workload that is memory-bound by construction — a false negative. Batch 1 -> 256
moved HBM utilisation 0.595 -> 0.295 and SM occupancy 0.124 -> 0.170, a clean
walk off the memory-bound corner in the direction Pope et al. predict.

The defect is not the weighting. `parsers/nsight.py::_aggregate_phase` is
already duration-weighted and still produces the failure:

    0.720 x (1 - 0.174) = 0.5947        vs        0.595 measured

Near-zero-bandwidth kernels (elementwise / norm / rotary) held 17.4% of step
time at batch 1. Weighting by duration does not remove them from the
denominator; only scoping does. The dilution also fails toward silence exactly
where the rule hunts: at batch 256 those kernels do real work and hold ~1% of
step time.

So r01 declares a `KERNEL_SCOPE` and reads its signal off the dominant kernel in
`dx.layers`. When there are no per-kernel records (the DCGM path) it falls back
to the collapsed `dx.decode.hbm_bandwidth_util` and says so in the evidence
(`hbm_read_locus`).

Known limits: the 0.80 threshold is not re-derived (0.795 on the carrying kernel
suggests the seed is close, but one model on one GPU is not enough to move it),
and the guard's 0.30 gate has not been tested against a real compute-bound,
high-occupancy capture.

WHERE the occupancy number is read
----------------------------------
`KERNEL_SCOPE` binds **both** kernel-derived legs. On a mixed capture, idle
kernels can drag aggregate occupancy under the 0.30 gate while the kernel r01
actually read is busy — so a guard reading the aggregate fails open on exactly
the mode it exists to catch. Occupancy therefore comes off the **same
`LayerMetrics` record** the bandwidth reading came from. If that kernel carries
no occupancy counter, occupancy falls back to `dx.decode.sm_occupancy`, and the
evidence records `occupancy_read_locus` plus a `read_loci_agree` flag.

The selection is by total time among kernels carrying a **bandwidth** counter;
occupancy never influences which kernel is picked.

WHAT the occupancy leg is for
-----------------------------
In field testing SM occupancy was 0.124 and 0.170 — both far under 0.30 — so the
leg contributed no separation; all discrimination came from bandwidth. It is kept
as a declared false-positive guard, `OCCUPANCY_FP_GUARD`: high bandwidth *with*
high occupancy is a healthily saturated compute-bound kernel, which a
bandwidth-only rule would wrongly tell the operator to batch up.

* `signal_strength` is the **bandwidth leg alone**. A guard suppresses or it does
  not; it does not grade confidence.
* The gate is a hard precondition: a high-occupancy kernel cannot fire r01 no
  matter how hot its bandwidth reading.

The guard is covered synthetically by
`tests/rules/test_r01_decode_memory_bound.py::TestOccupancyFpGuard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput, LayerMetrics

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
# _HBM_THRESHOLD is a `measured`-label CANDIDATE, not a measured value. Field
# testing put the carrying kernel at 0.795 against this 0.80 seed — close enough
# that the seed may well survive scoping, and far too little evidence to re-seat
# it: one model, one GPU, two batch sizes, on an ncu capture whose serialised
# kernel replay may itself depress bandwidth. Re-derivation needs a
# paired hardware run with an nsys cross-check; until then this number stays
# where it was and THRESHOLDS_UNCALIBRATED caps confidence.
_HBM_THRESHOLD = 0.80

# The false-positive guard's gate. NOT a detection threshold — see
# OCCUPANCY_FP_GUARD. A kernel at or above this occupancy is doing enough work
# that a high bandwidth reading is evidence of health, not of a memory bound.
_SM_THRESHOLD = 0.30

# Signal-strength normalisation span for the DETECTION leg. There is no
# corresponding `_SM_SPAN`: the occupancy leg gates, it does not grade.
_HBM_SPAN = 0.20   # 0.80 -> 1.00 maps to strength 0 -> 1

# Confidence model constants.
# Ceiling is 0.9: a single rule should not claim certainty without corroborating
# layer-level evidence (other rules supply that).
_CONFIDENCE_FLOOR = 0.5
_CONFIDENCE_SCALE = 0.8
_CONFIDENCE_CEILING = 0.9

# r01 was the last rule in the library still claiming an uncapped 0.9 while
# carrying an open field finding against its own gate (the 0.80 seed did not
# reproduce on real A100 decode). Scoping the read fixes *where* the number
# comes from; it does not validate *what the number should be*. Until a hardware
# run re-derives the threshold, this rule hedges like the other eight.
THRESHOLDS_UNCALIBRATED = True
_UNCALIBRATED_CEILING = 0.65


# --------------------------------------------------------------------------- #
# Evidence tokens
# --------------------------------------------------------------------------- #
# Locus tokens for `hbm_read_locus` and `occupancy_read_locus`. Machine-readable
# and stable — a reader (or a future corpus query) must be able to tell the two
# loci apart, on either leg, without parsing prose.
LOCUS_SCOPED = "layers.dominant_by_total_ms"
LOCUS_COLLAPSED = "decode.collapsed_aggregate"

# Evidence KEY names, one per locus per leg. The two loci get different key names
# on purpose: the scope-dilution bug was hard to find precisely because
# `decode.hbm_bandwidth_util` looked like a decode measurement and was in fact an
# average over every kernel in the trace. Naming them as constants keeps the
# emitted evidence and the FpGuard declaration below from drifting apart.
EV_HBM_SCOPED = "layers.dominant.hbm_bandwidth_util"
EV_HBM_COLLAPSED = "decode.hbm_bandwidth_util"
EV_OCC_SCOPED = "layers.dominant.sm_occupancy"
EV_OCC_COLLAPSED = "decode.sm_occupancy"


# --------------------------------------------------------------------------- #
# False-positive guards — terms that suppress, rather than detect
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FpGuard:
    """A predicate retained to suppress a named false-positive mode.

    The distinction from a detection leg is not decorative. A detector's job is
    to separate the pathological arm from the healthy one, and it is judged on
    whether it does; a guard's job is to stay out of the way until a specific
    wrong answer becomes reachable, and it is judged on whether that answer is
    reachable past it. The two are therefore falsified by different experiments,
    and — this is the part r01 got wrong — **a guard that never triggers in a
    validation pair is not thereby shown to be useless.** It is shown to be
    untested, which is a reason to build the arm that tests it.

    Fields:
        mode:        the false positive this term exists to prevent, named so it
                     can be argued with.
        signal:      the schema path read.
        gate:        the predicate, as it appears in the firing condition.
        grades_confidence:
                     whether the term contributes to ``signal_strength``. False
                     for every guard: suppression is binary, and grading a
                     diagnosis by how comfortably it cleared a guard imports
                     detection semantics the term has not earned.
        status:      what evidence, if any, has exercised it.
    """

    mode: str
    signal: str
    gate: str
    grades_confidence: bool
    status: str


OCCUPANCY_FP_GUARD = FpGuard(
    mode="saturated_compute_bound_kernel",
    # Read under KERNEL_SCOPE, off the SAME kernel as the detection leg.
    # The collapsed aggregate is the fallback, not the
    # primary, and the evidence names which one produced the verdict.
    signal=f"{EV_OCC_SCOPED} (fallback: {EV_OCC_COLLAPSED})",
    gate=f"< {_SM_THRESHOLD}",
    grades_confidence=False,
    status=(
        "not exercised by field captures, which measured occupancy 0.124 and "
        "0.170 — both far under the gate, so the guard passed in both and "
        "separated nothing. The compute-bound, high-occupancy case is covered "
        "synthetically by TestOccupancyFpGuard. Scoped to the dominant kernel: "
        "reading the collapsed aggregate made the guard fail open on its own FP "
        "mode."
    ),
)
"""Why r01 keeps a leg that never discriminated in the field.

A kernel can move a great deal of bandwidth *because it is busy*. Large-batch
GEMMs, fused attention at long context, a well-shaped MoE expert — these sit high
on the bandwidth axis while the SMs are genuinely occupied, and they are the
healthy case. Without the occupancy term r01 would read them as memory-bound and
recommend raising the batch, which for an already-saturated kernel is at best a
no-op and at worst pushes the operator into KV pressure chasing a bottleneck that
was never there.

Occupancy is read under `KERNEL_SCOPE`, off the **same kernel** as the detection
leg. It did not used to be, and the interim state was the worst of
the three available: on a mixed capture idle-ish kernels drag the phase aggregate
under 0.30 while the dominant kernel is busy, so the guard went *permissive* in
exactly the FP direction — a saturated compute-bound kernel diluted into looking
low-occupancy, which is the one wrong answer this term exists to prevent. A guard
that reads a different kernel from the leg it is guarding is not a guard.

Falling back to the collapsed aggregate is still allowed — the DCGM path has no
kernels, and an Nsight capture can carry a duration/bandwidth record with no
occupancy counter on it — but the fallback is *recorded*: `occupancy_read_locus`
and `read_loci_agree` appear in every diagnosis, so a verdict whose two legs came
from different loci says so instead of reading like one about a single kernel.

What is still open is the gate, not the locus: whether 0.30 sits in the right
place still needs a compute-bound, high-occupancy capture on real hardware.
"""

FP_GUARDS = (OCCUPANCY_FP_GUARD,)
"""Every term in r01's firing condition that suppresses rather than detects."""


# --------------------------------------------------------------------------- #
# Kernel scope — WHERE the bandwidth signal is read from
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KernelScope:
    """A declaration of which kernels a rule's kernel-derived signals are read over.

    It binds **every** kernel-derived leg of the predicate, not only the one that
    motivated it. Scoping a subset is worse than scoping nothing: the
    dilution the scope exists to remove does not disappear, it relocates into
    whichever term was left collapsed, where no evidence key points at it. See the
    module docstring, "WHERE the occupancy number is read".

    Fields:
        select:     how the kernel of interest is picked. Only
                    ``"dominant_by_total_ms"`` is implemented: sum every record
                    sharing a ``layer_name`` and take the largest total.
        phase:      which inference phase the scope is restricted to. Only
                    ``"any"`` is implemented — see the note on KERNEL_SCOPE.
        aggregate:  how selected kernels are combined. ``"none"`` means exactly
                    one kernel is read and NOTHING is averaged across kernels.
                    This is the whole point of the scope: field testing showed that a
                    duration-weighted mean across kernels still buries the
                    signal, because weighting cannot remove a zero-bandwidth
                    kernel from the denominator.
        min_share:  a kernel must hold at least this fraction of total profiled
                    kernel time to be eligible. Guards the opposite failure from
                    the one that motivated the scope: a 0.4 ms kernel at 0.99
                    utilisation is not a decode bottleneck, and without a floor
                    the scoped read would happily promote it.

    This lives here as a local structure on purpose: r01 is its only user.
    """

    select: str
    phase: str
    aggregate: str
    min_share: float


KERNEL_SCOPE = KernelScope(
    select="dominant_by_total_ms",
    phase="any",
    aggregate="none",
    min_share=0.15,
)
"""r01's read scope for BOTH kernel-derived legs — `hbm_bandwidth_util`
(detection) and `sm_occupancy` (the FP guard).

``phase="any"``, NOT ``phase="decode"``, and the distinction is load-bearing.
`parsers/nsight.py` (see the comment above its `_aggregate_phase` call) states
that an Nsight CSV carries no prefill/decode tags, so it folds EVERY kernel into
a single `PhaseMetrics` and hands it to the `decode=` slot; `parsers/dcgm.py`
does the same with a device-level counter. r01's "decode" reading is therefore a
**mislabel inherited from the parser, not a selection** — nothing in the input
has separated decode kernels from anything else.

`LayerMetrics` has no `phase` field, and no timestamp to correlate against the
`NsysTimeline` NVTX step ranges, so this rule *cannot* honestly restrict itself
to decode today. Declaring `phase="decode"` over that parser would be a quieter
version of the exact error kernel scoping exists to fix: a scope claiming a
precision the data does not carry. Saying `any` is the honest label.

Consequence, stated plainly for anyone reading a diagnosis: on a capture that
contains prefill work, the "dominant kernel" may be a prefill GEMM. Real phase
separation needs the schema 1.8.0 `phase` field to be populated.
"""

_FIX = (
    "Increase decode batch size to improve arithmetic intensity. "
    "For FP16 weights, each decode step loads all parameters once regardless "
    "of batch, so arithmetic intensity ≈ batch_size FLOP/byte "
    "(Pope et al. 2023, §3). "
    "H100's compute/memory roofline is ~295 FLOP/byte, so target "
    "batch ≥ 256 to approach the compute-bound regime; halve that target "
    "for FP8 weights. "
    "If KV cache headroom is the limiting factor, enable chunked prefill to "
    "free capacity."
)

# Evidence token for `occupancy_leg_role`. Machine-readable, same reason as the
# locus tokens: a corpus query must be able to tell a detector from a guard
# without parsing prose.
OCCUPANCY_LEG_ROLE = f"fp_guard:{OCCUPANCY_FP_GUARD.mode}"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Scoped read (pure helpers)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Kernel:
    """One kernel name's totals, collapsed across its own repeated launches."""

    name: str
    total_ms: float
    hbm_util: Optional[float]
    sm_occupancy: Optional[float]


@dataclass(frozen=True)
class ScopedKernelRead:
    """The result of applying KERNEL_SCOPE to a per-kernel breakdown.

    Carries BOTH kernel-derived signals r01's predicate uses, read off the same
    selected kernel. `sm_occupancy` is Optional where `hbm_util` is not: the
    selection requires a bandwidth counter (there is nothing to detect on without
    one), while a capture can carry duration and bandwidth for a kernel and no
    occupancy metric. That case falls back on the occupancy leg alone, and the
    diagnosis records the split.
    """

    hbm_util: float
    sm_occupancy: Optional[float]
    kernel_name: str
    kernel_ms: float
    kernel_share: float          # of total profiled kernel time
    kernels_named: int           # distinct kernel names carrying time
    kernels_in_scope: int        # eligible after min_share + counter filters
    kernels_below_min_share: int # excluded by min_share


def _kernel_totals(layers: list[LayerMetrics]) -> list[_Kernel]:
    """Collapse `layers` into one record per kernel name.

    `parsers/nsight.py` already sums repeated launches of a kernel name into one
    `LayerMetrics`, so this is usually a no-op — but grouping here makes
    ``select="dominant_by_total_ms"`` literally true rather than true-by-luck,
    and keeps the scope correct for any future parser that emits one record per
    launch.

    Combining *within* one kernel name is not the aggregation that caused scope dilution:
    repeated launches of the same kernel do the same work with the same
    arithmetic intensity, so their duration-weighted mean is a measurement of
    that kernel. The damage came from averaging *across different* kernels.

    Bandwidth and occupancy are weighted over their OWN denominators — only the
    launches that actually carry each counter. Sharing one denominator would let
    a launch missing an occupancy metric silently deflate the occupancy mean,
    which is the dilution defect in miniature.
    """
    # name -> [total_ms, hbm*ms sum, ms with hbm, occ*ms sum, ms with occ]
    totals: dict[str, list[float]] = {}
    for layer in layers:
        ms = layer.duration_ms
        if ms is None or ms <= 0:
            continue
        acc = totals.setdefault(layer.layer_name, [0.0, 0.0, 0.0, 0.0, 0.0])
        acc[0] += ms
        if layer.hbm_bandwidth_util is not None:
            acc[1] += layer.hbm_bandwidth_util * ms
            acc[2] += ms
        if layer.sm_occupancy is not None:
            acc[3] += layer.sm_occupancy * ms
            acc[4] += ms

    out: list[_Kernel] = []
    for name, (total_ms, hbm_ms, hbm_den, occ_ms, occ_den) in totals.items():
        out.append(
            _Kernel(
                name=name,
                total_ms=total_ms,
                hbm_util=(hbm_ms / hbm_den) if hbm_den > 0 else None,
                sm_occupancy=(occ_ms / occ_den) if occ_den > 0 else None,
            )
        )
    return out


def apply_kernel_scope(
    layers: Optional[list[LayerMetrics]],
    scope: KernelScope = KERNEL_SCOPE,
) -> Optional[ScopedKernelRead]:
    """Read `hbm_bandwidth_util` AND `sm_occupancy` off the kernel `scope` selects.

    Both come from the same selected kernel, which is the entire point: a
    predicate whose legs describe different kernels is not a statement
    about any kernel.

    Returns None when the scope cannot be applied — no per-kernel records, no
    kernel carrying positive duration, or no kernel that both clears
    ``min_share`` and carries a bandwidth counter. None means "fall back to the
    collapsed scalars and say so", never "abstain": the DCGM path has no kernels
    at all and must keep working.

    The `min_share` denominator is the total time of ALL timed kernels, not just
    eligible ones — a kernel's share of the step is what makes it dominant, and
    measuring it against a filtered denominator would inflate every share.

    "Dominant" means dominant among kernels the scope can actually read: a
    duration-only record (a capture without bandwidth counters) cannot be the
    selection even if it is the longest, because there is nothing to read off
    it. Its time still counts in the denominator.

    Eligibility keys on the **bandwidth** counter only, and occupancy rides along
    on whatever the winner happens to carry. Making occupancy a selection
    criterion would let the guard's data availability move the *detection*
    reading onto a different kernel — a quieter restatement of the same two-loci
    defect. A selected kernel with no occupancy metric yields
    ``sm_occupancy=None`` and the caller falls back on that leg alone.
    """
    if scope.select != "dominant_by_total_ms":
        raise ValueError(f"unsupported KernelScope.select: {scope.select!r}")
    if scope.aggregate != "none":
        raise ValueError(
            f"unsupported KernelScope.aggregate: {scope.aggregate!r}. r01 reads a "
            f"single kernel by design; aggregating across kernels is the scope-dilution "
            f"defect, not a configuration option."
        )
    if not layers:
        return None

    kernels = _kernel_totals(layers)
    total_ms = sum(k.total_ms for k in kernels)
    if total_ms <= 0:
        return None

    below_min_share = sum(
        1 for k in kernels if (k.total_ms / total_ms) < scope.min_share
    )
    eligible = [
        k for k in kernels
        if (k.total_ms / total_ms) >= scope.min_share and k.hbm_util is not None
    ]
    if not eligible:
        return None

    # Deterministic selection: largest total time, ties broken by name. An exact
    # tie has no principled winner, and picking by bandwidth would bias the gate
    # (toward firing if max, toward the original silence if min) — so break it on
    # something neutral and stable instead. Tests pin determinism.
    dominant = min(eligible, key=lambda k: (-k.total_ms, k.name))

    return ScopedKernelRead(
        hbm_util=dominant.hbm_util,          # not None: filtered above
        sm_occupancy=dominant.sm_occupancy,  # may be None: see the docstring
        kernel_name=dominant.name,
        kernel_ms=dominant.total_ms,
        kernel_share=dominant.total_ms / total_ms,
        kernels_named=len(kernels),
        kernels_in_scope=len(eligible),
        kernels_below_min_share=below_min_share,
    )


def flat_hbm_mean(layers: Optional[list[LayerMetrics]]) -> Optional[float]:
    """The duration-weighted mean across ALL kernels — the read that failed.

    Reproduces `parsers/nsight.py::_aggregate_phase` for `hbm_bandwidth_util`.
    r01 never gates on this; it is computed only so a fired diagnosis can show
    the reader what the flat read would have said (0.595 vs 0.795 on a real
    A100 capture). Publishing the contrast is what turns "the rule fired" into "the
    rule fired, and here is the number that used to hide it".
    """
    if not layers:
        return None
    num = den = 0.0
    for layer in layers:
        ms = layer.duration_ms
        if layer.hbm_bandwidth_util is not None and ms is not None and ms > 0:
            num += layer.hbm_bandwidth_util * ms
            den += ms
    return num / den if den > 0 else None


class DecodeMemoryBoundRule(Rule):
    """Decode phase is memory-bound because the batch is too small."""

    rule_id = "r01"
    title = "Decode memory-bound at low batch"
    references = (
        "Williams et al., 'Roofline: An Insightful Visual Performance Model "
        "for Floating-Point Programs and Multiprocessors,' CACM 2009.",
        "Pope et al., 'Efficiently Scaling Transformer Inference,' MLSys 2023.",
        "Kwon et al., 'Efficient Memory Management for Large Language Model "
        "Serving with PagedAttention,' SOSP 2023.",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Read BOTH kernel-derived signals, scoped where the data allows --
        # One selection, one kernel, two legs. Reading them from different
        # kernels is the guard fail-open defect; see the module docstring.
        scoped = apply_kernel_scope(dx.layers, KERNEL_SCOPE)
        collapsed_hbm = dx.decode.hbm_bandwidth_util if dx.decode is not None else None
        collapsed_occ = dx.decode.sm_occupancy if dx.decode is not None else None

        hbm_util = scoped.hbm_util if scoped is not None else collapsed_hbm
        hbm_locus = LOCUS_SCOPED if scoped is not None else LOCUS_COLLAPSED

        if scoped is not None and scoped.sm_occupancy is not None:
            sm_occ: Optional[float] = scoped.sm_occupancy
            occ_locus = LOCUS_SCOPED
        else:
            sm_occ = collapsed_occ
            occ_locus = LOCUS_COLLAPSED

        missing: list[str] = []
        if hbm_util is None:
            # Named for the collapsed field even when `layers` was the locus that
            # came up empty: it is the field a user can actually go supply (the
            # DCGM/live path), and the CLI prints this string as advice.
            missing.append(EV_HBM_COLLAPSED)
        if sm_occ is None:
            missing.append(EV_OCC_COLLAPSED)
        if missing:
            return InsufficientData(missing=tuple(missing))

        # ---- Detection leg --------------------------------------------------
        # The bandwidth reading is the whole of r01's detection power. Strict
        # `>`: the boundary value does not fire.
        if not hbm_util > _HBM_THRESHOLD:
            return Abstention.BELOW_THRESHOLD

        # ---- False-positive guard -------------------------------------------
        # OCCUPANCY_FP_GUARD. Evaluated separately from the detection leg on
        # purpose: it does not add detection power, it removes a wrong answer.
        # A saturated compute-bound kernel can be hot on bandwidth *because it is
        # busy*; telling its operator to raise the batch would be the FP this
        # term exists to prevent. Still a hard, strict gate — and read off
        # the same kernel the detection leg was read off, so idle
        # kernels can no longer dilute the guard into letting that FP through.
        if not sm_occ < _SM_THRESHOLD:
            return Abstention.BELOW_THRESHOLD

        # signal_strength is the DETECTION leg alone. The guard does not grade
        # it (OCCUPANCY_FP_GUARD.grades_confidence is False): it was never shown
        # to carry separation — 0.124 vs 0.170 across the field captures, both far
        # under the gate — so folding "how far under 0.30 we landed" into the
        # score would report confidence the evidence does not support. It
        # previously did exactly that, via a geometric mean of the two legs.
        signal_strength = _clamp01((hbm_util - _HBM_THRESHOLD) / _HBM_SPAN)
        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )
        if THRESHOLDS_UNCALIBRATED:
            confidence = min(confidence, _UNCALIBRATED_CEILING)

        batch_size = dx.batch_size
        # Computed once and threaded through: the cause prose and the evidence
        # must never disagree about what the flat read said.
        flat = flat_hbm_mean(dx.layers) if scoped is not None else None
        cause = self._cause(scoped, hbm_util, sm_occ, occ_locus, flat, batch_size)
        evidence = self._evidence(
            scoped, hbm_util, sm_occ, occ_locus, collapsed_occ, flat, batch_size
        )

        # Completeness is the share of the predicate's two kernel-derived legs
        # that were read at the scoped locus, floored at 0.5 when neither was.
        # The middle rung is real and is worth reporting: a diagnosis whose legs
        # came from two different loci is weaker evidence than one about a single
        # kernel, even when every number in it is present.
        legs_scoped = (hbm_locus == LOCUS_SCOPED) + (occ_locus == LOCUS_SCOPED)
        _GUARD_TAIL = (
            "Strength is the bandwidth leg alone; occupancy is a false-positive "
            "guard against a saturated compute-bound kernel, not a co-equal "
            f"detector, and does not grade this score. Capped at "
            f"{_CONFIDENCE_CEILING}"
        )
        if legs_scoped == 2:
            notes = (
                f"Both legs scoped to the dominant kernel "
                f"('{scoped.kernel_name}', {scoped.kernel_share:.0%} of profiled "
                f"kernel time) — one kernel, one verdict. " + _GUARD_TAIL
            )
            data_completeness = 1.0
        elif legs_scoped == 1:
            notes = (
                f"HBM read scoped to the dominant kernel "
                f"('{scoped.kernel_name}', {scoped.kernel_share:.0%} of profiled "
                f"kernel time), but that kernel carries no occupancy counter, so "
                f"the guard fell back to the COLLAPSED phase aggregate. **The two "
                f"legs describe different loci**: a collapsed occupancy can be "
                f"dragged under the gate by idle kernels while the scoped kernel "
                f"is busy, which makes the guard permissive here. "
                + _GUARD_TAIL
            )
            data_completeness = 0.75
        else:
            notes = (
                "Both legs read from the COLLAPSED phase aggregate — this input "
                "carries no usable per-kernel `layers`, so the kernel scope could "
                "not be applied and both readings are diluted by every kernel in "
                "the capture (scope dilution). " + _GUARD_TAIL
            )
            data_completeness = 0.5
        if THRESHOLDS_UNCALIBRATED:
            notes += (
                f"; thresholds uncalibrated, confidence capped at "
                f"{_UNCALIBRATED_CEILING}"
            )

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=_FIX,
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=data_completeness,
                notes=notes,
            ),
            evidence=evidence,
        )

    # ----------------------------------------------------------------------- #
    # Presentation
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _cause(
        scoped: Optional[ScopedKernelRead],
        hbm_util: float,
        sm_occ: float,
        occ_locus: str,
        flat: Optional[float],
        batch_size: Optional[int],
    ) -> str:
        if scoped is not None:
            occ_clause = (
                f"with SM occupancy {sm_occ:.0%} on that same kernel"
                if occ_locus == LOCUS_SCOPED
                else (
                    f"with SM occupancy {sm_occ:.0%} from the collapsed phase "
                    f"aggregate (that kernel carries no occupancy counter, so the "
                    f"two readings describe different loci)"
                )
            )
            cause = (
                f"HBM bandwidth utilization was {hbm_util:.0%} on "
                f"'{scoped.kernel_name}' — the dominant kernel at "
                f"{scoped.kernel_share:.0%} of profiled kernel time — "
                f"{occ_clause}, placing the kernel in the memory-bound "
                f"regime of the roofline."
            )
            # Surface the contrast only when it changed the verdict — i.e. the
            # flat read would have stayed under the gate. Printing a 2-point gap
            # on every firing would be noise; printing THIS gap is the finding.
            if flat is not None and flat <= _HBM_THRESHOLD:
                cause += (
                    f" A flat duration-weighted mean over every kernel in the same "
                    f"capture reads {flat:.0%} — under the {_HBM_THRESHOLD:.0%} "
                    f"gate — so an unscoped read would have stayed silent here "
                    f"(scope dilution)."
                )
        else:
            cause = (
                f"Decode phase HBM bandwidth utilization was {hbm_util:.0%} with "
                f"SM occupancy {sm_occ:.0%}, placing the kernel in the "
                f"memory-bound regime of the roofline."
            )

        if batch_size is not None:
            cause += (
                f" At batch size {batch_size}, there is insufficient arithmetic "
                f"intensity to hide memory latency."
            )
        return cause

    @staticmethod
    def _evidence(
        scoped: Optional[ScopedKernelRead],
        hbm_util: float,
        sm_occ: float,
        occ_locus: str,
        collapsed_occ: Optional[float],
        flat: Optional[float],
        batch_size: Optional[int],
    ) -> dict[str, object]:
        """Evidence names the LOCUS of every reading, not just its value.

        The two loci get different key names on purpose, on BOTH legs. The
        scope-dilution bug was hard to find precisely because `decode.hbm_bandwidth_util` looked
        like a decode measurement and was in fact an average over every kernel in
        the trace; a reader must never again have to open the parser to find out
        which number they are looking at. The same lesson applies
        to the guard, so occupancy carries the same treatment: a locus key of its
        own, and `read_loci_agree` so a two-locus verdict cannot pass for a
        one-kernel verdict at a glance or in a corpus query.
        """
        if scoped is not None:
            evidence: dict[str, object] = {
                EV_HBM_SCOPED: round(hbm_util, 4),
                "hbm_read_locus": LOCUS_SCOPED,
                "dominant_kernel": scoped.kernel_name,
                "dominant_kernel_time_share": round(scoped.kernel_share, 3),
                "kernels_in_scope": scoped.kernels_in_scope,
                "kernels_below_min_share": scoped.kernels_below_min_share,
            }
            if flat is not None:
                # The number the unscoped rule would have gated on.
                evidence["flat_hbm_mean_all_kernels"] = round(flat, 4)
        else:
            evidence = {
                EV_HBM_COLLAPSED: hbm_util,
                "hbm_read_locus": LOCUS_COLLAPSED,
            }

        if occ_locus == LOCUS_SCOPED:
            evidence[EV_OCC_SCOPED] = round(sm_occ, 4)
            if collapsed_occ is not None:
                # NOT the number the guard gated on — published so the dilution
                # gap is visible in the report. The bandwidth leg publishes the
                # same contrast as `flat_hbm_mean_all_kernels`; this is its
                # occupancy twin, and a large gap between the two is the
                # guard fail-open shape.
                evidence["collapsed_sm_occupancy"] = collapsed_occ
        else:
            evidence[EV_OCC_COLLAPSED] = sm_occ
        evidence["occupancy_read_locus"] = occ_locus
        evidence["read_loci_agree"] = evidence["hbm_read_locus"] == occ_locus
        # Says what the occupancy number *is* in this rule. A reader comparing
        # two r01 reports must not infer that a lower occupancy means a stronger
        # diagnosis: it means the guard cleared, nothing more.
        evidence["occupancy_leg_role"] = OCCUPANCY_LEG_ROLE
        evidence["batch_size"] = batch_size
        return evidence


__all__ = [
    "DecodeMemoryBoundRule",
    "EV_HBM_COLLAPSED",
    "EV_HBM_SCOPED",
    "EV_OCC_COLLAPSED",
    "EV_OCC_SCOPED",
    "FP_GUARDS",
    "FpGuard",
    "KERNEL_SCOPE",
    "KernelScope",
    "LOCUS_COLLAPSED",
    "LOCUS_SCOPED",
    "OCCUPANCY_FP_GUARD",
    "OCCUPANCY_LEG_ROLE",
    "ScopedKernelRead",
    "THRESHOLDS_UNCALIBRATED",
    "apply_kernel_scope",
    "flat_hbm_mean",
]
