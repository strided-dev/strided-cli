"""
r05 — NCCL collective dominates step time.

Fires when NCCL collective communication (all-reduce / all-gather) eats a
disproportionate fraction of step time — `nccl_time_pct` above a topology-relative
bar. In tensor parallelism every rank blocks on a collective after each parallel
matmul; when those collectives, rather than compute, gate the step, the fix is a
network / topology review (link path, NCCL algo/protocol, TP degree), NOT a kernel
tweak.

Two design choices, both deliberate (see spec §Design):

1. **Self-guard against r04.** `nccl_time_pct` from Nsight is the local rank's time
   *inside* the NCCL kernel — which includes busy-wait. So a high value can be the
   downstream *symptom* of an r04 straggler (this rank idling at the barrier) rather
   than genuine collective cost. r05 reads the raw `tp_rank_sm_clocks` itself and, if
   a clear isolated straggler exists, steps aside so r04 owns the diagnosis. r05
   carries no knowledge of r04's internals: it reuses the shared pure helpers
   `slowness_from_sm_clocks` + `compute_imbalance` from `rules/_stats` (median+MAD
   outlier stats, not cross-rule logic) — it does not import r04.

2. **Topology-aware thresholds.** A static dump cannot tell single-node (NVLink) from
   multi-node (IB/Ethernet); we can only infer *scale* from `tensor_parallel_size` /
   `num_gpus`. High NCCL% is partly inherent on multi-node, so the bar is higher
   there, and confidence is reduced when scale is unknown.

The signal is weak from static dumps and the rule says so: it is a triage flag, not
a network diagnosis (link health and topology mismatch are invisible to a static
kernel trace). Confidence is capped while THRESHOLDS_UNCALIBRATED is True.
"""

from __future__ import annotations

from typing import Literal, Optional

from rules._stats import compute_imbalance, outlier_persists, slowness_from_sm_clocks
from rules.base import Abstention, ConfidenceBreakdown, Diagnosis, Rule, RuleResult
from schema import DiagnosisInput
from schema.diagnosis_input import TpRankSample

# --------------------------------------------------------------------------- #
# Thresholds — CALIBRATION SEEDS, not literature values.
# Domino (arXiv:2409.15241) puts TP communication overhead at 17–45% of step time;
# "LLM Inference Beyond a Single Node" (arXiv:2511.09557) gives empirical NCCL
# fractions. There is no published "fraction that means a fault," so these bands are
# seeds: a single-node NVLink fabric should keep collectives well under ~20%, while
# multi-node fabrics inherently run higher — only ~45%+ is clearly diagnosable there.
# Validate on real multi-rank dumps before trusting the bands; until then
# THRESHOLDS_UNCALIBRATED caps confidence.
# --------------------------------------------------------------------------- #
FIRE_SINGLE_NODE = 0.20      # inferred single-node (NVLink): collectives should be cheap
FIRE_MULTI_NODE = 0.45       # inferred multi-node: high NCCL% is partly inherent
SINGLE_NODE_MAX_GPUS = 8     # TP/num_gpus <= 8 → inferred single-node (one box)

STRAGGLER_LAG = 0.10         # slowest rank >10% over median peer → defer to r04
STRAGGLER_MIN_RANKS = 3      # below this you cannot call one rank an outlier

# Confidence model (mirrors r04's geometric/linear skeleton).
_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.80
_CONFIDENCE_CEILING = 0.90
_UNKNOWN_TOPOLOGY_PENALTY = 0.10   # scale could not be inferred → less trustworthy
THRESHOLDS_UNCALIBRATED = True
_UNCALIBRATED_CEILING = 0.65

TopologyScale = Literal["single_node", "multi_node", "unknown"]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #

def infer_topology_scale(dx: DiagnosisInput) -> TopologyScale:
    """Infer deployment scale from the parallel degree.

    Prefer `tensor_parallel_size` (the count that actually drives the collective),
    fall back to `num_gpus`. <= SINGLE_NODE_MAX_GPUS fits one NVLink box; more
    implies a multi-node fabric. None of either → unknown (handled conservatively).
    """
    degree = dx.tensor_parallel_size if dx.tensor_parallel_size is not None else dx.num_gpus
    if degree is None:
        return "unknown"
    return "single_node" if degree <= SINGLE_NODE_MAX_GPUS else "multi_node"


def _threshold_for(scale: TopologyScale) -> float:
    """Fire threshold for the inferred scale; unknown uses the conservative bar."""
    if scale == "single_node":
        return FIRE_SINGLE_NODE
    return FIRE_MULTI_NODE  # multi_node and unknown both use the higher bar


def straggler_present(
    sm_clocks: Optional[list[float]],
    history: Optional[list[TpRankSample]] = None,
) -> bool:
    """True when one TP rank is materially slower than its peers (r04 territory).

    Reads the same raw `tp_rank_sm_clocks` r04 reads and derives the slowness ratio
    the same way (`slowness_from_sm_clocks`), then reuses `compute_imbalance` from
    `rules/_stats` (pure median+MAD outlier stats) — r05 carries no knowledge of
    r04's internals, only the shared helpers. We gate only on `relative_lag` here —
    a broader guard than r04's firing condition — because r05 should step aside
    whenever *any* plausible straggler could explain the busy-wait, even if it would
    not itself clear r04's stricter z-score gate.

    When a `tp_rank_history` is supplied, the outlier must also *persist* across
    consecutive samples (`outlier_persists`, same shared helper r04 gates on). A
    rank that is slowest on one tick and fastest on the next is a DVFS transient,
    not a straggler — there is nothing for r05 to defer to, so it should speak.
    Deferring anyway would silence a real interconnect finding on the strength of
    a straggler r04 itself declines to claim. Breadth is preserved: we still pass
    only `lag_min`, never r04's stricter z-score bar.

    Note: absent/short/corrupt clocks return False (no rank data → no detectable
    straggler). The caller cannot evaluate the r04 deferral in that case and says so.
    """
    if sm_clocks is None or len(sm_clocks) < STRAGGLER_MIN_RANKS:
        return False
    timings = slowness_from_sm_clocks(sm_clocks)
    if timings is None:
        return False
    stats = compute_imbalance(timings)
    if stats is None or not stats.relative_lag > STRAGGLER_LAG:
        return False
    return outlier_persists(history, stats.slowest_index, lag_min=STRAGGLER_LAG)


def _fix() -> str:
    return (
        "Review the collective communication path and topology — this is not a "
        "kernel tweak. (1) Confirm the collectives actually run on NVLink/NVSwitch "
        "and have not fallen back to PCIe (check `NCCL_P2P_LEVEL`, the NCCL topology "
        "it prints at init). (2) On multi-node, confirm IB/RoCE with GPUDirect RDMA "
        "and tune `NCCL_IB_*`; verify the right HCAs are used. (3) Tune `NCCL_ALGO` "
        "(Ring vs Tree) and `NCCL_PROTO`, and supply an `NCCL_TOPO_FILE` if the auto-"
        "detected topology is wrong. (4) Consider lowering the tensor-parallel degree "
        "or shifting to pipeline / sequence parallelism to shrink the per-layer "
        "collective. (5) Rule out CPU-side stalls: a single stalled host core can "
        "amplify into a cluster-wide busy-wait that is indistinguishable from network "
        "cost in a kernel trace (arXiv:2603.22774). Note: link health and topology "
        "mismatch cannot be confirmed from a static dump — treat this as a triage "
        "flag, not a network diagnosis. (Domino, arXiv:2409.15241.)"
    )


class NcclCollectiveDominantRule(Rule):
    """NCCL collectives, not compute, gate the step; route to a network review."""

    rule_id = "r05"
    title = "NCCL collective dominates step time"
    references = (
        "Wang et al., 'Domino: Eliminating Communication in LLM Training by Generic "
        "Tensor Slicing and Overlapping,' arXiv:2409.15241. (17–45% communication "
        "overhead in tensor parallelism.)",
        "'LLM Inference Beyond a Single Node: From Bottleneck Diagnosis to "
        "Multi-Node Optimization,' arXiv:2511.09557. (Empirical NCCL overhead.)",
        "NVIDIA Collective Communications Library (NCCL) documentation. "
        "https://docs.nvidia.com/deeplearning/nccl/",
        "On CPU-induced stragglers amplifying into barrier busy-wait: arXiv:2603.22774.",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required field ------------------------------------------------
        pct = dx.nccl_time_pct
        if pct is None:
            return Abstention.INSUFFICIENT_DATA

        # ---- Topology + threshold -----------------------------------------
        scale = infer_topology_scale(dx)
        threshold = _threshold_for(scale)

        # ---- Firing condition ---------------------------------------------
        if pct < threshold:
            return Abstention.BELOW_THRESHOLD

        # ---- r04 self-guard ------------------------------------------------
        # A high local NCCL fraction may just be this rank busy-waiting on a slow
        # peer. If an isolated straggler exists, defer — r04 is the specific cause.
        # We read the same raw SM clocks r04 reads (not a schema-stored ratio), and
        # pass the same history, so we only defer to a straggler r04 would itself
        # still claim once its persistence gate has had a say.
        if straggler_present(dx.tp_rank_sm_clocks, dx.tp_rank_history):
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength ----------------------------------------------
        signal_strength = _clamp01((pct - threshold) / (1.0 - threshold))

        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )
        if scale == "unknown":
            confidence -= _UNKNOWN_TOPOLOGY_PENALTY
        if THRESHOLDS_UNCALIBRATED:
            confidence = min(confidence, _UNCALIBRATED_CEILING)
        if confidence < _CONFIDENCE_FLOOR:
            return Abstention.BELOW_THRESHOLD

        # ---- Build the diagnosis ------------------------------------------
        scale_phrase = {
            "single_node": "inferred single-node (NVLink)",
            "multi_node": "inferred multi-node",
            "unknown": "an unknown",
        }[scale]

        cause = (
            f"NCCL collective communication consumed {pct:.0%} of step time, above "
            f"the ~{threshold:.0%} expected for {scale_phrase} topology. The "
            f"all-reduce / all-gather collectives — not compute — are gating the "
            f"step: in tensor parallelism every rank blocks on a collective after "
            f"each parallel matmul, so when the collective dominates, throughput is "
            f"bound by the interconnect rather than the GPUs."
        )

        notes = (
            f"topology={scale}; threshold={threshold:.0%}; "
            "linear signal in NCCL fraction above the topology bar"
        )
        if scale == "unknown":
            notes += f"; scale unknown, confidence reduced by {_UNKNOWN_TOPOLOGY_PENALTY}"
        if dx.tp_rank_sm_clocks is None:
            # No rank data: the r04 self-guard could not be evaluated, so we cannot
            # rule out that this NCCL% is a straggler's busy-wait. Say so explicitly.
            notes += "; no tp_rank_sm_clocks — r04 straggler deferral not evaluated"
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=_fix(),
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=self._data_completeness(dx),
                notes=notes,
            ),
            evidence={
                "nccl_time_pct": round(pct, 3),
                "threshold_used": threshold,
                "inferred_topology": scale,
                "tensor_parallel_size": dx.tensor_parallel_size,
                "num_gpus": dx.num_gpus,
                # False = self-guard ran and ruled a straggler out; None = could not
                # be evaluated (no SM clocks) — *not evaluated* is not *ruled out*.
                "straggler_suspected": False if dx.tp_rank_sm_clocks is not None else None,
                "signal_strength": round(signal_strength, 2),
            },
        )

    @staticmethod
    def _data_completeness(dx: DiagnosisInput) -> float:
        """Fraction of corroborators that pin down topology and rule out a straggler."""
        optional = [
            dx.tensor_parallel_size is not None,
            dx.num_gpus is not None,
            dx.tp_rank_sm_clocks is not None,   # lets the r04 self-guard actually run
        ]
        return sum(optional) / len(optional)
