"""The bridge from a ``Diagnosis`` to a concrete, sizable config change.

The engine's ``Diagnosis`` carries a free-text ``fix`` for a human to read, not a
machine-actionable change. This module is the small, explicit registry that turns a
rule id into: which param to touch, how to size the new value from the current one,
and — critically — the checkable :class:`Prediction` to record *before* acting.

v1 registers exactly one rule end-to-end: **r03 (KV-cache fragmentation)**, whose
vLLM fix is to reduce ``--block-size`` so each sequence's final block wastes fewer
dead slots. New rules are added by registering another :class:`FixSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from rules.base import Diagnosis

from gait.types import (
    Abstained,
    AbstentionReason,
    Comparison,
    Prediction,
    PredictionCheck,
    Snapshot,
)

# r03's firing thresholds, mirrored here as the targets a successful fix should
# bring the metrics back under. Kept in sync with rules/r03_kv_fragmentation.py.
_R03_FRAG_FLOOR = 0.20
_R03_UTIL_FLOOR = 0.80

# vLLM engines this block-size fix applies to. For anything else the recommendation
# is structurally different (adopt paging / different knob), so gait abstains rather
# than apply the wrong fix.
_PAGED_VLLM_ENGINES = ("vllm", "sglang")


@dataclass(frozen=True)
class FixSpec:
    """How to turn one rule's diagnosis into an actionable, predicted change."""

    rule_id: str
    param: str
    rationale: str
    # Gate: may abstain if the snapshot is not a shape this fix applies to.
    applicable: Callable[[Diagnosis, Snapshot], Optional[Abstained]]
    # Size the new value from the current one (read out of the target).
    propose_value: Callable[[Any], Any]
    # The checkable commitment, recorded before acting.
    predict: Callable[[Diagnosis, Snapshot, Any, Any], Prediction]


# --------------------------------------------------------------------------- #
# r03 — KV-cache fragmentation → reduce vLLM --block-size
# --------------------------------------------------------------------------- #

def _r03_applicable(d: Diagnosis, snap: Snapshot) -> Optional[Abstained]:
    if snap.inference_engine not in _PAGED_VLLM_ENGINES:
        return Abstained(
            AbstentionReason.NO_FIX_MAPPING,
            {
                "rule_id": d.rule_id,
                "inference_engine": snap.inference_engine,
                "detail": (
                    "the block-size fix applies to paged vLLM/SGLang engines; "
                    f"snapshot reports inference_engine={snap.inference_engine!r}"
                ),
            },
        )
    return None


def _r03_propose_value(current: Any) -> Any:
    """Halve the block size (floor at 1). Smaller blocks → less internal waste.

    TODO(v2): size from the workload rather than blind halving — e.g. derive a target
    block size from ``seq_len_distribution.mean`` so the new value matches the real
    sequence-length profile instead of always halving.
    """
    try:
        cur = int(current)
    except (TypeError, ValueError):
        return current
    return max(1, cur // 2)


def _r03_predict(d: Diagnosis, snap: Snapshot, current: Any, proposed: Any) -> Prediction:
    return Prediction(
        checks=(
            PredictionCheck(
                field="kv_cache_fragmentation",
                comparison=Comparison.LT,
                threshold=_R03_FRAG_FLOOR,
                description=f"fragmentation falls below the {_R03_FRAG_FLOOR:.0%} firing floor",
            ),
            PredictionCheck(
                field="kv_cache_util",
                comparison=Comparison.LT,
                threshold=_R03_UTIL_FLOOR,
                description=f"utilization eases below {_R03_UTIL_FLOOR:.0%}",
                # Advisory: a softer, second-order effect of the block-size change.
                # A genuine defrag win where the workload legitimately keeps the cache
                # near capacity must not read as NO_CHANGE, so verify reports util but
                # does not gate the verdict on it.
                advisory=True,
            ),
        ),
        summary=(
            f"reducing --block-size {current} → {proposed} should cut KV-cache "
            f"fragmentation below {_R03_FRAG_FLOOR:.0%} and relieve utilization "
            f"below {_R03_UTIL_FLOOR:.0%}"
        ),
    )


_R03 = FixSpec(
    rule_id="r03",
    param="block-size",
    rationale=(
        "KV-cache fragmentation wastes the dead tail of each sequence's final block. "
        "A smaller --block-size shrinks that tail, raising effective cache utilization "
        "without changing the model or the paging backend."
    ),
    applicable=_r03_applicable,
    propose_value=_r03_propose_value,
    predict=_r03_predict,
)


# The registry. Keyed by rule id; one entry per rule gait can actually act on.
FIX_REGISTRY: dict[str, FixSpec] = {
    _R03.rule_id: _R03,
}


def fix_for(rule_id: str) -> Optional[FixSpec]:
    return FIX_REGISTRY.get(rule_id)


__all__ = ["FixSpec", "FIX_REGISTRY", "fix_for"]
