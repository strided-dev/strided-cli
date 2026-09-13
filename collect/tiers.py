"""Which live sources can satisfy which rule — the honest live/deep tiering.

Turns "is the live signal rich enough?" into explicit product structure: live
mode runs the rules its connected sources can feed; rules that need a captured
dump (r05 / NCCL → Nsight) are surfaced as such, not silently abstaining every
tick. This is data *about* the rules, kept out of the sacred ``rules/base.py``
contract; ``tests/collect/test_tiers.py`` pins it to the rules' real
INSUFFICIENT_DATA behaviour so it cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

VLLM = "vllm"
DCGM = "dcgm"
NSIGHT = "nsight"


@dataclass(frozen=True)
class RuleTier:
    rule_id: str
    title: str
    live_sources: tuple[str, ...]  # live sources that can feed this rule
    dump_only: bool = False        # needs a captured dump; never fires from live polling
    temporal: bool = False         # observes over a sustained run (watch builds its
                                   # history); can never fire from a one-shot dump


# The banner only describes registered rules; a row for an unregistered rule is
# inert. r06 reads throughput_history, which only the watch loop builds (from the
# vLLM token rate over successive ticks) — so it is a live vLLM rule that needs a
# sustained run before it can fire, never a one-shot dump.
_TIERS: dict[str, RuleTier] = {
    "r01": RuleTier("r01", "Decode memory-bound", (DCGM,)),
    "r02": RuleTier("r02", "Colocation contention", (VLLM,)),
    "r03": RuleTier("r03", "KV cache fragmentation", (VLLM,)),
    "r04": RuleTier("r04", "TP imbalance", (DCGM,)),
    "r05": RuleTier("r05", "NCCL dominance", (), dump_only=True),
    "r06": RuleTier("r06", "Throughput decay", (VLLM,), temporal=True),
    # r07 reads the per-kernel `layers` breakdown only the Nsight parser
    # populates — dump-only like r05; it never fires from live polling.
    "r07": RuleTier("r07", "Attention bottleneck", (), dump_only=True),
    # r08 reads the nsys timeline (`nsys_timeline.steps`) an Nsight Systems
    # capture provides — dump-only like r05/r07; it never fires from live polling,
    # so its per-tick InsufficientData is suppressed in watch.
    "r08": RuleTier("r08", "Prefill↔decode interference", (), dump_only=True),
    "r12": RuleTier("r12", "Queue growth", (VLLM,), temporal=True),
}

# Which live source provides a given schema field — for the "why isn't this
# firing" message: an InsufficientData.missing field maps to the source to enable.
_FIELD_SOURCE: dict[str, str] = {
    "decode.sm_occupancy": DCGM,
    "decode.hbm_bandwidth_util": DCGM,
    "tp_rank_sm_clocks": DCGM,
    "kv_cache_fragmentation": VLLM,
    "kv_cache_util": VLLM,
    "vllm_serving": VLLM,
    "tpot_ms": VLLM,
    "throughput_history": VLLM,
    "layers": NSIGHT,
}


def tier(rule_id: str) -> Optional[RuleTier]:
    return _TIERS.get(rule_id)


def is_dump_only(rule_id: str) -> bool:
    t = _TIERS.get(rule_id)
    return bool(t and t.dump_only)


def is_temporal(rule_id: str) -> bool:
    """Whether a rule observes over a sustained run and so cannot fire one-shot.

    One-shot ``diagnose`` collapses these rules' InsufficientData rows into a
    single quiet pointer at ``watch``; unknown rule ids are not temporal
    (fail loud, not silent).
    """
    t = _TIERS.get(rule_id)
    return bool(t and t.temporal)


def is_surfaced(rule_id: str) -> bool:
    """Whether an INSUFFICIENT_DATA note for this rule should be shown live.

    Dump-only rules (r05) can never fire from live polling, so their per-tick
    insufficiency is noise — suppress it (the startup banner explains them once).
    Unknown rule ids surface by default (fail loud, not silent).
    """
    return not is_dump_only(rule_id)


def source_for_field(field: str) -> Optional[str]:
    """The live source that provides a schema field path, or None if unknown."""
    if field in _FIELD_SOURCE:
        return _FIELD_SOURCE[field]
    return _FIELD_SOURCE.get(field.split(".")[0])


def startup_banner(have_dcgm: bool, registered_ids: tuple[str, ...]) -> str:
    """A one-time honest summary of what live mode can and cannot diagnose.

    Groups the *registered* rules by the source that feeds them, flags whether
    DCGM is connected, and points dump-only rules at ``diagnose``.
    """
    groups: dict[str, list[RuleTier]] = {VLLM: [], DCGM: [], "dump": []}
    for rid in registered_ids:
        t = _TIERS.get(rid)
        if t is None:
            continue
        if t.dump_only:
            groups["dump"].append(t)
        elif DCGM in t.live_sources:
            groups[DCGM].append(t)
        else:
            groups[VLLM].append(t)

    def _names(tiers: list[RuleTier]) -> str:
        return ", ".join(f"{t.rule_id} ({t.title})" for t in tiers)

    lines = ["watch · live rules by data source"]
    if groups[VLLM]:
        lines.append(f"  vLLM /metrics : {_names(groups[VLLM])}")
    if groups[DCGM]:
        status = "connected" if have_dcgm else "not connected, pass --dcgm to enable"
        lines.append(f"  DCGM          : {_names(groups[DCGM])}   [{status}]")
    if groups["dump"]:
        lines.append(
            f"  Nsight (dump) : {_names(groups['dump'])}"
            "   capture, then: strided diagnose --nsight <file>"
        )
    return "\n".join(lines)


__all__ = [
    "RuleTier",
    "VLLM",
    "DCGM",
    "NSIGHT",
    "tier",
    "is_dump_only",
    "is_temporal",
    "is_surfaced",
    "source_for_field",
    "startup_banner",
]
