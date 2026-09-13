"""Explicit registry of the diagnostic rules the engine runs.

Why an explicit tuple rather than filesystem auto-discovery:

- Security: importing whatever happens to match ``rules/rNN_*.py`` is an
  arbitrary-code-execution vector the moment an untrusted file lands in that
  directory. An explicit list imports only rules we vetted by name.
- Determinism: the tuple fixes evaluation order identically across machines.
- Inspectability: the entire active rule set is readable in one place.

The cost of an explicit list is the "I forgot to register my rule" failure
mode. That is covered by ``tests/engine/test_registry.py``, which asserts every
``rNN_*.py`` file under ``rules/`` contributes exactly one class here — the
safety net of auto-discovery without its runtime magic.
"""

from __future__ import annotations

from rules.base import Rule
from rules.r01_decode_memory_bound import DecodeMemoryBoundRule
from rules.r02_colocation_contention import ColocationContentionRule
from rules.r03_kv_fragmentation import KvCacheFragmentationRule
from rules.r04_tp_imbalance import TpRankImbalanceRule
from rules.r05_nccl_dominant import NcclCollectiveDominantRule
from rules.r06_throughput_decay import ThroughputDecayRule
from rules.r07_attention_bottleneck import AttentionBottleneckRule
from rules.r08_prefill_decode_interference import PrefillDecodeInterferenceRule
from rules.r12_queue_growth import QueueGrowthRule

# The active rule set, in evaluation order. Append new rules here as they land.
ALL_RULES: tuple[type[Rule], ...] = (
    DecodeMemoryBoundRule,
    ColocationContentionRule,
    KvCacheFragmentationRule,
    TpRankImbalanceRule,
    NcclCollectiveDominantRule,
    ThroughputDecayRule,
    AttentionBottleneckRule,
    PrefillDecodeInterferenceRule,
    QueueGrowthRule,
)


def rule_ids() -> tuple[str, ...]:
    """The ``rule_id`` of every registered rule, in registration order."""
    return tuple(rule.rule_id for rule in ALL_RULES)


__all__ = ["ALL_RULES", "rule_ids"]
