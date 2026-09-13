"""
r03 — KV cache fragmentation.

Fires when KV cache fragmentation is high AND the cache is near capacity, meaning
allocated KV blocks hold dead slots that cap the effective batch size. The fix is
engine-specific: a vLLM/SGLang user already has PagedAttention, so the
recommendation is block-size tuning / a v2 backend / pressure relief — NOT "enable
PagedAttention." That recommendation is reserved for the unknown/non-paged branch.

The signal is an *estimate* of internal fragmentation when derived from
block_size × sequence length (all a vLLM scrape supports — external fragmentation
is not exported, and PagedAttention drives it to ~0 anyway). While
THRESHOLDS_UNCALIBRATED is True, confidence is capped: the 0.20 firing threshold
is loosely anchored to PagedAttention's <4% healthy residual (Kwon et al., SOSP
2023), but the proxy has not been validated against real customer dumps.
"""

from __future__ import annotations

import math

from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput

# --------------------------------------------------------------------------- #
# Firing thresholds — CALIBRATION SEEDS.
# FRAG_MIN is anchored to PagedAttention's <4% healthy residual (a well-managed
# paged cache wastes little); >20% signals block-size mismatch, a legacy kernel,
# or a non-paged path. Fragmentation only costs throughput when the cache is
# under pressure, so firing also requires a pressure signal — but NOT the
# instantaneous kv_cache_usage gauge alone: a real-GPU sweep (2026-06-12) showed
# that gauge sawtooths under churn, so a single customer scrape lands in the
# trough (read 0.54-0.67 while window peaks hit 0.86-0.98) and a hard UTIL_MIN
# gate abstains on genuinely fragmented, pressured workloads. Pressure is
# therefore corroborated by EITHER util > UTIL_MIN OR the cumulative preemption
# rate >= PREEMPT_MILD; the counter is immune to the sawtooth because one scrape
# carries its whole history. Validate on real dumps before trusting the bands;
# until then THRESHOLDS_UNCALIBRATED caps confidence.
# --------------------------------------------------------------------------- #
FRAG_MIN = 0.20
UTIL_MIN = 0.80                 # util now corroborates pressure, not a hard veto
PREEMPT_MILD = 0.01            # >=1% lifetime preemption rate = cache hit capacity
PREEMPT_SEVERE = 0.05          # >=5% = severe pressure (strength normalisation)
# Mean request queue time. The 2026-06-13 real-GPU sweep showed vLLM V1 *queues*
# under capacity pressure (mean queue time ~21ms) while num_preemptions stays 0
# and the util gauge sits in its trough — so queue time is the pressure signal
# that actually moves there. A cumulative histogram *survives* one scrape, but its
# mean is then a LIFETIME average; difference two scrapes (pressure_window="delta")
# to read the current window. Calibration seeds (absolute ms; normalising by
# service time is the calibration follow-up).
QUEUE_TIME_MIN_MS = 10.0       # >=10ms mean queue = scheduler at capacity
QUEUE_TIME_SEVERE_MS = 100.0   # >=100ms = severe (strength normalisation)

# Confidence model constants (see r01 for the geometric-mean rationale).
_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.80
_CONFIDENCE_CEILING = 0.90

THRESHOLDS_UNCALIBRATED = True  # flip to False only after Day-60 calibration
_UNCALIBRATED_CEILING = 0.65    # confidence cap while uncalibrated

_FIX_VLLM = (
    "You already run PagedAttention, so the fix is not to enable it. Fragmentation "
    "this high points to (a) block_size too large for your sequence mix — reduce "
    "`--block-size` (default 16) toward your workload; (b) a legacy "
    "paged-attention-v1 kernel — ensure the FlashAttention / paged-attention-v2 "
    "backend is active; or (c) eviction churn under pressure — check "
    "`vllm:num_preemptions_total` and raise `gpu_memory_utilization` or add "
    "capacity. (Kwon et al., SOSP 2023.)"
)

_FIX_TRTLLM = (
    "Verify the paged KV cache is enabled (`paged_kv_cache=true`) and tune "
    "`tokens_per_block` to your sequence-length distribution. A block size large "
    "relative to your sequences leaves dead slots in every sequence's final block."
)

_FIX_ADOPT_PAGED = (
    "Adopt PagedAttention (vLLM or SGLang) if you are not already using a paged KV "
    "cache. Contiguous KV allocation wastes 60–80% of cache to reservation and "
    "fragmentation; PagedAttention raises effective utilization to >96% and up to "
    "~4x throughput (Kwon et al., SOSP 2023)."
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fix_for_engine(engine: str) -> str:
    if engine in ("vllm", "sglang"):
        return _FIX_VLLM
    if engine == "trt-llm":
        return _FIX_TRTLLM
    return _FIX_ADOPT_PAGED


class KvCacheFragmentationRule(Rule):
    """KV cache fragmentation wastes block capacity and caps the batch size."""

    rule_id = "r03"
    title = "KV cache fragmentation"
    references = (
        "Kwon et al., 'Efficient Memory Management for Large Language Model "
        "Serving with PagedAttention,' SOSP 2023. arXiv:2309.06180.",
        "Zheng et al., 'SGLang: Efficient Execution of Structured Language Model "
        "Programs' (RadixAttention), arXiv:2312.07104.",
        "vLLM Metrics design. https://docs.vllm.ai/en/latest/design/metrics/",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required field: fragmentation is the primary, trusted signal --
        frag = dx.kv_cache_fragmentation
        if frag is None:
            return InsufficientData(missing=("kv_cache_fragmentation",))

        # ---- Pressure evidence (cumulative where possible) ----------------
        # The instantaneous util gauge sawtooths, so it cannot veto on its own.
        # Two CUMULATIVE signals survive a single customer scrape: the preemption
        # rate (engines that evict) and the mean request queue time (engines that
        # queue). vLLM V1 keeps num_preemptions=0 under genuine pressure and
        # queues instead — observed on a real GPU 2026-06-13 — so queue time is
        # the signal that actually moves there. Any one corroborates capacity; a
        # two-scrape window (dx.pressure_window) makes them reflect *current*
        # pressure rather than a dilutable lifetime average.
        util = dx.kv_cache_util
        serving = dx.vllm_serving
        preemption_rate = None
        if (
            serving is not None
            and serving.num_preemptions_total is not None
            and serving.request_success_total
        ):
            preemption_rate = serving.num_preemptions_total / max(
                serving.request_success_total, 1
            )
        queue_ms = dx.queue_time_ms.mean if dx.queue_time_ms is not None else None

        # Need at least one way to assess pressure, else we cannot judge cost.
        if util is None and preemption_rate is None and queue_ms is None:
            return Abstention.INSUFFICIENT_DATA

        util_pressure = util is not None and util > UTIL_MIN
        preempt_pressure = preemption_rate is not None and preemption_rate >= PREEMPT_MILD
        queue_pressure = queue_ms is not None and queue_ms >= QUEUE_TIME_MIN_MS

        # ---- Firing condition: real fragmentation AND corroborated pressure
        if not (frag > FRAG_MIN and (util_pressure or preempt_pressure or queue_pressure)):
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength ----------------------------------------------
        frag_strength = _clamp01((frag - FRAG_MIN) / (1.0 - FRAG_MIN))
        util_strength = (
            _clamp01((util - UTIL_MIN) / (1.0 - UTIL_MIN)) if util is not None else 0.0
        )
        preempt_strength = (
            _clamp01(preemption_rate / PREEMPT_SEVERE) if preemption_rate is not None else 0.0
        )
        queue_strength = (
            _clamp01(queue_ms / QUEUE_TIME_SEVERE_MS) if queue_ms is not None else 0.0
        )
        # These are three views of one condition (cache at capacity), so pressure
        # strength is the strongest, not their product. Geometric mean with frag
        # avoids the quadratic collapse of a plain product (same rationale as r01).
        pressure_strength = max(util_strength, preempt_strength, queue_strength)
        signal_strength = math.sqrt(frag_strength * pressure_strength)

        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )
        # Pressure read from a single scrape of a cumulative metric (queue time,
        # preemption rate) is a LIFETIME average — dilutable by calm history,
        # inflatable by a past burst. Keep such firings capped until a two-scrape
        # window (pressure_window="delta") confirms *current* pressure, even after
        # THRESHOLDS_UNCALIBRATED is flipped off.
        pressure_window = dx.pressure_window or "lifetime"
        lifetime_pressure = pressure_window != "delta"
        if THRESHOLDS_UNCALIBRATED or lifetime_pressure:
            confidence = min(confidence, _UNCALIBRATED_CEILING)

        sources = []
        if queue_pressure:
            sources.append("queue")
        if preempt_pressure:
            sources.append("preemptions")
        if util_pressure:
            sources.append("utilization")
        pressure_source = "+".join(sources)

        # ---- Cause --------------------------------------------------------
        block_clause = ""
        if dx.kv_block_size is not None:
            block_clause = f" at block size {dx.kv_block_size}"
        clause_parts = []
        if queue_pressure:
            clause_parts.append(f"mean queue time {queue_ms:.0f}ms")
        if preempt_pressure:
            clause_parts.append(f"a {preemption_rate:.1%} preemption rate")
        if util_pressure:
            clause_parts.append(f"utilization {util:.0%}")
        pressure_clause = ", ".join(clause_parts)
        cause = (
            f"KV cache fragmentation was {frag:.0%} under memory pressure ({pressure_clause}): "
            f"roughly {frag:.0%} of allocated KV cache slots{block_clause} held no useful "
            f"tokens while the cache ran near capacity, capping the effective batch size."
        )

        notes = (
            "geometric mean of fragmentation and pressure strength "
            f"(pressure_source={pressure_source}, pressure_window={pressure_window})"
        )
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"
        elif lifetime_pressure:
            notes += (
                f"; lifetime pressure proxy (single scrape), confidence capped at "
                f"{_UNCALIBRATED_CEILING}"
            )

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=_fix_for_engine(dx.inference_engine),
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=self._data_completeness(dx),
                notes=notes,
            ),
            evidence={
                "kv_cache_fragmentation": round(frag, 4),
                "kv_cache_util": round(util, 4) if util is not None else None,
                "preemption_rate": round(preemption_rate, 4) if preemption_rate is not None else None,
                "queue_time_ms": round(queue_ms, 1) if queue_ms is not None else None,
                "pressure_source": pressure_source,
                "pressure_window": pressure_window,
                "inference_engine": dx.inference_engine,
                "kv_block_size": dx.kv_block_size,
                "signal_strength": round(signal_strength, 2),
            },
        )

    @staticmethod
    def _data_completeness(dx: DiagnosisInput) -> float:
        """Fraction of optional corroborating fields the rule had available."""
        optional = [
            dx.kv_block_size is not None,
            dx.inference_engine != "unknown",
            dx.kv_num_blocks_used is not None,
            dx.seq_len_distribution is not None,
        ]
        return sum(optional) / len(optional)
